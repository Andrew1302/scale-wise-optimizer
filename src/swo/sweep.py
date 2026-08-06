"""Evaluate a benchmark once per image resolution budget, into one tidy table.

    python -m swo.sweep --task seedbench_2_plus --budgets 2000,100000,800000 \
        --backend vllm --backend-args '{"model": "Qwen/Qwen3-VL-8B-Instruct"}'

Writes ``samples.csv`` (one row per document per budget) and ``summary.csv``
(one row per lmms-eval invocation). ``samples.csv`` is the input to the
clustering step.

Long runs should pass ``--chunk-size``: each budget is then evaluated in
document-range slices, and every finished slice is on disk before the next one
starts. Resuming is just re-running the same command — the rows already in
``samples.csv`` say where to pick up.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from lmms_eval.api.model import lmms
from loguru import logger

from swo.model import BudgetedModel
from swo.probe import ProbeBackend

SAMPLES_FILE = "samples.csv"
SUMMARY_FILE = "summary.csv"

#: Logged-sample keys that are bookkeeping rather than task metrics.
_NON_METRIC_KEYS = frozenset(
    {"doc", "doc_id", "target", "arguments", "resps", "filtered_resps", "token_counts", "doc_hash", "input_media"}
)


def run_sweep(
    task: str,
    budgets: Iterable[int],
    backend: str | lmms,
    backend_args: dict | None = None,
    output_dir: Path | None = None,
    run_name: str | None = None,
    limit: int | None = None,
    chunk_size: int | None = None,
    gen_kwargs: str | None = None,
) -> Path:
    """Run ``task`` once per budget and return the directory holding the results.

    The backend is constructed once and reused across budgets, so a local engine
    loads its weights a single time. Work already in ``samples.csv`` is skipped,
    so an interrupted sweep resumes where it stopped — including part-way
    through a budget when ``chunk_size`` is set.
    """
    budgets = list(budgets)
    run_dir = Path(output_dir or Path("benchmarks") / task) / (run_name or _slug(_model_label(backend, backend_args)))
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_path, summary_path = run_dir / SAMPLES_FILE, run_dir / SUMMARY_FILE

    total = _recorded_total(summary_path, limit)
    done = _rows_per_budget(samples_path)
    if total is not None and all(done.get(budget, 0) >= total for budget in budgets):
        logger.info(f"nothing to do: every budget already has {total} rows in {samples_path}")
        return run_dir

    # Imported here, and only once we know there is work: it pulls in torch and
    # the task registry, and constructing the backend loads model weights.
    from lmms_eval.evaluator import simple_evaluate

    model = BudgetedModel(backend=backend, resolution_budget=budgets[0], backend_args=backend_args)

    for budget in budgets:
        model.resolution_budget = budget
        offset = done.get(budget, 0)
        while total is None or offset < total:
            started = time.monotonic()
            results = simple_evaluate(
                model=model,
                tasks=[task],
                limit=_chunk_limit(chunk_size, limit, total, offset),
                offset=offset,
                log_samples=True,
                gen_kwargs=gen_kwargs,
                # The response cache keys on prompt text only (lmms_eval.caching.response_cache),
                # so enabling it would replay one budget's answers for every other budget.
                use_cache=None,
            )
            if results is None:  # non-zero rank under distributed evaluation
                return run_dir

            elapsed = time.monotonic() - started
            total = _total_docs(results, task, limit)
            records = model.pop_records()
            rows = _sample_rows(results, task, budget, offset, records)
            if rows:
                _append_csv(pd.DataFrame(rows), samples_path)
                summary = _summary_row(results, task, budget, offset, total, records, elapsed)
                _append_csv(pd.DataFrame([summary]), summary_path)
                offset += len(rows)
                logger.info(f"budget={budget}: {offset}/{total if total is not None else '?'} docs → {samples_path}")

            # With a known total the loop condition ends the budget. Without one,
            # a short slice is the only signal that the documents ran out.
            if not rows or chunk_size is None or (total is None and len(rows) < chunk_size):
                break

    if summary_path.exists():
        logger.info(f"\n{pd.read_csv(summary_path).to_string(index=False)}")
    return run_dir


def _chunk_limit(chunk_size: int | None, limit: int | None, total: int | None, offset: int) -> int | None:
    """Documents to ask for in one invocation.

    The last slice of a budget must be trimmed to what is left, or it overshoots
    an explicit ``--limit`` and writes more rows than the run asked for.
    """
    if chunk_size is None:
        return limit
    return chunk_size if total is None else max(0, min(chunk_size, total - offset))


def _rows_per_budget(samples_path: Path) -> dict[int, int]:
    """How many documents each budget has already produced — the resume cursor."""
    if not samples_path.exists():
        return {}
    counts = pd.read_csv(samples_path, usecols=["budget"])["budget"].value_counts()
    return {int(budget): int(count) for budget, count in counts.items()}


def _recorded_total(summary_path: Path, limit: int | None) -> int | None:
    """The task's document count, as recorded by an earlier run."""
    if not summary_path.exists() or "n_docs_total" not in pd.read_csv(summary_path, nrows=0).columns:
        return None
    totals = pd.read_csv(summary_path, usecols=["n_docs_total"])["n_docs_total"].dropna()
    if totals.empty:
        return None
    return min(int(totals.max()), limit) if limit else int(totals.max())


def _total_docs(results: dict, task: str, limit: int | None) -> int | None:
    original = results.get("n-samples", {}).get(task, {}).get("original")
    if original is None:
        return None
    return min(int(original), limit) if limit else int(original)


def _sample_rows(results: dict, task: str, budget: int, offset: int, records: dict) -> list[dict]:
    samples = results.get("samples", {}).get(task)
    if samples is None:
        raise RuntimeError(f"no logged samples for task {task!r} at budget {budget}, offset {offset}")

    return [
        {
            "task": task,
            "budget": budget,
            "doc_id": sample["doc_id"],
            "target": sample["target"],
            "response": _first(sample["filtered_resps"]),
            **_pixel_columns(records.get((task, sample["doc_id"]))),
            **_token_columns(sample),
            **_metric_columns(sample),
        }
        for sample in samples
    ]


def _token_columns(sample: dict) -> dict:
    """Per-sample token usage, as lmms-eval reported it.

    Which counts exist depends on the backend: local HF backends report
    ``output_tokens`` only, while API-style ones also report ``input_tokens``
    — the number that includes vision tokens, and so the direct evidence that a
    pixel budget reached the model.
    """
    counts = _first(sample.get("token_counts"))
    return dict(counts) if isinstance(counts, dict) else {}


def _summary_row(
    results: dict,
    task: str,
    budget: int,
    offset: int,
    total: int | None,
    records: dict,
    elapsed_s: float,
) -> dict:
    """One row per lmms-eval invocation, with that invocation's own aggregates.

    With ``--chunk-size`` a budget spans several rows; a budget-level metric is
    their ``n_docs``-weighted mean. ``n_docs_total`` is the task's full document
    count, which is what lets a later run know where each budget should end.
    ``elapsed_s`` is wall clock for the invocation, model load excluded only for
    the first budget — compare budgets, not absolutes.
    """
    pixels = list(records.values())
    aggregates = {key: value for key, value in results["results"][task].items() if key != "alias" and _is_scalar(value)}
    return {
        "budget": budget,
        "offset": offset,
        "n_docs": len(pixels),
        "n_docs_total": total,
        "elapsed_s": round(elapsed_s, 2),
        "mean_original_px": _mean(record.original_px for record in pixels),
        "mean_sent_px": _mean(record.sent_px for record in pixels),
        **aggregates,
    }


def _metric_columns(sample: dict) -> dict:
    """Task metrics as flat columns.

    Metrics whose value is a dict — seedbench_2_plus reports
    ``{pred, answer, question_id}`` per subject — expand to ``<metric>.<key>``,
    which is where per-sample correctness lives.
    """
    columns = {}
    for key, value in sample.items():
        if key in _NON_METRIC_KEYS:
            continue
        if _is_scalar(value):
            columns[key] = value
        elif isinstance(value, dict):
            columns.update({f"{key}.{sub}": item for sub, item in value.items() if _is_scalar(item)})
    return columns


def _pixel_columns(record) -> dict:
    return dataclasses.asdict(record) if record else {}


def _append_csv(frame: pd.DataFrame, path: Path) -> None:
    """Append `frame`, keeping the existing header's column order.

    Appending a frame whose columns drifted would silently misalign every value,
    so a genuinely new column is an error rather than a shifted row.
    """
    if not path.exists():
        frame.to_csv(path, index=False)
        return
    columns = list(pd.read_csv(path, nrows=0).columns)
    if unknown := [column for column in frame.columns if column not in columns]:
        raise ValueError(f"{path} has no columns for {unknown}; delete it or pass a fresh --run-name")
    frame.reindex(columns=columns).to_csv(path, mode="a", header=False, index=False)


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else value


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _model_label(backend: str | lmms, backend_args: dict | None) -> str:
    """Identify the VLM behind a run, so results are never anonymous."""
    if isinstance(backend, lmms):
        return type(backend).__name__
    args = backend_args or {}
    return str(args.get("model") or args.get("pretrained") or backend)


def _slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-")


def _budget_list(raw: str) -> list[int]:
    budgets = [int(part) for part in raw.split(",") if part.strip()]
    if not budgets or any(budget < 1 for budget in budgets):
        raise argparse.ArgumentTypeError(f"budgets must be a comma-separated list of positive integers, got {raw!r}")
    return budgets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="lmms-eval task name, e.g. seedbench_2_plus")
    parser.add_argument("--budgets", required=True, type=_budget_list, help="comma-separated max pixels per image")
    parser.add_argument("--backend", default="vllm", help="lmms-eval chat model id used for inference")
    parser.add_argument("--backend-args", type=json.loads, default=None, help="JSON object of backend kwargs")
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to benchmarks/<task>")
    parser.add_argument("--run-name", default=None, help="subdirectory name; defaults to the backend model")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N documents")
    parser.add_argument(
        "--chunk-size", type=int, default=None, help="documents per invocation; enables resume mid-budget"
    )
    parser.add_argument(
        "--gen-kwargs", default=None, help="override the task's generation_kwargs, e.g. max_new_tokens=512"
    )
    parser.add_argument("--dry-run", action="store_true", help="exercise the resize path with no model loaded")
    args = parser.parse_args(argv)

    if args.dry_run and args.backend_args:
        parser.error("--backend-args has no effect with --dry-run; no model is loaded")

    run_dir = run_sweep(
        task=args.task,
        budgets=args.budgets,
        backend=ProbeBackend() if args.dry_run else args.backend,
        backend_args=None if args.dry_run else args.backend_args,
        output_dir=args.output_dir,
        run_name=args.run_name,
        limit=args.limit,
        chunk_size=args.chunk_size,
        gen_kwargs=args.gen_kwargs,
    )
    logger.info(f"results in {run_dir}")


if __name__ == "__main__":
    main()
