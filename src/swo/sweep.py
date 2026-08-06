"""Evaluate a benchmark once per image resolution budget, into one tidy table.

    python -m swo.sweep --task seedbench_2_plus --budgets 2000,100000,800000 \
        --backend vllm --backend-args '{"model": "Qwen/Qwen3-VL-8B-Instruct"}'

Writes ``samples.csv`` (one row per document per budget) and ``summary.csv``
(per-budget means). ``samples.csv`` is the input to the clustering step.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
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
) -> Path:
    """Run ``task`` once per budget and return the directory holding the results.

    The backend is constructed once and reused across budgets, so a local engine
    loads its weights a single time. Budgets already present in an existing
    ``samples.csv`` are skipped, so an interrupted sweep resumes where it stopped.
    """
    from lmms_eval.evaluator import simple_evaluate  # deferred: pulls in torch and the task registry

    budgets = list(budgets)
    run_dir = Path(output_dir or Path("benchmarks") / task) / (run_name or _slug(_model_label(backend, backend_args)))
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_path = run_dir / SAMPLES_FILE

    pending = _pending_budgets(budgets, samples_path)
    if not pending:
        logger.info(f"nothing to do: all {len(budgets)} budgets already in {samples_path}")
        return run_dir

    model = BudgetedModel(backend=backend, resolution_budget=pending[0], backend_args=backend_args)
    for budget in pending:
        model.resolution_budget = budget
        results = simple_evaluate(
            model=model,
            tasks=[task],
            limit=limit,
            log_samples=True,
            # The response cache keys on prompt text only (lmms_eval.caching.response_cache),
            # so enabling it would replay one budget's answers for every other budget.
            use_cache=None,
        )
        if results is None:  # non-zero rank under distributed evaluation
            continue
        records = model.pop_records()
        rows = _sample_rows(results, task, budget, records)
        _append_csv(pd.DataFrame(rows), samples_path)
        _append_csv(pd.DataFrame([_summary_row(results, task, budget, records)]), run_dir / SUMMARY_FILE)
        logger.info(f"budget={budget}: appended {len(rows)} rows to {samples_path}")

    logger.info(f"\n{pd.read_csv(run_dir / SUMMARY_FILE).to_string(index=False)}")
    return run_dir


def _pending_budgets(budgets: list[int], samples_path: Path) -> list[int]:
    if not samples_path.exists():
        return budgets
    done = set(pd.read_csv(samples_path, usecols=["budget"])["budget"])
    pending = [budget for budget in budgets if budget not in done]
    if skipped := sorted(set(budgets) - set(pending)):
        logger.info(f"skipping budgets already in {samples_path}: {skipped}")
    return pending


def _sample_rows(results: dict, task: str, budget: int, records: dict) -> list[dict]:
    samples = results.get("samples", {}).get(task)
    if not samples:
        raise RuntimeError(f"no logged samples for task {task!r} at budget {budget}")

    return [
        {
            "task": task,
            "budget": budget,
            "doc_id": sample["doc_id"],
            "target": sample["target"],
            "response": _first(sample["filtered_resps"]),
            **_pixel_columns(records.get((task, sample["doc_id"]))),
            **_metric_columns(sample),
        }
        for sample in samples
    ]


def _summary_row(results: dict, task: str, budget: int, records: dict) -> dict:
    """One row of aggregate metrics, straight from lmms-eval rather than re-derived."""
    pixels = list(records.values())
    aggregates = {key: value for key, value in results["results"][task].items() if key != "alias" and _is_scalar(value)}
    return {
        "budget": budget,
        "n_docs": len(pixels),
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
    )
    logger.info(f"results in {run_dir}")


if __name__ == "__main__":
    main()
