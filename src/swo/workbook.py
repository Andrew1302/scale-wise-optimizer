"""Export a sweep as the workbook the clustering notebooks already read.

    python -m swo.workbook benchmarks/seedbench_2_plus/Qwen-Qwen3.5-4B

Reproduces the layout of the legacy ``*_resolution_analysis.xlsx`` so the
existing clustering stage can consume a fresh sweep unchanged:

``Consolidated_Scores``
    one row per document — ``doc_id``, ``target``, then one 0/1 column per
    pixel budget. This is the sheet the notebooks actually load.
``Aggregated_Scores``
    one row per budget, with accuracy per task metric (per subject and overall).
``Resolution_<budget>``
    one row per document — ``doc_id``, ``target``, ``predicted``, ``input``,
    ``score``.

``input`` (the rendered prompt) is not something the sweep records, so it is
joined from a samples jsonl when one is available; the notebooks read prompts
from that file anyway.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from swo.report import correctness, metric_names, overall_metric, prediction_column
from swo.sweep import SAMPLES_FILE

WORKBOOK_FILE = "resolution_analysis.xlsx"

#: Where the prompt text lives, if the legacy per-sample log is still around.
PROMPTS_FILE = "test_questions.jsonl"


def build_workbook(run_dir: Path, prompts: Path | None = None) -> dict[str, pd.DataFrame]:
    """The sheets of the legacy workbook, keyed by sheet name."""
    samples = pd.read_csv(run_dir / SAMPLES_FILE)
    metric = overall_metric(samples)
    if metric is None:
        raise SystemExit(f"{run_dir / SAMPLES_FILE} has no <metric>.pred/.answer pair to score")

    samples = samples.assign(score=correctness(samples, metric).fillna(False).astype(int))
    budgets = sorted(samples["budget"].unique())
    inputs = _prompts(prompts)

    sheets = {
        "Consolidated_Scores": _consolidated(samples, budgets),
        "Aggregated_Scores": _aggregated(samples, budgets),
    }
    for budget in budgets:
        sheets[f"Resolution_{budget}"] = _per_resolution(samples, budget, metric, inputs)
    return sheets


def _consolidated(samples: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    """One row per document, one 0/1 column per budget — what the notebooks load."""
    wide = samples.pivot_table(index="doc_id", columns="budget", values="score", aggfunc="first")
    wide = wide.reindex(columns=budgets)
    wide.columns = [str(budget) for budget in budgets]

    targets = samples.drop_duplicates("doc_id").set_index("doc_id")["target"]
    return wide.join(targets).reset_index()[["doc_id", "target", *(str(b) for b in budgets)]]


def _aggregated(samples: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    """One row per budget: accuracy for every task metric, subjects then overall."""
    metrics = sorted(metric_names(samples), key=lambda name: (name.endswith("_all"), name))
    rows = []
    for budget in budgets:
        at_budget = samples[samples["budget"] == budget]
        row = {"Resolution": budget}
        for name in metrics:
            scored = correctness(at_budget, name)
            row[name] = scored.mean() if scored.notna().any() else None
        rows.append(row)
    return pd.DataFrame(rows)


def _per_resolution(samples: pd.DataFrame, budget: int, metric: str, inputs: dict) -> pd.DataFrame:
    at_budget = samples[samples["budget"] == budget].copy()
    at_budget["predicted"] = at_budget[prediction_column(at_budget, metric)]
    at_budget["input"] = at_budget["doc_id"].map(inputs)
    return at_budget[["doc_id", "target", "predicted", "input", "score"]].reset_index(drop=True)


def _prompts(path: Path | None) -> dict:
    """doc_id -> rendered prompt, from a per-sample jsonl. Empty when unavailable."""
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return {record["doc_id"]: record.get("input") for record in records if "doc_id" in record}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="a sweep's output directory")
    parser.add_argument("-o", "--output", type=Path, default=None, help=f"defaults to <run_dir>/{WORKBOOK_FILE}")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help=f"jsonl holding the rendered prompts; defaults to <run_dir>/../{PROMPTS_FILE}",
    )
    args = parser.parse_args(argv)

    prompts = args.prompts if args.prompts is not None else args.run_dir.parent / PROMPTS_FILE
    sheets = build_workbook(args.run_dir, prompts)

    output = args.output or args.run_dir / WORKBOOK_FILE
    with pd.ExcelWriter(output) as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)

    logger.info(f"wrote {output} — sheets: {', '.join(sheets)}")
    print(sheets["Aggregated_Scores"].to_string(index=False))


if __name__ == "__main__":
    main()
