"""Summarise a sweep: accuracy, tokens and latency per resolution budget.

    python -m swo.report benchmarks/seedbench_2_plus/Qwen-Qwen3.5-4B

Reads the ``samples.csv`` / ``summary.csv`` a sweep wrote and prints one row per
budget. The token columns are the evidence that a budget actually reached the
model — if they do not move with the budget, the backend re-resized the images
and the sweep measured nothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from swo.sweep import SAMPLES_FILE, SUMMARY_FILE

#: Token columns lmms-eval may report, in the order we display them.
TOKEN_COLUMNS = ("input_tokens", "output_tokens", "reasoning_tokens")


def build_report(run_dir: Path) -> pd.DataFrame:
    """One row per budget, ordered by budget."""
    samples = pd.read_csv(run_dir / SAMPLES_FILE)
    report = samples.groupby("budget").apply(_budget_row, include_groups=False).reset_index()

    summary_path = run_dir / SUMMARY_FILE
    if summary_path.exists() and "elapsed_s" in pd.read_csv(summary_path, nrows=0).columns:
        # Summed across chunks, so a chunked run reports the whole budget's time.
        elapsed = pd.read_csv(summary_path).groupby("budget")["elapsed_s"].sum()
        report["elapsed_s"] = report["budget"].map(elapsed).round(1)
        report["s_per_doc"] = (report["elapsed_s"] / report["n"]).round(2)

    return report.sort_values("budget").reset_index(drop=True)


def _budget_row(group: pd.DataFrame) -> pd.Series:
    row = {"n": len(group)}

    accuracy = _accuracy(group)
    if accuracy is not None:
        row["accuracy"] = round(accuracy, 4)

    for column in ("original_px", "sent_px", *TOKEN_COLUMNS):
        if column in group and group[column].notna().any():
            row[f"mean_{column}"] = round(group[column].mean(), 1)

    return pd.Series(row)


def _accuracy(group: pd.DataFrame) -> float | None:
    """Fraction correct, from the ``<metric>.pred`` / ``<metric>.answer`` pair.

    Prefers the task's ``_all`` metric when present, since per-subject metrics
    cover only part of the documents.
    """
    pairs = [column[: -len(".pred")] for column in group.columns if column.endswith(".pred")]
    pairs = [name for name in pairs if f"{name}.answer" in group.columns]
    if not pairs:
        return None
    metric = next((name for name in pairs if name.endswith("_all")), pairs[0])

    # Null-checked before any string coercion: pandas does not render missing
    # values as the literal "nan", so a string comparison would let them through
    # — and an unanswered document must count as wrong, not as a match.
    expected_raw = group[f"{metric}.answer"]
    if not expected_raw.notna().any():
        return None
    predicted = group[f"{metric}.pred"].where(group[f"{metric}.pred"].notna(), "").astype(str).str.strip().str.lower()
    expected = expected_raw.astype(str).str.strip().str.lower()
    scored = expected_raw.notna()
    return (predicted[scored] == expected[scored]).mean()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="a sweep's output directory")
    parser.add_argument("--markdown", action="store_true", help="emit a markdown table")
    args = parser.parse_args(argv)

    report = build_report(args.run_dir)
    print(report.to_markdown(index=False) if args.markdown else report.to_string(index=False))


if __name__ == "__main__":
    main()
