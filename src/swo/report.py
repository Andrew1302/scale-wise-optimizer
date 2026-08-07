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


#: What tasks call the prediction they scored, most specific first. seedbench_2_plus
#: emits ``.pred``; mmmu_pro runs its own parser and emits ``.parsed_pred``.
PRED_SUFFIXES = (".parsed_pred", ".pred")


def prediction_column(frame: pd.DataFrame, metric: str) -> str | None:
    """The column holding ``metric``'s prediction, whatever the task named it."""
    return next((f"{metric}{suffix}" for suffix in PRED_SUFFIXES if f"{metric}{suffix}" in frame.columns), None)


def metric_names(frame: pd.DataFrame) -> list[str]:
    """Metrics carrying a prediction/answer pair, i.e. the scorable ones."""
    candidates = [column[: -len(".answer")] for column in frame.columns if column.endswith(".answer")]
    return [name for name in candidates if prediction_column(frame, name)]


def overall_metric(frame: pd.DataFrame) -> str | None:
    """The task-wide metric, preferred over per-subject ones that cover a subset."""
    names = metric_names(frame)
    return next((name for name in names if name.endswith("_all")), names[0] if names else None)


def correctness(frame: pd.DataFrame, metric: str) -> pd.Series:
    """Per-row correctness for one metric: True, False, or NA where not scored.

    Null-checked before any string coercion — pandas does not render missing
    values as the literal ``"nan"``, so a string comparison would let them
    through, and an unanswered document must count as wrong rather than match.
    """
    expected_raw = frame[f"{metric}.answer"]
    predicted_raw = frame[prediction_column(frame, metric)]
    predicted = predicted_raw.where(predicted_raw.notna(), "").astype(str).str.strip().str.lower()
    expected = expected_raw.astype(str).str.strip().str.lower()
    return (predicted == expected).where(expected_raw.notna())


def _accuracy(group: pd.DataFrame) -> float | None:
    """Fraction correct over the documents this metric actually scores."""
    metric = overall_metric(group)
    if metric is None or not group[f"{metric}.answer"].notna().any():
        return None
    return correctness(group, metric).mean()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="a sweep's output directory")
    parser.add_argument("--markdown", action="store_true", help="emit a markdown table")
    args = parser.parse_args(argv)

    report = build_report(args.run_dir)
    print(report.to_markdown(index=False) if args.markdown else report.to_string(index=False))


if __name__ == "__main__":
    main()
