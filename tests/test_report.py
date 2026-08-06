import pandas as pd
import pytest

from swo.report import build_report

TASK = "seedbench_2_plus"


def sample(budget: int, doc_id: int, pred: str, answer: str, **extra) -> dict:
    return {
        "task": TASK,
        "budget": budget,
        "doc_id": doc_id,
        "original_px": 640_000,
        "sent_px": min(budget, 640_000),
        f"{TASK}_all.pred": pred,
        f"{TASK}_all.answer": answer,
        **extra,
    }


@pytest.fixture
def run_dir(tmp_path):
    pd.DataFrame(
        [
            sample(2_000, 0, "A", "A", output_tokens=2, input_tokens=40),
            sample(2_000, 1, "B", "A", output_tokens=2, input_tokens=40),
            sample(800_000, 0, "A", "A", output_tokens=3, input_tokens=665),
            sample(800_000, 1, "A", "A", output_tokens=3, input_tokens=665),
        ]
    ).to_csv(tmp_path / "samples.csv", index=False)
    return tmp_path


def test_one_row_per_budget_in_ascending_order(run_dir):
    report = build_report(run_dir)

    assert list(report["budget"]) == [2_000, 800_000]
    assert list(report["n"]) == [2, 2]


def test_accuracy_comes_from_the_pred_answer_pair(run_dir):
    report = build_report(run_dir)

    assert list(report["accuracy"]) == [0.5, 1.0]


def test_token_and_pixel_means_are_reported(run_dir):
    report = build_report(run_dir)

    assert list(report["mean_input_tokens"]) == [40.0, 665.0]
    assert list(report["mean_sent_px"]) == [2_000.0, 640_000.0]


def test_case_and_whitespace_do_not_break_scoring(tmp_path):
    pd.DataFrame([sample(2_000, 0, " a ", "A")]).to_csv(tmp_path / "samples.csv", index=False)

    assert build_report(tmp_path)["accuracy"].iloc[0] == 1.0


def test_latency_is_summed_across_chunks(run_dir):
    pd.DataFrame(
        [
            {"budget": 2_000, "offset": 0, "elapsed_s": 4.0},
            {"budget": 2_000, "offset": 1, "elapsed_s": 6.0},
            {"budget": 800_000, "offset": 0, "elapsed_s": 20.0},
        ]
    ).to_csv(run_dir / "summary.csv", index=False)

    report = build_report(run_dir)

    assert list(report["elapsed_s"]) == [10.0, 20.0]
    assert list(report["s_per_doc"]) == [5.0, 10.0]


def test_missing_summary_is_not_fatal(run_dir):
    assert "elapsed_s" not in build_report(run_dir).columns


def test_an_empty_response_counts_as_wrong_not_as_missing(tmp_path):
    """A model that answered nothing must not be silently excluded from accuracy."""
    pd.DataFrame(
        [
            {**sample(2_000, 0, "A", "A")},
            {**sample(2_000, 1, None, "B")},  # no answer produced
        ]
    ).to_csv(tmp_path / "samples.csv", index=False)

    assert build_report(tmp_path)["accuracy"].iloc[0] == 0.5


def test_report_without_a_metric_pair_still_describes_the_run(tmp_path):
    pd.DataFrame([{"budget": 2_000, "doc_id": 0, "sent_px": 1_936}]).to_csv(tmp_path / "samples.csv", index=False)

    report = build_report(tmp_path)

    assert list(report["n"]) == [1]
    assert "accuracy" not in report.columns
