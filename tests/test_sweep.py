import argparse

import pandas as pd
import pytest

from swo.model import ResizeRecord
from swo.sweep import (
    _append_csv,
    _budget_list,
    _metric_columns,
    _model_label,
    _pending_budgets,
    _sample_rows,
    _slug,
    _summary_row,
)

TASK = "seedbench_2_plus"


def logged_sample(doc_id: int) -> dict:
    """A logged sample shaped like lmms-eval's, including a dict-valued metric."""
    return {
        "doc_id": doc_id,
        "doc": {"question": "..."},
        "target": "D",
        "arguments": [],
        "resps": [["D"]],
        "filtered_resps": ["D"],
        "doc_hash": "abc",
        "exact_match": 1.0,
        f"{TASK}_Chart": {"pred": "D", "answer": "D", "question_id": doc_id},
    }


def test_scalar_metrics_become_columns():
    assert _metric_columns(logged_sample(0))["exact_match"] == 1.0


def test_dict_metrics_are_flattened_so_correctness_survives():
    columns = _metric_columns(logged_sample(3))

    assert columns[f"{TASK}_Chart.pred"] == "D"
    assert columns[f"{TASK}_Chart.answer"] == "D"
    assert columns[f"{TASK}_Chart.question_id"] == 3


def test_bookkeeping_keys_are_not_treated_as_metrics():
    columns = _metric_columns(logged_sample(0))

    assert not {"doc", "doc_id", "target", "resps", "filtered_resps", "doc_hash"} & set(columns)


def test_sample_rows_carry_pixels_and_response():
    results = {"samples": {TASK: [logged_sample(0)]}}
    records = {(TASK, 0): ResizeRecord(n_images=1, original_px=640_000, sent_px=1_936)}

    (row,) = _sample_rows(results, TASK, 2_000, records)

    assert row["budget"] == 2_000
    assert row["response"] == "D"
    assert (row["n_images"], row["original_px"], row["sent_px"]) == (1, 640_000, 1_936)


def test_sample_rows_fail_loudly_when_nothing_was_logged():
    with pytest.raises(RuntimeError, match="no logged samples"):
        _sample_rows({"samples": {}}, TASK, 2_000, {})


def test_summary_row_uses_lmms_eval_aggregates():
    results = {"results": {TASK: {"alias": TASK, f"{TASK}_all,none": 0.42, f"{TASK}_all_stderr,none": "N/A"}}}
    records = {(TASK, 0): ResizeRecord(1, 640_000, 1_936), (TASK, 1): ResizeRecord(1, 320_000, 1_936)}

    row = _summary_row(results, TASK, 2_000, records)

    assert row[f"{TASK}_all,none"] == 0.42
    assert "alias" not in row
    assert (row["n_docs"], row["mean_original_px"], row["mean_sent_px"]) == (2, 480_000, 1_936)


def test_append_csv_creates_then_appends(tmp_path):
    path = tmp_path / "samples.csv"

    _append_csv(pd.DataFrame([{"budget": 2_000, "doc_id": 0}]), path)
    _append_csv(pd.DataFrame([{"budget": 12_500, "doc_id": 0}]), path)

    assert list(pd.read_csv(path)["budget"]) == [2_000, 12_500]


def test_append_csv_realigns_shuffled_columns(tmp_path):
    path = tmp_path / "samples.csv"
    _append_csv(pd.DataFrame([{"budget": 2_000, "doc_id": 7}]), path)

    _append_csv(pd.DataFrame([{"doc_id": 9, "budget": 12_500}]), path)

    assert list(pd.read_csv(path)["doc_id"]) == [7, 9]


def test_append_csv_rejects_a_new_column_rather_than_shifting_rows(tmp_path):
    path = tmp_path / "samples.csv"
    _append_csv(pd.DataFrame([{"budget": 2_000}]), path)

    with pytest.raises(ValueError, match="surprise"):
        _append_csv(pd.DataFrame([{"budget": 12_500, "surprise": 1}]), path)


def test_all_budgets_pending_without_an_existing_file(tmp_path):
    assert _pending_budgets([2_000, 12_500], tmp_path / "missing.csv") == [2_000, 12_500]


def test_completed_budgets_are_skipped(tmp_path):
    path = tmp_path / "samples.csv"
    pd.DataFrame([{"budget": 2_000}, {"budget": 2_000}]).to_csv(path, index=False)

    assert _pending_budgets([2_000, 12_500], path) == [12_500]


@pytest.mark.parametrize(
    ("backend", "backend_args", "expected"),
    [
        ("vllm", {"model": "Qwen/Qwen3-VL-8B"}, "Qwen/Qwen3-VL-8B"),
        ("hf", {"pretrained": "org/name"}, "org/name"),
        ("vllm", None, "vllm"),
    ],
)
def test_runs_are_labelled_by_the_model_behind_them(backend, backend_args, expected):
    assert _model_label(backend, backend_args) == expected


def test_slug_is_path_safe():
    assert _slug("Qwen/Qwen3-VL-8B-Instruct") == "Qwen-Qwen3-VL-8B-Instruct"


def test_budget_list_parses_a_comma_separated_list():
    assert _budget_list("2000,12500, 800000") == [2_000, 12_500, 800_000]


@pytest.mark.parametrize("raw", ["", "0", "-5", "2000,0"])
def test_budget_list_rejects_non_positive_values(raw):
    with pytest.raises((argparse.ArgumentTypeError, ValueError)):
        _budget_list(raw)
