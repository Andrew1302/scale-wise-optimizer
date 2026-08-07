import argparse

import pandas as pd
import pytest

from swo.model import ResizeRecord
from swo.sweep import (
    _append_csv,
    _budget_list,
    _chunk_limit,
    _metric_columns,
    _model_label,
    _recorded_total,
    _rows_per_budget,
    _sample_rows,
    _slug,
    _summary_row,
    _total_docs,
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

    (row,) = _sample_rows(results, TASK, 2_000, 0, records)

    assert row["budget"] == 2_000
    assert row["response"] == "D"
    assert (row["n_images"], row["original_px"], row["sent_px"]) == (1, 640_000, 1_936)


def test_token_counts_are_captured_as_columns():
    """lmms-eval logs these per sample; they are the evidence a budget reached the model."""
    sample = {**logged_sample(0), "token_counts": [{"input_tokens": 1234, "output_tokens": 2}]}
    results = {"samples": {TASK: [sample]}}

    (row,) = _sample_rows(results, TASK, 2_000, 0, {})

    assert (row["input_tokens"], row["output_tokens"]) == (1234, 2)


def test_missing_token_counts_are_not_fatal():
    """Local HF backends report only some counts, and some report none."""
    results = {"samples": {TASK: [{**logged_sample(0), "token_counts": [None]}]}}

    (row,) = _sample_rows(results, TASK, 2_000, 0, {})

    assert "input_tokens" not in row


def test_sample_rows_fail_loudly_when_nothing_was_logged():
    with pytest.raises(RuntimeError, match="no logged samples"):
        _sample_rows({"samples": {}}, TASK, 2_000, 0, {})


def test_sample_rows_allow_an_empty_slice_past_the_end():
    assert _sample_rows({"samples": {TASK: []}}, TASK, 2_000, 9_999, {}) == []


def test_summary_row_uses_lmms_eval_aggregates():
    results = {"results": {TASK: {"alias": TASK, f"{TASK}_all,none": 0.42, f"{TASK}_all_stderr,none": "N/A"}}}
    records = {(TASK, 0): ResizeRecord(1, 640_000, 1_936), (TASK, 1): ResizeRecord(1, 320_000, 1_936)}

    row = _summary_row(results, TASK, 2_000, 500, 2_277, records, 12.5)

    assert row[f"{TASK}_all,none"] == 0.42
    assert "alias" not in row
    assert (row["budget"], row["offset"], row["n_docs"], row["n_docs_total"]) == (2_000, 500, 2, 2_277)
    assert (row["mean_original_px"], row["mean_sent_px"]) == (480_000, 1_936)


@pytest.mark.parametrize(
    ("limit", "expected"),
    [(None, 2_277), (100, 100), (9_999, 2_277)],
)
def test_total_docs_respects_an_explicit_limit(limit, expected):
    results = {"n-samples": {TASK: {"original": 2_277, "effective": 2_277}}}

    assert _total_docs(results, TASK, limit) == expected


def test_total_docs_is_unknown_when_lmms_eval_does_not_report_it():
    assert _total_docs({}, TASK, None) is None


@pytest.mark.parametrize(
    ("chunk_size", "limit", "total", "offset", "expected"),
    [
        (None, 12, 12, 0, 12),  # unchunked: one invocation for the whole budget
        (None, None, 2_277, 0, None),  # unchunked, no limit: everything
        (5, 12, 12, 0, 5),  # a full slice
        (5, 12, 12, 10, 2),  # the last slice is trimmed to what is left
        (500, None, 2_277, 2_000, 277),
        (5, 12, None, 10, 5),  # total unknown: ask for a full slice, detect the short one
        (5, 12, 12, 12, 0),  # nothing left
    ],
)
def test_chunk_limit_never_overshoots_the_run(chunk_size, limit, total, offset, expected):
    assert _chunk_limit(chunk_size, limit, total, offset) == expected


# -- resume ---------------------------------------------------------------


def test_no_rows_yet_means_every_budget_starts_at_zero(tmp_path):
    assert _rows_per_budget(tmp_path / "missing.csv") == {}


def test_row_counts_are_the_resume_cursor(tmp_path):
    path = tmp_path / "samples.csv"
    pd.DataFrame([{"budget": 2_000}] * 500 + [{"budget": 12_500}] * 120).to_csv(path, index=False)

    assert _rows_per_budget(path) == {2_000: 500, 12_500: 120}


def test_total_is_recovered_from_an_earlier_run(tmp_path):
    path = tmp_path / "summary.csv"
    pd.DataFrame([{"budget": 2_000, "n_docs_total": 2_277}]).to_csv(path, index=False)

    assert _recorded_total(path, limit=None) == 2_277
    assert _recorded_total(path, limit=100) == 100


def test_total_is_unknown_before_the_first_run(tmp_path):
    assert _recorded_total(tmp_path / "missing.csv", limit=None) is None


def test_a_summary_without_the_total_column_is_tolerated(tmp_path):
    path = tmp_path / "summary.csv"
    pd.DataFrame([{"budget": 2_000}]).to_csv(path, index=False)

    assert _recorded_total(path, limit=None) is None


# -- csv appends ----------------------------------------------------------


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


# -- labelling ------------------------------------------------------------


@pytest.mark.parametrize(
    ("backend", "backend_args", "expected"),
    [
        ("vllm", {"model": "Qwen/Qwen3-VL-8B"}, "Qwen/Qwen3-VL-8B"),
        ("hf", {"pretrained": "org/name"}, "org/name"),
        # Served backends name the weights differently; missing this labelled a
        # whole vLLM sweep "async_openai" instead of the model it ran.
        ("async_openai", {"model_version": "Qwen/Qwen3.5-4B", "base_url": "http://x"}, "Qwen/Qwen3.5-4B"),
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
