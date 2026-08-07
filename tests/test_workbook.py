import json

import pandas as pd
import pytest

from swo.workbook import build_workbook

TASK = "seedbench_2_plus"


def sample(budget: int, doc_id: int, pred: str, answer: str, subject: str = "Chart") -> dict:
    """One sweep row; per-subject metric columns are sparse, as lmms-eval emits them."""
    row = {
        "task": TASK,
        "budget": budget,
        "doc_id": doc_id,
        "target": answer,
        "response": pred,
        f"{TASK}_all.pred": pred,
        f"{TASK}_all.answer": answer,
    }
    row[f"{TASK}_{subject}.pred"] = pred
    row[f"{TASK}_{subject}.answer"] = answer
    return row


@pytest.fixture
def run_dir(tmp_path):
    pd.DataFrame(
        [
            sample(2_000, 0, "A", "A", "Chart"),
            sample(2_000, 1, "B", "C", "Map"),
            sample(12_500, 0, "A", "A", "Chart"),
            sample(12_500, 1, "C", "C", "Map"),
        ]
    ).to_csv(tmp_path / "samples.csv", index=False)
    return tmp_path


def test_sheets_match_the_legacy_workbook(run_dir):
    sheets = build_workbook(run_dir)

    assert list(sheets) == ["Consolidated_Scores", "Aggregated_Scores", "Resolution_2000", "Resolution_12500"]


def test_consolidated_is_one_row_per_document_with_a_column_per_budget(run_dir):
    consolidated = sheets_of(run_dir)["Consolidated_Scores"]

    assert list(consolidated.columns) == ["doc_id", "target", "2000", "12500"]
    assert list(consolidated["doc_id"]) == [0, 1]
    assert list(consolidated["2000"]) == [1, 0]
    assert list(consolidated["12500"]) == [1, 1]


def test_targets_survive_the_pivot(run_dir):
    assert list(sheets_of(run_dir)["Consolidated_Scores"]["target"]) == ["A", "C"]


def test_aggregated_reports_each_subject_and_the_overall(run_dir):
    aggregated = sheets_of(run_dir)["Aggregated_Scores"]

    assert list(aggregated.columns) == [
        "Resolution",
        f"{TASK}_Chart",
        f"{TASK}_Map",
        f"{TASK}_all",
    ], "subjects sorted, overall last — the legacy column order"
    assert list(aggregated["Resolution"]) == [2_000, 12_500]
    assert list(aggregated[f"{TASK}_all"]) == [0.5, 1.0]


def test_a_subject_is_scored_only_over_its_own_documents(run_dir):
    aggregated = sheets_of(run_dir)["Aggregated_Scores"].set_index("Resolution")

    assert aggregated.loc[2_000, f"{TASK}_Chart"] == 1.0, "doc 0 only; doc 1 is a Map question"
    assert aggregated.loc[2_000, f"{TASK}_Map"] == 0.0


def test_per_resolution_sheets_have_the_legacy_columns(run_dir):
    sheet = sheets_of(run_dir)["Resolution_2000"]

    assert list(sheet.columns) == ["doc_id", "target", "predicted", "input", "score"]
    assert list(sheet["predicted"]) == ["A", "B"]
    assert list(sheet["score"]) == [1, 0]


def test_prompts_are_joined_when_a_samples_log_is_available(run_dir, tmp_path):
    prompts = tmp_path / "test_questions.jsonl"
    prompts.write_text(json.dumps({"doc_id": 0, "input": "What colour?"}) + "\n", encoding="utf-8")

    sheet = build_workbook(run_dir, prompts)["Resolution_2000"]

    assert sheet.loc[0, "input"] == "What colour?"
    assert pd.isna(sheet.loc[1, "input"]), "documents absent from the log are left blank"


def test_missing_prompts_file_is_not_fatal(run_dir, tmp_path):
    sheet = build_workbook(run_dir, tmp_path / "absent.jsonl")["Resolution_2000"]

    assert sheet["input"].isna().all()


def test_unanswered_documents_score_zero(tmp_path):
    pd.DataFrame([sample(2_000, 0, None, "A")]).to_csv(tmp_path / "samples.csv", index=False)

    assert sheets_of(tmp_path)["Consolidated_Scores"]["2000"].iloc[0] == 0


def test_a_sweep_without_scorable_metrics_is_rejected(tmp_path):
    pd.DataFrame([{"budget": 2_000, "doc_id": 0, "target": "A"}]).to_csv(tmp_path / "samples.csv", index=False)

    with pytest.raises(SystemExit, match="score"):
        build_workbook(tmp_path)


def sheets_of(run_dir):
    return build_workbook(run_dir)
