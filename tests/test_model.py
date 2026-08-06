import pytest
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.utils import simple_parse_args_string
from PIL import Image

from swo.model import BudgetedModel, _as_kwargs

TASK, SPLIT = "fake_task", "test"
PROMPT = "What colour is the sky?\nA. blue\nB. red"
BUDGET = 10_000


class RecordingBackend(lmms):
    """Captures the messages it is asked to build, so tests can inspect them."""

    is_simple = False

    def __init__(self) -> None:
        super().__init__()
        self.built: list[list[dict]] = []

    def generate_until(self, requests):
        for request in requests:
            _, doc_to_messages, _, doc_id, task, split = request.arguments
            self.built.append(doc_to_messages(self.task_dict[task][split][doc_id]))
        return [f"answer-{index}" for index in range(len(requests))]

    def generate_until_multi_round(self, requests):
        raise NotImplementedError

    def loglikelihood(self, requests):
        raise NotImplementedError


def make_doc(*media) -> dict:
    """A minimal task document, from ``(content_type, url)`` pairs."""
    return {"media": list(media)}


def picture(width: int, height: int) -> tuple[str, Image.Image]:
    return ("image", Image.new("RGB", (width, height)))


def doc_to_messages(doc: dict) -> list[dict]:
    # Cached on the document on purpose: real tasks can hand back the same objects on
    # every call, so this is what catches the wrapper mutating what it was given.
    if "messages" not in doc:
        content = [{"type": kind, "url": url} for kind, url in doc["media"]]
        content.append({"type": "text", "text": PROMPT})
        doc["messages"] = [{"role": "user", "content": content}]
    return doc["messages"]


def evaluate(model: BudgetedModel, docs: dict) -> list[str]:
    """Drive `model` over `docs` (doc_id -> document) the way lmms-eval would."""
    model.task_dict = {TASK: {SPLIT: docs}}
    requests = [
        Instance(
            request_type="generate_until",
            arguments=("ctx", doc_to_messages, {"max_new_tokens": 16}, doc_id, TASK, SPLIT),
            idx=index,
            metadata={"task": TASK, "doc_id": doc_id, "repeats": 1},
        )
        for index, doc_id in enumerate(docs)
    ]
    return model.generate_until(requests)


def images_of(messages: list[dict]) -> list:
    return [item["url"] for message in messages for item in message["content"] if item["type"] == "image"]


@pytest.fixture
def backend() -> RecordingBackend:
    return RecordingBackend()


@pytest.fixture
def model(backend: RecordingBackend) -> BudgetedModel:
    return BudgetedModel(backend=backend, resolution_budget=BUDGET)


def test_oversized_image_is_downscaled(model, backend):
    evaluate(model, {0: make_doc(picture(800, 600))})

    (sent,) = images_of(backend.built[0])
    assert sent.width * sent.height <= BUDGET
    assert sent.width / sent.height == pytest.approx(800 / 600, rel=0.01)


def test_prompt_text_is_untouched(model, backend):
    evaluate(model, {0: make_doc(picture(800, 600))})

    texts = [item["text"] for item in backend.built[0][0]["content"] if item["type"] == "text"]
    assert texts == [PROMPT]


def test_image_within_budget_is_passed_through_unchanged(model, backend):
    original = Image.new("RGB", (50, 50))
    evaluate(model, {0: make_doc(("image", original))})

    assert images_of(backend.built[0]) == [original]


def test_non_image_content_is_left_alone(model, backend):
    evaluate(model, {0: make_doc(("video", "clip.mp4"), picture(800, 600))})

    videos = [item for item in backend.built[0][0]["content"] if item["type"] == "video"]
    assert videos == [{"type": "video", "url": "clip.mp4"}]


def test_response_order_and_count_are_preserved(model):
    docs = {doc_id: make_doc(picture(200, 200)) for doc_id in range(3)}

    assert evaluate(model, docs) == ["answer-0", "answer-1", "answer-2"]


def test_records_report_pixels_before_and_after(model):
    evaluate(model, {7: make_doc(picture(800, 600))})

    record = model.pop_records()[(TASK, 7)]
    assert record.n_images == 1
    assert record.original_px == 800 * 600
    assert record.sent_px <= BUDGET
    assert model.pop_records() == {}, "records should reset once collected"


def test_documents_are_not_mutated_between_budgets(model):
    """A sweep re-runs the same documents; resizing must never touch the task's own data."""
    docs = {0: make_doc(picture(800, 600))}
    evaluate(model, docs)

    model.resolution_budget = 10**6
    evaluate(model, docs)

    assert images_of(docs[0]["messages"])[0].size == (800, 600)


def test_path_within_budget_stays_a_path(model, backend, tmp_path):
    path = tmp_path / "small.png"
    Image.new("RGB", (50, 50)).save(path)

    evaluate(model, {0: make_doc(("image", str(path)))})

    assert images_of(backend.built[0]) == [str(path)]


def test_oversized_path_is_decoded_and_downscaled(model, backend, tmp_path):
    path = tmp_path / "big.png"
    Image.new("RGB", (800, 600)).save(path)

    evaluate(model, {0: make_doc(("image", str(path)))})

    (sent,) = images_of(backend.built[0])
    assert isinstance(sent, Image.Image)
    assert sent.width * sent.height <= BUDGET


def test_task_dict_is_forwarded_to_the_backend(model, backend):
    model.task_dict = {TASK: {SPLIT: {}}}

    assert backend.task_dict is model.task_dict


@pytest.mark.parametrize("value", ["100000", 1e5, 100_000.4])
def test_budget_accepts_the_types_the_cli_produces(model, value):
    model.resolution_budget = value

    assert model.resolution_budget == 100_000


@pytest.mark.parametrize("value", [0, -1])
def test_rejects_non_positive_budget(backend, value):
    with pytest.raises(ValueError, match="resolution_budget"):
        BudgetedModel(backend=backend, resolution_budget=value)


def test_rejects_a_simple_backend(backend):
    backend.is_simple = True

    with pytest.raises(TypeError, match="chat"):
        BudgetedModel(backend=backend, resolution_budget=BUDGET)


def test_rejects_backend_args_alongside_a_built_backend(backend):
    with pytest.raises(ValueError, match="backend_args"):
        BudgetedModel(backend=backend, resolution_budget=BUDGET, backend_args={"model": "x"})


def test_loglikelihood_is_unsupported(model):
    with pytest.raises(NotImplementedError):
        model.loglikelihood([])


def test_cli_model_args_survive_the_lmms_eval_parser():
    """`--model_args` is split on commas, so nested JSON is the fragile part."""
    parsed = simple_parse_args_string(
        'backend=vllm,resolution_budget=100000,backend_args={"model":"Qwen/Qwen3-VL-8B","mm_processor_kwargs":{"min_pixels":256}}'
    )

    assert parsed["backend"] == "vllm"
    assert parsed["resolution_budget"] == 100_000
    assert _as_kwargs(parsed["backend_args"]) == {
        "model": "Qwen/Qwen3-VL-8B",
        "mm_processor_kwargs": {"min_pixels": 256},
    }


def test_backend_args_reject_a_non_mapping():
    with pytest.raises(TypeError, match="backend_args"):
        _as_kwargs(["model=x"])
