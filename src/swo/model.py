"""An lmms-eval model that caps image resolution before delegating to a real VLM."""

from __future__ import annotations

import dataclasses
import json
import threading
from typing import Any

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from loguru import logger
from PIL import Image

from swo.downscale import fit_to_budget


@dataclasses.dataclass(frozen=True)
class ResizeRecord:
    """What the wrapper actually sent for one document."""

    n_images: int
    original_px: int
    sent_px: int


class BudgetedModel(lmms):
    """Downscales every image to a pixel budget, then delegates to a chat backend.

    Wraps any chat-capable lmms-eval model. Text, video and audio content pass
    through untouched, so the prompt the backend sees is byte-identical to an
    unbudgeted run — only the pixels change.

    Caveat: this controls what we *send*. Most VLM processors re-resize on their
    own side, and a ``min_pixels`` floor (Qwen defaults to 200704) will upscale
    small images straight back up, silently erasing the effect of a low budget.
    Push the backend's pixel floor below the smallest budget you intend to
    sweep, and check the per-run summary logged by this class.
    """

    is_simple = False

    def __init__(
        self,
        backend: str | lmms,
        resolution_budget: int | str,
        backend_args: dict | str | None = None,
        batch_size: Any = None,
        max_batch_size: Any = None,
        device: Any = None,
    ) -> None:
        """
        Args:
            backend: an lmms-eval model id, or an already-constructed model. Passing
                an instance lets a sweep load the weights once and reuse them.
            resolution_budget: maximum total pixels (width * height) per image.
            backend_args: constructor kwargs for the backend, as a dict or a JSON
                object string (which is how they survive the lmms-eval CLI parser).
            batch_size, max_batch_size, device: injected by lmms-eval's
                ``create_from_arg_string``; forwarded to the backend when set.
        """
        if isinstance(backend, lmms):
            if backend_args is not None:
                raise ValueError("backend_args cannot be combined with an already-constructed backend.")
            self.backend = backend
        else:
            # Assigned before super().__init__(), which sets task_dict — a property
            # this class forwards to the backend.
            self.backend = _build_backend(
                str(backend),
                backend_args,
                batch_size=batch_size,
                max_batch_size=max_batch_size,
                device=device,
            )
        if self.backend.is_simple:
            raise TypeError(
                f"{type(self.backend).__name__} is a simple (non-chat) model. "
                "BudgetedModel wraps chat models; pass a chat-capable backend id."
            )

        super().__init__()
        self.resolution_budget = resolution_budget
        self._records: dict[tuple[str, Any], ResizeRecord] = {}
        self._records_lock = threading.Lock()

    # -- configuration ----------------------------------------------------

    @property
    def resolution_budget(self) -> int:
        return self._resolution_budget

    @resolution_budget.setter
    def resolution_budget(self, value: Any) -> None:
        # CLI args arrive as strings or floats ("1e5" parses to 100000.0), never as ints.
        budget = int(float(value))
        if budget < 1:
            raise ValueError(f"resolution_budget must be >= 1, got {value!r}")
        self._resolution_budget = budget

    # -- delegation to the backend ----------------------------------------

    @property
    def task_dict(self) -> dict:
        return self.backend.task_dict

    @task_dict.setter
    def task_dict(self, value: dict) -> None:
        self.backend.task_dict = value

    @property
    def rank(self) -> int:
        return self.backend.rank

    @property
    def world_size(self) -> int:
        return self.backend.world_size

    def set_cache_hook(self, cache_hook) -> None:
        super().set_cache_hook(cache_hook)
        self.backend.set_cache_hook(cache_hook)

    def clean(self) -> None:
        """Deliberately a no-op — the backend must outlive one evaluation.

        ``lmms_eval.evaluator.evaluate`` calls ``clean()`` after *every*
        invocation, and the base implementation deletes every ``nn.Module``
        attribute. A sweep calls the evaluator once per budget against this same
        instance, so delegating would delete the weights after the first budget
        and break the second. Memory is reclaimed when the process exits.
        """

    # -- generation --------------------------------------------------------

    def generate_until(self, requests: list[Instance]) -> list[Any]:
        responses = self.backend.generate_until([self._with_budget(r) for r in requests])
        self._log_summary()
        return responses

    def generate_until_multi_round(self, requests: list[Instance]) -> list[Any]:
        responses = self.backend.generate_until_multi_round([self._with_budget(r) for r in requests])
        self._log_summary()
        return responses

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            f"{type(self).__name__} wraps generation-only chat backends; loglikelihood tasks are unsupported."
        )

    # -- measurement -------------------------------------------------------

    def pop_records(self) -> dict[tuple[str, Any], ResizeRecord]:
        """Return the per-document resize records collected so far, and reset them."""
        with self._records_lock:
            records, self._records = self._records, {}
        return records

    def _log_summary(self) -> None:
        with self._records_lock:
            records = list(self._records.values())
        images = sum(record.n_images for record in records)
        if not images:
            return
        original_px = sum(record.original_px for record in records)
        sent_px = sum(record.sent_px for record in records)
        logger.info(
            f"resolution_budget={self.resolution_budget}: {len(records)} docs, {images} images, "
            f"mean {original_px // images} -> {sent_px // images} px/image "
            f"({1 - sent_px / original_px:.1%} fewer pixels sent)"
        )

    # -- request rewriting -------------------------------------------------

    def _with_budget(self, request: Instance) -> Instance:
        """Copy ``request`` with a ``doc_to_messages`` that downscales images.

        Rewriting the request rather than the document leaves lmms-eval's own
        state untouched and defers the resize to whenever the backend actually
        builds the prompt.
        """
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments

        def budgeted_doc_to_messages(doc: dict, **kwargs: Any):
            return self._shrink(doc_to_messages(doc, **kwargs), task, doc_id)

        return dataclasses.replace(
            request,
            arguments=(ctx, budgeted_doc_to_messages, gen_kwargs, doc_id, task, split),
        )

    def _shrink(self, payload: Any, task: str, doc_id: Any) -> Any:
        # Rounds after the first return (messages, terminal, previous_output,
        # previous_round_info), with messages None on the terminal round.
        if isinstance(payload, tuple):
            messages, *rest = payload
            return (self._shrink_messages(messages, task, doc_id) if messages else messages, *rest)
        return self._shrink_messages(payload, task, doc_id)

    def _shrink_messages(self, messages: list[dict], task: str, doc_id: Any) -> list[dict]:
        budget = self.resolution_budget
        n_images = original_px = sent_px = 0
        shrunk = []

        for message in messages:
            content = []
            for item in message["content"]:
                if item.get("type") != "image":
                    content.append(item)
                    continue
                url, before, after = _fit_url_to_budget(item["url"], budget)
                n_images += 1
                original_px += before
                sent_px += after
                content.append(item if url is item["url"] else {**item, "url": url})
            # Rebuilt, never mutated: a sweep calls doc_to_messages again for the next
            # budget and the task may hand back references into its own cached document.
            shrunk.append({**message, "content": content})

        with self._records_lock:  # chat backends build requests in a thread pool
            self._records[(task, doc_id)] = ResizeRecord(n_images, original_px, sent_px)
        return shrunk


def _fit_url_to_budget(url: Any, budget_px: int) -> tuple[Any, int, int]:
    """Return ``(url_for_the_backend, original_px, sent_px)``.

    Path urls already within budget are returned unchanged so lmms-eval's base64
    path cache keeps hitting; they are decoded only when a resize is actually due.
    """
    if isinstance(url, Image.Image):
        resized = fit_to_budget(url, budget_px)
        return resized, _pixels(url), _pixels(resized)

    with Image.open(url) as probe:
        original_px = _pixels(probe)
        if original_px <= budget_px:
            return url, original_px, original_px
        resized = fit_to_budget(probe, budget_px)
    return resized, original_px, _pixels(resized)


def _pixels(image: Image.Image) -> int:
    width, height = image.size
    return width * height


def _build_backend(name: str, backend_args: dict | str | None, **injected: Any) -> lmms:
    # Deferred import: lmms_eval.models imports this package while its own module
    # body is still executing (see swo/__init__.py).
    from lmms_eval.models import get_model

    kwargs = _as_kwargs(backend_args)
    kwargs.update({key: value for key, value in injected.items() if value is not None})
    return get_model(name)(**kwargs)


def _as_kwargs(backend_args: dict | str | None) -> dict:
    if backend_args is None:
        return {}
    if isinstance(backend_args, dict):
        return dict(backend_args)
    if isinstance(backend_args, str):
        # The CLI parser passes JSON objects through as raw strings (lmms_eval.utils.handle_arg_string).
        return json.loads(backend_args)
    raise TypeError(f"backend_args must be a dict or a JSON object string, got {type(backend_args).__name__}")
