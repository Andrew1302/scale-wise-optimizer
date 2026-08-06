"""A backend that loads no model, for exercising the request path cheaply."""

from __future__ import annotations

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms


class ProbeBackend(lmms):
    """Builds every prompt but runs no inference, answering with a fixed string.

    Wrapped by :class:`~swo.model.BudgetedModel`, this drives the whole resize
    path — task loading, image decoding, downscaling, bookkeeping — without a
    GPU, which is how a budget is validated before spending VM time. Accuracy
    from such a run is meaningless; the pixel counts are the point.
    """

    is_simple = False

    def __init__(self, response: str = "") -> None:
        super().__init__()
        self.response = response

    def generate_until(self, requests: list[Instance]) -> list[str]:
        for request in requests:
            _, doc_to_messages, _, doc_id, task, split = request.arguments
            doc_to_messages(self.task_dict[task][split][doc_id])  # triggers the wrapper's resize
        return [self.response] * len(requests)

    def generate_until_multi_round(self, requests: list[Instance]) -> list[str]:
        raise NotImplementedError(f"{type(self).__name__} only supports single-round generation.")

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError(f"{type(self).__name__} only supports single-round generation.")
