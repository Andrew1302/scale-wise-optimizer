"""Registers ``BudgetedModel`` with lmms-eval.

``ModelRegistryV2.load_entrypoint_manifests`` reads the ``lmms_eval.models``
entry-point group (declared in pyproject.toml) at import time, so this makes
``lmms-eval --model resolution_budget`` work with no change to the lmms-eval
installation. The class path stays a string: lmms-eval imports it lazily, which
is what keeps this module free of the circular import described in ``swo``.
"""

from lmms_eval.models.registry_v2 import ModelManifest

MODEL_ID = "resolution_budget"

MANIFEST = ModelManifest(model_id=MODEL_ID, chat_class_path="swo.model.BudgetedModel")
