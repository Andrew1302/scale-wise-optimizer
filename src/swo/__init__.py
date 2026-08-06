"""Scale-wise optimizer: per-cluster image resolution policies for VLM benchmarks.

Nothing is re-exported here on purpose. ``swo.registry`` is imported by lmms-eval
*while* ``lmms_eval.models`` is still executing its module body, which runs this
file too; pulling ``swo.model`` in here would import the half-initialised
``lmms_eval.models`` and fail. Import from the submodules directly.
"""
