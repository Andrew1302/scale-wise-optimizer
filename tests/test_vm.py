import dataclasses
import json
import re
import shlex

import pytest

import swo.vm as vm_module
from swo.vm import MCQ_SYSTEM_PROMPT, SERVER_SESSION, VMS, resolve_vm, serve_command


def test_every_vm_derives_its_paths_from_one_base():
    vm = VMS["vm03"]

    assert vm.workdir == "/media/vm03/ssd1T/andrew/scale-wise-optimizer"
    assert vm.run_dir.startswith(vm.workdir)


def test_caches_are_redirected_off_the_home_partition():
    exports = VMS["vm03"].env_exports()

    for name in ("HF_HOME", "UV_CACHE_DIR", "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TMPDIR"):
        assert f"export {name}=/media/vm03/ssd1T/andrew" in exports


def test_path_export_expands_on_the_vm():
    """Quoting this one would blank the VM's PATH and lose every core utility."""
    assert 'export PATH="$HOME/.local/bin:$PATH"' in VMS["vm03"].env_exports()


def test_borrowed_machine_is_marked_shared_and_pinned_to_one_gpu():
    c2d = VMS["c2d"]

    assert c2d.shared
    assert "export CUDA_VISIBLE_DEVICES=1" in c2d.env_exports()


@pytest.mark.parametrize("name", ["vm02", "vm03"])
def test_lab_machines_are_not_gpu_pinned(name):
    assert "CUDA_VISIBLE_DEVICES" not in VMS[name].env_exports()


def test_paths_with_spaces_are_quoted():
    vm = dataclasses.replace(VMS["vm03"], base="/media/some dir/andrew")

    assert "export HF_HOME='/media/some dir/andrew/hf_cache'" in vm.env_exports()


def test_environment_overrides_win(monkeypatch):
    monkeypatch.setenv("SWO_VM_HOST", "10.0.0.5")
    monkeypatch.setenv("SWO_VM_PORT", "2222")
    monkeypatch.setenv("SWO_VM_BASE", "/scratch/andrew")

    vm = resolve_vm("vm03")

    assert (vm.host, vm.port) == ("10.0.0.5", 2222)
    assert vm.workdir == "/scratch/andrew/scale-wise-optimizer"
    assert vm.user == "vm03", "unset fields keep the profile's value"


def test_unknown_vm_is_rejected():
    with pytest.raises(SystemExit, match="nope"):
        resolve_vm("nope")


# -- vLLM server ----------------------------------------------------------


def test_pixel_bounds_are_passed_to_the_server():
    """With vLLM the image processor lives server-side, so the floor must go there."""
    command = serve_command("Qwen/Qwen3.5-4B", min_pixels=256, max_pixels=16_777_216)

    assert '--mm-processor-kwargs \'{"min_pixels": 256, "max_pixels": 16777216}\'' in command


def test_defaults_keep_the_budget_binding():
    """The default floor must sit below the smallest budget we ever sweep (2000 px)."""
    assert json.loads(re.search(r"--mm-processor-kwargs '(.*?)'", serve_command("m")).group(1))["min_pixels"] < 2_000


def test_thinking_is_disabled_server_side_by_default():
    """The async_openai backend has no per-request switch, so the server must carry it."""
    assert "--default-chat-template-kwargs '{\"enable_thinking\": false}'" in serve_command("m")


def test_thinking_can_be_turned_back_on():
    assert "--default-chat-template-kwargs '{\"enable_thinking\": true}'" in serve_command("m", enable_thinking=True)


def test_gpu_memory_is_an_explicit_knob():
    assert "--gpu-memory-utilization 0.35" in serve_command("m", gpu_memory_utilization=0.35)


def test_tensor_parallel_lets_a_model_span_cards():
    assert "--tensor-parallel-size 2" in serve_command("m", tensor_parallel_size=2)


@pytest.mark.parametrize(
    ("kwargs", "absent"),
    [({}, "--max-model-len"), ({"extra": ""}, "--extra")],
)
def test_optional_flags_are_omitted_when_unset(kwargs, absent):
    assert absent not in serve_command("m", **kwargs)


def test_model_names_are_quoted():
    assert shlex.split(serve_command("org/model name"))[2] == "org/model name"


def test_system_prompt_is_the_one_validated_for_compliance():
    """Shared with the README/skill recipe; changing it changes measured accuracy."""
    assert "exactly one character" in MCQ_SYSTEM_PROMPT
    assert "Do not explain" in MCQ_SYSTEM_PROMPT


def test_server_session_is_separate_from_the_sweep_session():
    """The server must outlive any single sweep, so it cannot share a session name."""
    assert SERVER_SESSION != "swo"


def test_session_targets_match_exactly():
    """tmux -t matches on prefix, so plain 'swo' also resolves 'swo_vllm'."""
    assert vm_module._target("swo") == "=swo"


def test_the_sweep_session_never_resolves_the_server(monkeypatch):
    """Without exact matching, `stop` on the default name would kill the model server."""
    asked = []
    monkeypatch.setattr(vm_module, "_succeeds", lambda vm, cmd: asked.append(cmd) or False)

    vm_module.session_alive(VMS["vm03"], "swo")

    assert "=swo" in asked[0], "a prefix target would also match swo_vllm"


def test_deploy_never_prunes_another_step_s_packages(monkeypatch):
    """A plain `uv sync` would uninstall the vllm extra the running server needs."""
    sent = []
    monkeypatch.setattr(vm_module, "_start_session", lambda *a, **k: sent.append(a[2]))
    monkeypatch.setattr(vm_module, "session_alive", lambda *a, **k: False)
    monkeypatch.setattr(vm_module, "_session_exit_code", lambda *a, **k: 0)
    monkeypatch.setattr(vm_module.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": b""})())

    vm_module.deploy(VMS["vm03"], extras=("vllm",))

    assert "uv sync --inexact --extra vllm" in sent[0]
