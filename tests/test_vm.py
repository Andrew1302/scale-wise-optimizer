import dataclasses

import pytest

from swo.vm import VMS, resolve_vm


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
