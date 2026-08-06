"""Run a resolution sweep on a lab GPU VM and bring the results back.

    python -m swo.vm run -- --task seedbench_2_plus --budgets 2000,800000 \
        --chunk-size 500 --backend vllm --backend-args '{"model":"Qwen/Qwen3-VL-8B-Instruct"}'

Everything after ``--`` is passed straight to ``swo.sweep`` on the VM, so this
module never has to know about sweep's flags.

``run`` deploys the package, starts the sweep in a detached tmux session, waits,
then fetches results. **Recovering from any failure is re-running the same
command**: the sweep skips the rows already in ``samples.csv``, so a crashed or
killed run resumes instead of starting over. Ctrl-C detaches without stopping
the VM-side run.

Other commands: ``check`` (GPU availability), ``status``, ``logs``, ``fetch``,
``stop``.

These are shared lab machines — read ``.claude/skills/vm-runs/SKILL.md`` before
launching anything on one, especially c2d, which is borrowed.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

#: Paths that carry the sweep. Results are never uploaded — they only come back.
UPLOAD_PATHS = ("src", "pyproject.toml", "uv.lock", "README.md")

#: Where the sweep writes on the VM, relative to the workdir; fetched verbatim.
RESULTS_PATH = "benchmarks"

#: A GPU holding more than this is treated as in use by someone else.
BUSY_GPU_MIB = 4096


@dataclasses.dataclass(frozen=True)
class VM:
    """A lab machine. Paths mirror the lab's other tooling so caches are shared."""

    user: str
    host: str
    port: int
    key: str
    base: str
    shared: bool = False
    cuda_visible_devices: str | None = None

    @property
    def workdir(self) -> str:
        return f"{self.base}/scale-wise-optimizer"

    @property
    def run_dir(self) -> str:
        return f"{self.workdir}/.run"

    def env_exports(self) -> str:
        """Redirect every cache onto the big partition — /home is small and vllm fills it."""
        env = {
            "HF_HOME": f"{self.base}/hf_cache",
            "UV_CACHE_DIR": f"{self.base}/uv_cache",
            "XDG_CACHE_HOME": f"{self.base}/.cache",
            "TRITON_HOME": self.base,
            "TRITON_CACHE_DIR": f"{self.base}/.triton/cache",
            "TMPDIR": f"{self.base}/tmp",
            "FLASHINFER_WORKSPACE_BASE": self.base,
        }
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        exports = [f"export {key}={shlex.quote(value)}" for key, value in env.items()]
        # Deliberately unquoted — this one has to expand on the VM, not here.
        # uv installs itself to ~/.local/bin, which a non-login shell misses.
        exports.append('export PATH="$HOME/.local/bin:$PATH"')
        return "\n".join(exports)


VMS = {
    "vm03": VM(user="vm03", host="143.107.165.250", port=5022, key="~/.ssh/vm_key", base="/media/vm03/ssd1T/andrew"),
    "vm02": VM(user="vm02", host="143.107.165.249", port=5022, key="~/.ssh/vm_key", base="/media/vm02/ssd1T/andrew"),
    # Borrowed from another lab and shared with co-located training jobs. Pinned
    # to one GPU so a sweep can never evict someone else's work.
    "c2d": VM(
        user="andrew.carvalho",
        host="200.144.192.77",
        port=1232,
        key="~/.ssh/vm_c2d_s12",
        base="/home/andrew.carvalho/scale_wise_optimizer",
        shared=True,
        cuda_visible_devices="1",
    ),
}


def resolve_vm(name: str) -> VM:
    """The named VM, with any ``SWO_VM_*`` environment overrides applied."""
    if name not in VMS:
        raise SystemExit(f"unknown vm {name!r}; choose from {', '.join(VMS)}")
    overrides = {
        field: os.environ[key]
        for field, key in (
            ("user", "SWO_VM_USER"),
            ("host", "SWO_VM_HOST"),
            ("key", "SWO_SSH_KEY"),
            ("base", "SWO_VM_BASE"),
        )
        if key in os.environ
    }
    if "SWO_VM_PORT" in os.environ:
        overrides["port"] = int(os.environ["SWO_VM_PORT"])
    return dataclasses.replace(VMS[name], **overrides)


# -- ssh plumbing ---------------------------------------------------------


def _ssh_argv(vm: VM) -> list[str]:
    return [
        "ssh",
        "-i",
        os.path.expanduser(vm.key),
        "-p",
        str(vm.port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        f"{vm.user}@{vm.host}",
    ]


def ssh(vm: VM, command: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*_ssh_argv(vm), command],
        check=check,
        text=True,
        # Explicit UTF-8: remote logs carry tqdm block characters, and Windows
        # would otherwise decode them as cp1252 and raise.
        encoding="utf-8",
        errors="replace",
        capture_output=capture,
    )


def _succeeds(vm: VM, command: str) -> bool:
    return ssh(vm, command, check=False).returncode == 0


def session_alive(vm: VM, name: str) -> bool:
    return _succeeds(vm, f"tmux has-session -t {shlex.quote(name)} 2>/dev/null")


# -- commands -------------------------------------------------------------


def check(vm: VM) -> list[str]:
    """Print who and what is on the box. Returns the lines describing busy GPUs."""
    gpus = ssh(vm, "nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader").stdout.strip()
    logger.info(f"[{vm.host}] GPUs:\n{gpus}")
    logger.info(f"[{vm.host}] logged in: {ssh(vm, 'who | wc -l').stdout.strip()} session(s)")
    # Falls back to $HOME so the first check on a fresh VM still reports something.
    disk = ssh(vm, f'df -h {shlex.quote(vm.base)} 2>/dev/null || df -h "$HOME"', check=False)
    logger.info(f"[{vm.host}] disk: {disk.stdout.strip().splitlines()[-1] if disk.stdout.strip() else 'unknown'}")

    busy = []
    for line in gpus.splitlines():
        index, _name, used, _total = (part.strip() for part in line.split(","))
        if vm.cuda_visible_devices is not None and index != vm.cuda_visible_devices:
            continue
        if int(used.removesuffix(" MiB")) > BUSY_GPU_MIB:
            busy.append(line)
    return busy


def deploy(vm: VM) -> None:
    """Upload the package and sync dependencies. Idempotent."""
    repo = Path(__file__).resolve().parents[2]
    present = [path for path in UPLOAD_PATHS if (repo / path).exists()]
    logger.info(f"[{vm.host}] uploading {', '.join(present)} → {vm.workdir}")

    # tar over ssh rather than rsync: Git Bash on Windows has ssh and tar but no
    # rsync, and this keeps the whole flow off WSL.
    archive = subprocess.run(
        ["tar", "czf", "-", "-C", str(repo), *present],
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(
        [*_ssh_argv(vm), f"mkdir -p {shlex.quote(vm.workdir)} && tar xzf - -C {shlex.quote(vm.workdir)}"],
        input=archive,
        check=True,
    )

    logger.info(f"[{vm.host}] uv sync")
    setup = "\n".join(
        [
            "set -e",
            vm.env_exports(),
            'mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TMPDIR"',
            # Installs uv on first use, so a fresh VM needs no manual bootstrap.
            "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh",
            f"cd {shlex.quote(vm.workdir)}",
            "uv sync",
        ]
    )
    ssh(vm, f"bash -lc {shlex.quote(setup)}", capture=False)


def launch(vm: VM, name: str, sweep_args: list[str]) -> None:
    """Start the sweep in a detached tmux session, teeing to a log."""
    if session_alive(vm, name):
        raise SystemExit(f"tmux session {name!r} is already running — `stop` it first, or `logs` to watch it.")

    command = "python -m swo.sweep " + " ".join(shlex.quote(arg) for arg in sweep_args)
    launcher = f"""#!/bin/bash
# Written by swo.vm — do not edit on the VM.
set -u
mkdir -p {shlex.quote(vm.run_dir)}
rm -f {shlex.quote(vm.run_dir)}/exit_code
# Redirected straight to the file, not through `tee`: a process substitution
# can be torn down before it flushes, which loses the whole log — including the
# traceback you need. Nothing attaches to the tmux pane; `logs` tails this file.
exec > {shlex.quote(vm.run_dir)}/run.log 2>&1
{vm.env_exports()}
cd {shlex.quote(vm.workdir)}
source .venv/bin/activate
echo "[start] $(date -Iseconds)"
echo "[cmd]"
# Quoted heredoc, not echo: the command carries JSON, and bash would eat the
# inner double quotes inside an echo, logging a command that was never run.
cat <<'SWO_CMD_EOF'
{command}
SWO_CMD_EOF
{command}
ec=$?
echo "$ec" > {shlex.quote(vm.run_dir)}/exit_code
echo "[end] $(date -Iseconds) exit=$ec"
"""
    path = f"{vm.run_dir}/launch.sh"
    subprocess.run(
        [*_ssh_argv(vm), f"mkdir -p {shlex.quote(vm.run_dir)} && cat > {shlex.quote(path)}"],
        # Bytes, not text: on Windows a text-mode pipe rewrites \n as \r\n, and
        # bash then reads `set -u\r` and cd's into a path with a trailing \r.
        input=launcher.encode(),
        check=True,
    )
    ssh(vm, f"chmod +x {shlex.quote(path)} && tmux new-session -d -s {shlex.quote(name)} 'bash {shlex.quote(path)}'")
    # A short run can already be over by the time we look, so the sentinel counts
    # as evidence the launcher ran — otherwise a fast failure reads as "never started".
    if not session_alive(vm, name) and not _succeeds(vm, f"test -f {shlex.quote(vm.run_dir)}/exit_code"):
        raise SystemExit(f"tmux session {name!r} did not start — check `swo.vm logs`")
    logger.info(f"[{vm.host}] running in tmux session {name!r}")


def status(vm: VM, name: str) -> int | None:
    """Log whether the run is alive and how far it has got. Returns the exit code if finished."""
    alive = session_alive(vm, name)
    progress = ssh(
        vm,
        f"cd {shlex.quote(vm.workdir)} 2>/dev/null && wc -l {RESULTS_PATH}/*/*/samples.csv 2>/dev/null || true",
        check=False,
    ).stdout.strip()
    exit_code = ssh(vm, f"cat {shlex.quote(vm.run_dir)}/exit_code 2>/dev/null || true", check=False).stdout.strip()

    logger.info(f"[{vm.host}] session {name!r}: {'RUNNING' if alive else 'not running'}")
    if progress:
        logger.info(f"[{vm.host}] rows written (header included):\n{progress}")
    if not alive and exit_code:
        logger.info(f"[{vm.host}] last run exited {exit_code}")
    return int(exit_code) if not alive and exit_code.isdigit() else None


def wait(vm: VM, name: str, poll: int) -> int | None:
    """Block until the tmux session ends. Ctrl-C detaches; the VM keeps running."""
    logger.info(f"[{vm.host}] waiting (poll {poll}s) — Ctrl-C detaches without stopping the run")
    try:
        while session_alive(vm, name):
            time.sleep(poll)
    except KeyboardInterrupt:
        logger.info("detached; the run continues. `status` to check, `fetch` to pull results.")
        raise SystemExit(0) from None
    return status(vm, name)


def fetch(vm: VM) -> Path:
    """Pull the results tree down. Safe mid-run — the CSVs only ever hold whole rows."""
    repo = Path(__file__).resolve().parents[2]
    if not _succeeds(vm, f"test -d {shlex.quote(f'{vm.workdir}/{RESULTS_PATH}')}"):
        raise SystemExit(f"no {RESULTS_PATH}/ on the VM yet — has the sweep produced anything? try `logs`")

    archive = subprocess.run(
        [*_ssh_argv(vm), f"tar czf - -C {shlex.quote(vm.workdir)} {RESULTS_PATH}"],
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(["tar", "xzf", "-", "-C", str(repo)], input=archive, check=True)

    ssh(vm, f"cat {shlex.quote(vm.run_dir)}/run.log 2>/dev/null || true", check=False)
    (repo / RESULTS_PATH).mkdir(exist_ok=True)
    logger.info(f"fetched → {repo / RESULTS_PATH}")
    return repo / RESULTS_PATH


def logs(vm: VM, tail: int) -> None:
    print(ssh(vm, f"tail -n {tail} {shlex.quote(vm.run_dir)}/run.log 2>/dev/null || echo '(no log yet)'").stdout)


def stop(vm: VM, name: str) -> None:
    ssh(vm, f"tmux kill-session -t {shlex.quote(name)} 2>/dev/null || true", check=False)
    logger.info(f"[{vm.host}] stopped {name!r}")


# -- cli ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vm", default=os.environ.get("SWO_VM", "vm03"), choices=list(VMS), help="target machine")
    parser.add_argument("--name", default="swo", help="tmux session name; one run per name")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="deploy, launch, wait, fetch (re-run to resume)")
    run.add_argument("--poll", type=int, default=60, help="seconds between liveness checks")
    run.add_argument("--detach", action="store_true", help="return once tmux is up instead of waiting")
    run.add_argument("--force", action="store_true", help="launch even if the GPU is in use")
    run.add_argument("sweep_args", nargs=argparse.REMAINDER, help="after `--`, passed to swo.sweep")

    subparsers.add_parser("check", help="show GPU, session and disk state")
    subparsers.add_parser("status", help="is it running, and how far along")
    subparsers.add_parser("fetch", help="pull results (safe mid-run)")
    subparsers.add_parser("stop", help="kill the tmux session")
    log_parser = subparsers.add_parser("logs", help="tail the remote run log")
    log_parser.add_argument("--tail", type=int, default=60)

    args = parser.parse_args(argv)
    vm = resolve_vm(args.vm)

    if args.command == "check":
        check(vm)
        status(vm, args.name)
    elif args.command == "status":
        status(vm, args.name)
    elif args.command == "fetch":
        fetch(vm)
    elif args.command == "logs":
        logs(vm, args.tail)
    elif args.command == "stop":
        stop(vm, args.name)
    elif args.command == "run":
        sweep_args = [arg for arg in args.sweep_args if arg != "--"]
        if not sweep_args:
            parser.error("nothing to run: put the swo.sweep flags after `--`")
        if busy := check(vm):
            message = f"GPU in use on {vm.host}:\n" + "\n".join(busy)
            if vm.shared and not args.force:
                raise SystemExit(f"{message}\n{vm.host} is a shared/borrowed machine — wait, or pass --force.")
            logger.warning(message)
        deploy(vm)
        launch(vm, args.name, sweep_args)
        if args.detach:
            logger.info("detached. `status` to check, `fetch` when done.")
            return
        wait(vm, args.name, args.poll)
        fetch(vm)


if __name__ == "__main__":
    sys.exit(main())
