---
name: vm-runs
description: Run resolution sweeps on the lab GPU VMs (vm03 default, vm02, c2d), monitor them, and bring results back, via `python -m swo.vm`. Use when the user asks to "run the sweep on the VM", "kick off X on vm03", "check the run", "fetch the results", or "it died — restart it". Covers shared-machine etiquette, which matters most on c2d (borrowed from another lab).
---

# Running sweeps on the lab GPU VMs

Everything goes through one module: `python -m swo.vm`. There are no shell
scripts, no job configs, no batch manifests, and **WSL is not needed** — Git Bash
has `ssh` and `tar`, which is all this uses.

## The whole surface

```bash
python -m swo.vm check                      # GPUs, sessions, disk — do this first
python -m swo.vm run -- <swo.sweep flags>   # deploy → launch → wait → fetch
python -m swo.vm status                     # alive? server healthy? rows written?
python -m swo.vm logs [--tail N] [--server]
python -m swo.vm fetch                      # safe mid-run
python -m swo.vm stop                       # kill the sweep

python -m swo.vm serve --model <hf-repo> [--gpu-memory-utilization 0.9]
python -m swo.vm serve-stop
```

## Serving with vLLM (preferred for real sweeps)

`serve` starts a vLLM OpenAI server in its own tmux session and blocks until
`/health` answers. It **outlives any number of sweeps** — start it once, run many
sweeps against it, stop it when done. Weights load once.

Pair it with the **`async_openai`** backend, not `openai`: `openai` has no
`system_prompt`, and losing that drops answer-format compliance from 100% to ~51%.
`async_openai` is the only backend with both `system_prompt` and real
`input_tokens` (`usage.prompt_tokens`).

```bash
python -m swo.vm --vm c2d serve --model Qwen/Qwen3.5-4B --gpu-memory-utilization 0.9
# `serve` prints the exact --backend-args to use; then:
python -m swo.vm --vm c2d run -- --task seedbench_2_plus --budgets ... \
  --chunk-size 500 --backend async_openai --backend-args '{...}'
python -m swo.vm --vm c2d serve-stop
```

Measured: **0.06 s/doc** vs 0.28–0.33 in-process (Qwen3.5-4B, 200 docs, one 5090)
— about 5×. Needs a few hundred documents before the adaptive concurrency ramps.

Three traps when serving:

- **Thinking must be disabled on the server.** `async_openai` has no per-request
  `chat_template_kwargs` (only the `openai` backend has `enable_thinking_kwarg`),
  so `serve` passes `--default-chat-template-kwargs '{"enable_thinking": false}'`
  by default. Leave it on and a reasoning model scores **0%** — it never reaches
  an answer. `--enable-thinking` turns it back on if a run genuinely needs it.

- **`min_pixels` moves server-side.** The image processor lives in vLLM now, so
  the floor is `--min-pixels` on `serve` (default 256), *not* a backend arg.
  Leave it at the model default and Qwen3.5's 65,536-px floor silently upscales
  every low budget.
- **vLLM pre-allocates GPU memory** and holds it for the server's whole life.
  `--gpu-memory-utilization` (default 0.9) is the knob; drop it on a busy box.
  `--tensor-parallel-size N` + `--gpus 0,1` shards a model across cards.

**Pass `--stop-server` to `run`.** It arms a watchdog *on the VM* that kills the
server the moment the sweep's tmux session ends — so a killed local process, a
dropped connection, or a closed laptop can never strand a server on a shared card.
Use it on every long run; it costs nothing and removes the whole failure mode.

**`serve-stop` the moment a run finishes — not when the conversation does.**
An idle server holds ~27 GB indefinitely at 0% utilisation, and these machines are
shared (c2d is borrowed). Bringing it back costs ~90 s, so there is never a reason
to keep a card reserved while waiting on a decision, reading results, or asking a
question. `check` and `status` both report whether a server is up; verify with:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

and check the owner of anything still listed — a colleague's job looks identical
to yours in the memory column.

**Default to one card.** Use `--data-parallel-size 2 --gpus 0,1` only when the
box is genuinely idle and the run is long enough to justify taking the whole
machine; drop back to one card for tests and probes.

`--vm vm03|vm02|c2d` (default `vm03`) goes **before** the subcommand.
Everything after `--` is passed verbatim to `swo.sweep`, so any sweep flag works
without changing this module.

Typical launch (the backend args below are the validated Qwen3.5 recipe — see
"Answer-format compliance" for why every one of them is there):

```bash
python -m swo.vm --vm c2d run -- \
  --task seedbench_2_plus \
  --budgets 2000,12500,25000,50000,100000,150000,250000,400000,600000 \
  --chunk-size 500 \
  --backend qwen3_5 \
  --backend-args '{"pretrained":"Qwen/Qwen3.5-4B","min_pixels":256,"max_pixels":16777216,"enable_thinking":false,"system_prompt":"You are answering a multiple-choice question. Reply with exactly one character: A, B, C, or D. Do not explain."}'
```

## Answer-format compliance — check this before every sweep

`seedbench_2_plus` scores `pred[0]`, the **first character** of the response, and
caps generation at 16 tokens. A chatty model is scored wrong however good its
answer is, and raising `max_new_tokens` does not help.

Worse, compliance *falls as resolution rises* (measured 80% → 40% on Qwen3.5-4B),
so it biases the resolution curve in the direction that matters.

**Required for the Qwen3/Qwen3.5 family**, and worth checking for any chatty VLM:

- `"enable_thinking": false` — it defaults to **`True`** in
  `lmms_eval/models/simple/qwen3_5.py`, which yields empty answers and 0.00
  accuracy at every budget. Do not "fix" this by raising `max_new_tokens`: at 512
  tokens half the responses still came back empty (`</think>` never closes) at
  18× the latency.
- `"system_prompt": "You are answering a multiple-choice question. Reply with exactly one character: A, B, C, or D. Do not explain."`
  — takes compliance from 51% to **100%** and cuts output to 2 tokens. The task's
  own `post_prompt` is not sufficient.

### The other half: is the model being cut off?

A reasoning model on a hard task blows through the task's `max_new_tokens` and is
scored on a truncated thought — at chance. Check `mean_output_tokens` in the
report against the task's cap:

| reads | meaning |
|---|---|
| ≈ the task's `max_new_tokens` | being truncated — raise it with `--gen-kwargs max_new_tokens=2048` |
| small (2–3) | answering directly; the cap is fine |

Measured on `mmmu_pro_standard`: at the default 256, 34% of answers truncated and
scored 0.121 (chance is 0.10) versus 0.407 untruncated. `max_new_tokens=2048`
recovered +7 points; 4096 added only +0.5, so 2048 is the knee. Keep the system
prompt — dropping it did not help and cost 1.6× the tokens.

**Always smoke-test compliance on 10 samples with a new model** before committing
GPU hours:

```bash
python -m swo.vm --vm <name> run -- --task <task> --budgets 2000,800000 \
  --limit 10 --backend <b> --backend-args '{...}' --run-name _compliance
python -m swo.report benchmarks/<task>/_compliance
```

`mean_output_tokens` near the task's `max_new_tokens` means the model is rambling
and being truncated. Near 2 means it is answering cleanly. Scratch runs use a
leading underscore in `--run-name`; those are gitignored.

## Etiquette — check before you launch

**These are shared lab machines, and c2d is borrowed from another lab.** Always
run `python -m swo.vm check --vm <name>` first and look at:

- **GPU memory in use.** Someone else's training job lives there. `run` already
  refuses to start on a shared machine when the target GPU holds >4 GiB, but on
  vm02/vm03 it only warns — read the warning, don't reflexively `--force`.
- **Logged-in sessions.** More than zero means a colleague is probably working.
- **Disk.** These fill up. Everything is redirected onto the big partition
  (`/media/<user>/ssd1T/andrew` on vm02/vm03), never `/home`, which is small.

On **c2d** specifically: it is pinned to `CUDA_VISIBLE_DEVICES=1` so a sweep can
never evict the co-located job, and `run` hard-refuses if that GPU is busy.
Do not lift the pin. If c2d is busy, use vm03 instead — don't wait it out by
force. Never delete caches or kill processes there.

If a machine is powered off, `C:\Users\Andrew\Msc\vm-ops\05_power_vms.sh --vm <name>`
can boot the lab boxes (needs the sudo password in that repo's `.env`).

**c2d needs its key on the Windows side first.** `vm_c2d_s12` currently exists only
inside WSL (`/home/andrew/.ssh/`), while this tool runs SSH from Windows. Copy it
once — `cp /home/andrew/.ssh/vm_c2d_s12* /mnt/c/Users/Andrew/.ssh/` from WSL — or
point `SWO_SSH_KEY` at another path. vm02/vm03 use `~/.ssh/vm_key`, which is
already there.

## Recovering from a failure

**Re-run the exact same `run` command.** That is the entire recovery procedure.

The sweep resumes from what is on disk: rows already in `samples.csv` tell it
which budget and which document offset to pick up from. A crashed chunk wrote
nothing (rows are only appended after an invocation returns), so there is no
partial state to clean up and no sentinel to reset.

Do **not**:
- delete `benchmarks/` on the VM to "force a retry" — that throws away finished work
- hand-edit `samples.csv` — appends are column-aligned against the header and a
  new column raises rather than silently shifting rows
- run a smoke test then a full rerun when a long run died mid-way; resuming
  replays only what's missing

Only start clean when the *configuration* changed (different model, different
budgets) — and then use a fresh `--run-name` rather than deleting anything.

## Reading a failure

`python -m swo.vm logs --tail 200`. Common causes, in rough order of likelihood:

| Symptom | Cause |
|---|---|
| `CUDA out of memory` during load | Model too big for the card. vm03 is a **12 GB RTX 4070** — an 8B VLM at bf16 will not fit. Use a smaller model or a quantized one. `gpu_memory_utilization` does not help here; it only budgets KV cache *after* weights load. |
| `No available memory for the cache blocks` | Weights fit, KV cache doesn't. Lower `max_model_len`, or `max_num_seqs=4`. |
| `No space left on device` on `/home` | A cache escaped the redirect. Check the `export` block at the top of the log. |
| accuracy ~0 with prose in `response` | The model is in thinking mode and burning the token budget before answering. Pass `chat_template_kwargs={"enable_thinking":false}` in `--backend-args`. |
| accuracy 0 at **every** budget, `response` **empty**, `output_tokens` pinned at the task's `max_new_tokens` | Same cause, worse symptom. `lmms_eval/models/simple/qwen3_5.py` defaults `enable_thinking=True`; `_strip_thinking` partitions on `</think>`, which never arrives within `max_new_tokens: 16`, so the stripper returns an empty string. Pass `"enable_thinking": false` in `--backend-args` (a plain kwarg for the qwen3_vl/qwen3_5 backends, *not* nested in `chat_template_kwargs`). Flat `output_tokens` exactly equal to `max_new_tokens` is the giveaway. |
| accuracy identical across all budgets | **The budget isn't binding.** The backend's processor is re-resizing. Set a `min_pixels` floor below the smallest budget. Confirm against the `mean X -> Y px/image` line the sweep logs each run. |

Note that lmms-eval's `cli_evaluate` swallows exceptions and can exit 0. Trust
the row counts from `status`, not the exit code.

## Verifying cheaply before spending GPU time

`--dry-run` runs the whole pipeline — deploy, tmux, task loading, image
downscaling, CSV writing, fetch — with a backend that loads no weights. Use it
to validate a new task or budget list without touching the GPU:

```bash
python -m swo.vm run -- --task <task> --budgets 2000,800000 --limit 12 \
  --chunk-size 5 --dry-run --run-name _smoke
```

Accuracy from a dry run is meaningless; the pixel columns are the point.

## Notes for agents

- Launch with `run_in_background: true` and wait for the notification. **Do not
  poll in a sleep loop** — `run` already blocks until the session ends.
- Ctrl-C (or killing the local process) detaches only; the VM keeps running.
  Reattach with `status` / `logs`, collect with `fetch`.
- One `tmux` session per `--name` (default `swo`). Use distinct names to run
  different tasks concurrently — but remember they share one GPU.
- `fetch` is safe at any time: the CSVs only ever contain complete rows.
