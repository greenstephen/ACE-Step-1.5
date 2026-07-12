# Multi-GPU follow-up notes

Operator / agent handoff for stacked multi-GPU work and deferred fixes.
Last updated: 2026-07-11.

## Open stacked PRs (do not disturb)

| Order | Branch | Upstream PR | Scope |
|------:|--------|-------------|--------|
| 1 | `feat/multi-gpu-pr1-device-map` | [#1262](https://github.com/ace-step/ACE-Step-1.5/pull/1262) | `ComponentDeviceMap` |
| 2 | `feat/multi-gpu-pr2-auto-layout` | [#1263](https://github.com/ace-step/ACE-Step-1.5/pull/1263) | Auto layout + cross-GPU routing |
| 3 | `feat/multi-gpu-pr3-cli-api` | [#1264](https://github.com/ace-step/ACE-Step-1.5/pull/1264) | CLI/API (`--gpu-mapping`, `--list-gpus`), status, docs |

**Policy:** leave these PRs alone until merged. Do not add unrelated follow-ups into their review threads. Prefer one follow-up PR at a time after the stack lands.

## Pending: nano-vllm LM on GPU 0 (wait to open PR)

**Decision (2026-07-11):** do **not** open a fourth review PR while #1262–#1264 are open. Keep the fix local (or draft on the fork without requesting review) until PR3 merges, then rebase onto `main` and open a focused PR.

| Item | Value |
|------|--------|
| Local branch | `fix/nanovllm-honor-cuda-device` (based on PR3 tip) |
| Status | Implemented and validated on 4×3090; **not** submitted for review yet |
| Depends on | PR3 (`--gpu-mapping` + `device_map.lm` → `cuda:N`) |

### Bug (validated)

With explicit mapping e.g. `dit:1,vae:0,text_encoder:2,lm:3`:

1. Init could log “LM on GPU 3” while weights still landed on **GPU 0**.
2. After fixing load placement, generate failed with: index tensors on `cuda:0`, weights on `cuda:3`.

### Root causes

1. **Load:** nano-vllm `ModelRunner` used `torch.cuda.set_device(rank)` where `rank` is tensor-parallel rank (always `0` for ACE-Step). ACE-Step’s `cuda_device=` kwarg was ignored by older installs that lacked `Config.cuda_device` (kwargs silently dropped).
2. **Generate:** bare `tensor.cuda()` follows `torch.cuda.current_device()`, which DiT/VAE often leave on GPU 0 after multi-GPU work.
3. **Install trap:** non-editable path install of `nano-vllm` + `uv run` could replace an editable fix with a stale site-packages copy. Prefer `editable = true` for the path source; use `uv run --no-sync` when avoiding resync.

### Fix contents (already on the branch)

- `Config.cuda_device` + `resolve_cuda_device_id` / `to_runner_cuda_device`
- `ModelRunner` pins load + transfers to `self.device_id`; `run()` calls `set_device(self.device_id)`
- ACE-Step passes `cuda_device=` from mapped LM device; warns if installed Config lacks the field
- Editable `nano-vllm` in `pyproject.toml` / `uv.lock`
- Unit tests: `nanovllm/cuda_device_test.py`, `acestep/llm_inference_cuda_index_test.py`

### CLI note

Official flag: **`--gpu-mapping`** (hyphen). `--gpu_mapping` is unrecognized.

### After PR3 merges

```bash
git fetch origin
git checkout fix/nanovllm-honor-cuda-device
git rebase origin/main   # or merge; resolve if needed
# open one focused PR: honor mapped cuda:N for nano-vllm LM load + generate
```

## Later follow-ups (after nano-vllm PR or in parallel once stack is in)

1. **Docs** — component placement ≠ pooled VRAM; `auto` vs explicit; 8 GB-class recipe.
2. **Capability UI/CLI** — per-GPU fit / utilization (after docs language exists).
3. **Turing/dtype** (optional) — honor `ACESTEP_DTYPE`; float16→INT8→float32 on pre-Ampere 8 GB.

## Hard facts from hardware smoke

- VRAM is per-card; largest **component** must fit on one GPU.
- `auto` needs free VRAM ≥ DiT peak (~7.53 GB turbo); near-full 8 GB cards often fall back to single-GPU.
- Explicit maps work when each component fits (e.g. turbo + 1.7B; XL + 4B on 4×3090 with `lm` on a free card).
- Turing + default float16 DiT → NaN risk; prefer INT8 / float32 smoke paths there.
