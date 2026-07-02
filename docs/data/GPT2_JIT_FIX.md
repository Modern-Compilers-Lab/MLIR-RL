# GPT2 PyTorch JIT Fix — Summary

## Problem

`torch.jit.trace()` and `torch.jit.script()` both failed on all three GPT2 variants
(gpt2, gpt2-medium, gpt2-large), leaving `jit_ns: null` in all baseline tables.

| Attempt | Error |
|---------|-------|
| `torch.jit.trace()` | `'Tensor' has no attribute 'get_seq_length'` |
| `torch.jit.script()` | `Compiled functions can't take variable number of arguments` |

**Root cause**: HuggingFace `GPT2Model.forward()` calls `create_causal_mask()` — a
complex utility mixing config access, version checks, and SDPA internals that the
tracer cannot capture. The existing `GPT2Wrapper` also called the inner model with
keyword arguments (`self.m(input_ids=ids, attention_mask=mask)`), which
`torch.jit.script()` rejects.

## Approach: Model Surgery (Fix #1)

Bypass `GPT2Model.forward()` entirely. Create a new `nn.Module` that:

1. Extracts weight-carrying submodules directly from the pretrained model
   (`model.wte`, `model.wpe`, `model.h`, `model.ln_f`)
2. Implements its own `forward()` with **only tensor operations** — no config
   objects, no complex mask factories
3. Builds the 4D causal + padding attention mask manually

The individual `GPT2Block` submodules trace through naturally — their internals
are standard PyTorch ops.

## Files

| File | Status | Purpose |
|------|--------|---------|
| `utils/gpt2_jit_compat.py` | **new** | `GPT2JITCompat(nn.Module)` class + `make_gpt2_jit_compat()` loader |
| `scripts/baseline/get_pytorch_baselines.py` | **modified** | Added `_load_gpt2_jit()` dispatcher; gpt2 variants auto-use the wrapper |
| `scripts/measure_full_model_baselines.py` | **deprecated** | Marked as superseded by consolidated script |
| `docs/FIX1_MODEL_SURGERY.md` | existing | Full implementation guide (reference) |
| `docs/PYTORCH_JIT_DEBUGGING.md` | existing | Root cause analysis + all 4 proposed fixes |

## Key design decision: causal mask

The original `GPT2Model.forward()` creates a causal mask via
`create_causal_mask()` — a HuggingFace utility that combines causal (lower
triangular) and padding masks. The wrapper replicates this with two pure-tensor
operations:

```python
# Causal mask: prevent attending to future tokens
causal_mask = torch.tril(torch.ones(seq, seq)).view(1, 1, seq, seq)
causal_mask = (1.0 - causal_mask) * -10000.0  # 0 where attend, -10000 where masked

# Padding mask: mask out padding tokens
padding_mask = (attention_mask[:, None, None, :] == 0).float() * -10000.0

# Combined
attention_mask_4d = causal_mask + padding_mask
```

Without causal masking, the wrapper produced **different outputs** from the
original model (max diff = 3.69). With it, outputs match exactly (max diff = 0.0).

## Validation

All three variants tested with `torch.jit.trace()`:

| Check | gpt2 | gpt2-medium | gpt2-large |
|-------|------|-------------|------------|
| Numerical match (eager) | 0.00 diff | 0.00 diff | 0.00 diff |
| Numerical match (traced) | 1.45e-04 diff | 9.16e-05 diff | 4.29e-06 diff |
| `torch.jit.trace()` | ✅ | ✅ | ✅ |
| `torch.jit.script()` | ❌ (expected) | ❌ (expected) | ❌ (expected) |

`torch.jit.script()` intentionally not supported — `GPT2Block` internals contain
keyword arguments that TorchScript rejects, but trace captures the actual
execution path correctly.

## Results

| Model | Eager (ns) | JIT (ns) | Speedup | Before Fix |
|-------|-----------|----------|---------|-----------|
| gpt2 | 96,597,159 | 87,667,781 | 1.10x | `null` |
| gpt2-medium | 322,024,471 | 283,268,936 | 1.14x | `null` |
| gpt2-large | 585,793,319 | 575,584,864 | 1.02x | `null` |

Speedup is modest (2-14%) because PyTorch eager is already highly optimized for
fused matmul/attention on CPU. JIT adds graph-level fusion but the individual ops
are already efficient.

## What's NOT fixed

- **vit_b_16** — **FIXED** (see below). Old `_VitWrapper` removed; `torch.jit.script()` now
  succeeds on the raw ViT model (newer PyTorch version resolved TorchScript
  compatibility). `torch.jit.trace()` also works with `check_trace=False`.

## ViT B/16 JIT Fix (Bonus)

**Root cause**: `_VitWrapper` used `out.logits if hasattr(out, "logits") else out`
which TorchScript cannot compile. The ViT already returns a plain `Tensor` (no
`.logits` path needed), so the wrapper was a useless no-op that broke script
compilation.

**Fix**: Removed `_VitWrapper` entirely. The existing `trace → script` fallback
in `_time_jit()` now succeeds: trace works with `check_trace=False` (avoids SDPA
dispatch divergence check), script works on the raw `torchvision` ViT.

**Results**:

| Model | Eager (ns) | JIT (ns) | Speedup | Before |
|-------|-----------|----------|---------|--------|
| vit_b_16 | 394,930,307 | 377,777,627 | 1.05x | `null` |

Zero files created — just removed the wrapper class and added `check_trace=False`.

## Usage

```bash
# Measure all 22 models (Slurm)
sbatch scripts/full_model/get_pytorch_full_times.sh

# Measure specific models (ad-hoc)
python scripts/baseline/get_pytorch_baselines.py \
    --models gpt2 gpt2-medium gpt2-large \
    --config results/.../pytorch_models.json \
    --output /tmp/jit_times.json
```
