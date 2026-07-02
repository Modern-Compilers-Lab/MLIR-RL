# Fix #1: Model Surgery — Implementation Guide

## Overview

Recommended approach to enable PyTorch JIT tracing for gpt2 models by removing problematic dynamic kwargs and config access, then integrating the resulting module back into the baseline pipeline.

**Effort**: 2–4 hours per model family (gpt2, gpt2-large, gpt2-medium share patterns)  
**Expected speedup**: 5–10% vs PyTorch eager  
**Testing time**: 30 min (numerical validation + baseline comparison)

---

## Architecture Overview

The goal is to create a pure `torch.nn.Module` subclass that:
1. Removes dependency on `GPT2Config` and `**kwargs` (JIT-incompatible)
2. Extracts and freezes weights from the pretrained model
3. Implements only the forward path (no config initialization)
4. Is fully traceable by `torch.jit.trace()`

### Current Problem

```python
class GPT2Model:
    def __init__(self, config):
        self.config = config  # ← Contains **kwargs, not JIT-compatible
        self.transformer = GPT2Transformer(config)
    
    def forward(self, input_ids, attention_mask):
        # Forward uses self.config.hidden_size, etc.
```

### Solution Structure

```python
class GPT2JITCompat(torch.nn.Module):
    """Pure forward-only model, no config object."""
    def __init__(self, pretrained_model, config_dict):
        super().__init__()
        
        # Freeze architecture constants
        self.hidden_size = config_dict['hidden_size']
        self.num_layers = config_dict['num_layers']
        
        # Extract & freeze model components
        self.embedding = pretrained_model.transformer.wte
        self.position_embedding = pretrained_model.transformer.wpe
        self.layers = torch.nn.ModuleList(pretrained_model.transformer.h)
        self.ln_f = pretrained_model.transformer.ln_f
    
    def forward(self, input_ids, attention_mask):
        # Pure tensor operations, no config access
        x = self.embedding(input_ids) + self.position_embedding(...)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.ln_f(x)
```

---

## Step-by-Step Implementation

### Step 1: Extract Config Parameters

```python
from transformers import GPT2Model

model = GPT2Model.from_pretrained("gpt2-large").eval()
model.config.use_cache = False

# Capture all relevant config as a dict (JIT-serializable)
config_dict = {
    'vocab_size': model.config.vocab_size,
    'hidden_size': model.config.hidden_size,
    'num_hidden_layers': model.config.num_hidden_layers,
    'num_attention_heads': model.config.num_attention_heads,
    'intermediate_size': model.config.intermediate_size,
    'hidden_act': model.config.hidden_act,
    'hidden_dropout_prob': model.config.hidden_dropout_prob,
    'attention_probs_dropout_prob': model.config.attention_probs_dropout_prob,
    'max_position_embeddings': model.config.max_position_embeddings,
    'initializer_range': model.config.initializer_range,
    'layer_norm_eps': model.config.layer_norm_eps,
}
```

### Step 2: Create JIT-Compatible Wrapper

```python
import torch
import torch.nn as nn

class GPT2JITCompat(nn.Module):
    """Minimal GPT2 wrapper for JIT tracing."""
    
    def __init__(self, pretrained_model, config_dict):
        super().__init__()
        
        # Freeze config (not part of state_dict, won't be trained)
        self.register_buffer('_hidden_size', 
                            torch.tensor(config_dict['hidden_size'], dtype=torch.long))
        self.register_buffer('_max_pos', 
                            torch.tensor(config_dict['max_position_embeddings'], dtype=torch.long))
        
        # Extract frozen model components
        self.wte = pretrained_model.transformer.wte  # Token embedding
        self.wpe = pretrained_model.transformer.wpe  # Position embedding
        
        # Freeze all attention/mlp layers
        self.h = nn.ModuleList(pretrained_model.transformer.h)
        self.ln_f = pretrained_model.transformer.ln_f  # Final layer norm
        
        # Dropout (set to eval mode in forward)
        self.dropout = nn.Dropout(config_dict['hidden_dropout_prob'])
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_length]
            attention_mask: [batch_size, seq_length]
        
        Returns:
            hidden_states: [batch_size, seq_length, hidden_size]
        """
        batch_size, seq_length = input_ids.shape
        
        # Token embeddings
        hidden_states = self.wte(input_ids)
        
        # Position embeddings (0-indexed positions)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.wpe(position_ids)
        
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)
        
        # Reshape attention mask for attention computation
        # attention_mask: [batch_size, seq_length] → [batch_size, 1, seq_length, seq_length]
        # Set to very large negative for masked positions (softmax will zero them)
        attention_mask_2d = (1.0 - attention_mask.float()) * -10000.0
        attention_mask_4d = attention_mask_2d.unsqueeze(1).unsqueeze(2)
        
        # Pass through transformer layers
        for layer in self.h:
            layer_outputs = layer(hidden_states, attention_mask=attention_mask_4d)
            hidden_states = layer_outputs[0]  # Extract hidden states (ignore cache)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        return hidden_states
```

### Step 3: Validate Numerical Equivalence

```python
def validate_jit_wrapper():
    """Verify wrapper produces same outputs as original model."""
    from transformers import GPT2Tokenizer
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load original model
    original_model = GPT2Model.from_pretrained("gpt2-large").eval()
    original_model.config.use_cache = False
    
    # Load wrapper
    wrapper = GPT2JITCompat(original_model, config_dict).eval()
    
    # Test inputs
    text = "The quick brown fox jumps over the lazy dog."
    encoded = tokenizer(text, return_tensors="pt", 
                        padding="max_length", max_length=16, truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    # Forward pass (no grad)
    with torch.no_grad():
        original_out = original_model(input_ids, attention_mask).last_hidden_state
        wrapper_out = wrapper(input_ids, attention_mask)
    
    # Check numerical equivalence (allow small floating point error)
    max_diff = (original_out - wrapper_out).abs().max().item()
    mean_diff = (original_out - wrapper_out).abs().mean().item()
    
    print(f"Numerical validation:")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  Result: {'✓ PASS' if max_diff < 1e-5 else '✗ FAIL'}")
    
    return max_diff < 1e-5
```

### Step 4: Trace to JIT

```python
def trace_to_jit(wrapper, input_ids, attention_mask):
    """Trace the wrapper and verify JIT works."""
    wrapper = wrapper.eval()
    
    print("Attempting torch.jit.trace()...")
    try:
        traced = torch.jit.trace(wrapper, (input_ids, attention_mask))
        print("✓ Trace successful")
        
        # Verify traced model works
        with torch.no_grad():
            output = traced(input_ids, attention_mask)
        print(f"✓ Traced model forward pass successful: {output.shape}")
        
        return traced
    except Exception as e:
        print(f"✗ Trace failed: {type(e).__name__}: {e}")
        return None
```

### Step 5: Benchmark & Compare

```python
def benchmark_jit_vs_eager():
    """Compare JIT vs eager performance."""
    import time
    import numpy as np
    
    traced = trace_to_jit(wrapper, input_ids, attention_mask)
    if traced is None:
        return
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = wrapper(input_ids, attention_mask)
            _ = traced(input_ids, attention_mask)
    
    # Eager timing
    eager_times = []
    for _ in range(20):
        start = time.perf_counter_ns()
        with torch.no_grad():
            _ = wrapper(input_ids, attention_mask)
        end = time.perf_counter_ns()
        eager_times.append(end - start)
    
    # JIT timing
    jit_times = []
    for _ in range(20):
        start = time.perf_counter_ns()
        with torch.no_grad():
            _ = traced(input_ids, attention_mask)
        end = time.perf_counter_ns()
        jit_times.append(end - start)
    
    eager_median_ms = int(np.median(eager_times)) / 1_000_000
    jit_median_ms = int(np.median(jit_times)) / 1_000_000
    speedup = eager_median_ms / jit_median_ms
    
    print(f"\nBenchmark Results:")
    print(f"  Eager: {eager_median_ms:.1f} ms")
    print(f"  JIT:   {jit_median_ms:.1f} ms")
    print(f"  Speedup: {speedup:.2f}x")
```

---

## Integration with `measure_full_model_baselines.py`

Once Model Surgery produces a JIT-traceable wrapper, integrate it into the baseline measurement script:

```python
# In scripts/measure_full_model_baselines.py

def _load_gpt2_jit_compat():
    """Load gpt2 with JIT-compatible wrapper."""
    from transformers import GPT2Tokenizer, GPT2Model
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained("gpt2-large").eval()
    model.config.use_cache = False
    
    # Create wrapper
    wrapper = GPT2JITCompat(model, config_dict_from_model(model)).eval()
    
    # Create inputs
    enc = tokenizer("Hello from MLIR-RL!", return_tensors="pt",
                    padding="max_length", max_length=16, truncation=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    
    return wrapper, (input_ids, attention_mask)

# Add to MODEL_LOADERS
MODEL_LOADERS["gpt2-large"] = _load_gpt2_jit_compat
```

Then `_time_jit()` will automatically trace and benchmark the JIT version.

---

## Troubleshooting

### Issue: "Module 'X' has no attribute 'Y'" during trace

**Cause**: Traced forward path tries to access a module that wasn't included in the frozen ModuleList.

**Solution**: Add all referenced layers to the `__init__`:
```python
# ✗ Wrong
self.h = pretrained_model.transformer.h  # Direct assignment

# ✓ Right
self.h = nn.ModuleList(pretrained_model.transformer.h)
```

### Issue: "Graphs differed across invocations" or branching errors

**Cause**: Forward path has dynamic control flow (e.g., `if dropout_prob > 0`).

**Solution**: Remove conditional branching, always execute same code path:
```python
# ✗ Wrong
if self.training:
    x = self.dropout(x)

# ✓ Right (always apply, dropout is no-op in eval mode)
x = self.dropout(x)  # Harmless when module.eval()
```

### Issue: JIT model produces different outputs than eager

**Cause**: Floating-point precision issues or dtype mismatches.

**Solution**: Verify dtypes match and use `torch.allclose()` with tolerance:
```python
torch.allclose(eager_out, jit_out, atol=1e-5, rtol=1e-3)
```

### Issue: Traced model crashes during benchmark with "can't find CUDA kernel"

**Cause**: Tracing captured CUDA-specific code paths.

**Solution**: Ensure model is on CPU before tracing:
```python
wrapper = wrapper.cpu().eval()  # Not GPU
traced = torch.jit.trace(wrapper, (input_ids.cpu(), attention_mask.cpu()))
```

---

## Performance Expectations

| Model | Eager (ms) | JIT (ms) | Speedup | Notes |
|-------|-----------|---------|---------|-------|
| gpt2-large | 354 | ~320 | 1.11× | 5-10% speedup typical |
| gpt2-medium | 242 | ~220 | 1.10× | Consistent across sizes |
| gpt2 | 63.5 | ~58 | 1.09× | Smaller models see similar % gain |
| vit_b_16 | 171 | ~160 | 1.07× | Vision models slightly less gain |

**Why modest speedup?** PyTorch eager is already highly optimized for fused operations (attention, matmul). JIT adds minimal additional optimization beyond what eager mode provides on CPU.

---

## References

- [PyTorch JIT Tracing Guide](https://pytorch.org/docs/stable/jit.html)
- [HuggingFace GPT2Model Source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)
- [PYTORCH_JIT_DEBUGGING.md](PYTORCH_JIT_DEBUGGING.md) — Root cause analysis
- [scripts/measure_full_model_baselines.py](../scripts/measure_full_model_baselines.py) — Integration point
