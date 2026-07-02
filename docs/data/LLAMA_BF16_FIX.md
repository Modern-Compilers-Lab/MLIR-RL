# Llama 3.2 1B bf16 → f32 Re-export Guide

## Problem

All ~900 Llama 3.2 1B benchmark files fail with `et = -1` during MLIR baseline timing.

**Root cause:** The raw model (`llama3_2_1b_linalg.mlir`) was exported with HuggingFace's default `torch_dtype=bfloat16`. The extracted blocks and single-ops inherit `tensor<...xbf16>` types. `mlir-cpu-runner` (LLVM JIT) cannot lower bf16 ops to executable x86 code — there is no bf16 hardware support on CPUs.

Every other model uses `tensor<...xf32>` and executes correctly.

## Fix Overview

Re-export Llama 3.2 1B in `float32`, then regenerate extracted blocks and single-ops.

## Step-by-Step

### 1. Re-export Raw Model in f32

Write a script (or run interactively) that does:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B"
OUTPUT_ONNX = "llama3_2_1b_f32.onnx"

# Load in float32 (NOT bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,  # <-- THIS IS THE KEY CHANGE
    device_map="cpu",
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Wrapper for DynamicCache (same as original export) ---
# The existing export used a wrapper that passes cache_position explicitly
# to bypass DynamicCache tracing issues in transformers >=4.57.
# Reuse the same wrapper class. It should look like:
#
# class LlamaWrapper(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#     def forward(self, input_ids, attention_mask):
#         return self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             use_cache=False,
#         ).logits

# Export to ONNX (opset 18)
dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 16))
dummy_attention_mask = torch.ones(1, 16, dtype=torch.long)

torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    OUTPUT_ONNX,
    opset_version=18,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits": {0: "batch", 1: "seq"},
    },
)
```

### 2. Convert ONNX → torch-mlir → linalg

```bash
# Convert ONNX to torch-mlir IR
torch-mlir-import-onnx llama3_2_1b_f32.onnx -o llama3_2_1b_torch.mlir

# Lower to linalg-on-tensors
torch-mlir-opt \
    --convert-torch-onnx-to-torch \
    --torch-backend-to-linalg-on-tensors-backend-pipeline \
    llama3_2_1b_torch.mlir \
    -o llama3_2_1b_linalg.mlir
```

Note: `torch-mlir-import-onnx` and `torch-mlir-opt` are part of the LLVM build at `$LLVM_BUILD_PATH/bin/`.

### 3. Replace Raw Model File

```bash
cp llama3_2_1b_linalg.mlir \
   data/new_dataset/nn/raw_bench/llama3_2_1b_linalg.mlir
```

Verify the new file has NO `bf16` occurrences:
```bash
grep -c "bf16" data/new_dataset/nn/raw_bench/llama3_2_1b_linalg.mlir
# Should print: 0
```

### 4. Re-extract Single Ops

```bash
python data_utils/extract/extract_ops.py \
    --input data/new_dataset/nn/raw_bench/llama3_2_1b_linalg.mlir \
    --output-dir data/new_dataset/nn/code_files/single_bench/ \
    --generic-ratio 0.25 \
    --model-prefix llama3_2_1b

python data_utils/extract/extract_ops.py \
    --input data/new_dataset/nn/raw_bench/llama3_2_1b_linalg.mlir \
    --output-dir data/new_dataset/nn/code_files/single_bench_full/ \
    --no-require-reduction \
    --model-prefix llama3_2_1b
```

### 5. Re-extract Blocks

```bash
python data_utils/extract/extract_blocks.py \
    --input data/new_dataset/nn/raw_bench/llama3_2_1b_linalg.mlir \
    --output-dir data/new_dataset/nn/code_files/bench_train/ \
    --skip-pure-elementwise \
    --window 5 --stride 3 \
    --model-prefix llama3_2_1b
```

### 6. Rebuild Flat Dataset

First, remove old bf16 llama files from the flat directories:
```bash
rm data/new_dataset/all/train/llama3_2_1b_*.mlir
rm data/new_dataset/all/eval/llama3_2_1b_*.mlir
rm data/new_dataset/all/eval_full/llama3_2_1b_*.mlir
```

Then re-run the stratification/split script to populate them with the new f32 files:
```bash
python scripts/data/split_json.py config/new_dataset/train/v4_7.json
```

### 7. Re-run MLIR Baselines

Delete the old baseline chunk files (they contain -1 for llama entries):
```bash
rm results/new_dataset_results/baselines/exec_times/train_base_chunk*.json
```

Re-submit:
```bash
sbatch scripts/baseline/get_new_dataset_base.sh
```

After train completes and is merged, submit eval and eval_full baselines.

## Verification

After re-export, confirm:
```bash
# No bf16 in new raw file
grep -c "bf16" data/new_dataset/nn/raw_bench/llama3_2_1b_linalg.mlir

# Sample block is executable (should print positive time)
python scripts/baseline/get_base.py \
    --benchmarks-dir data/new_dataset/all/train/ \
    --output /tmp/llama_test.json \
    --filter "llama3_2_1b_block_0"

python3 -c "import json; d=json.load(open('/tmp/llama_test.json')); print(d)"
# Should show a positive number (nanoseconds), NOT -1
```

## Notes

- The wrapper class for DynamicCache bypass is critical for transformers >=4.57. If you don't have it saved, you'll need to recreate it. The key requirement: pass `use_cache=False` and handle `cache_position` explicitly.
- Llama 3.2 1B in f32 should fit in Slurm memory (~4 GB weights). This is why the 1B variant was chosen over larger models.
- All other 17 models in the dataset already use f32 and require no changes.
