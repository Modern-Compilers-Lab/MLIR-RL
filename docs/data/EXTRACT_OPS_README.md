# extract_ops.py — Reference

Extracts individual linalg operations from a full-model MLIR file and
produces one self-contained benchmark file per operation, ready for RL
training or stand-alone performance measurement.

---

## Why this tool exists

Model-level MLIR files (e.g. `resnet18_linalg.mlir`) contain dozens or
hundreds of ops all wired together inside a single function.  The RL
auto-scheduler needs to evaluate each op in isolation.  `extract_ops.py`
automates the splitting, renaming, and wrapping that turns a monolithic file
into a directory of small, runnable benchmarks.

---

## How it works

The script performs two passes over the input file:

### Pass 1 — Pre-scan

Before touching any ops, the file is scanned once to collect:

| Collected item | What it is | Why it is needed |
|---|---|---|
| `affine_map` declarations | `#mapN = affine_map<...>` at file scope | `linalg.generic` ops reference maps by name; the map must accompany the extracted op |
| Scalar `arith.constant` values | Module-level constant definitions | Generic region bodies may reference outer constants that must be hoisted into the benchmark |

### Pass 2 — Op extraction

Each line is examined for a target op keyword.  When one is found:

1. **Collect** the full op text (multi-line for `linalg.generic`, single-line for named structured ops).
2. **Filter** — the op is *skipped* if any of the following apply:
   - It is a trivial op (`linalg.fill`, `linalg.broadcast`, `linalg.transpose`, `linalg.reduce`, `linalg.index`).
   - It is a `linalg.generic` with fewer parallel loops than `--min-parallel-loops` (default 2).
   - It produces a tuple result `-> (type1, type2, ...)`.
   - Its `ins`/`outs` blocks are empty or unparseable.
   - Any tensor dimension is still dynamic (`?`) after substituting `--batch-size`.
   - Any tensor element type is `i1` (boolean tensors).
   - The op body references a dense tensor weight that was elided (`dense_resource<__elided__>`).
   - The op body references an SSA value that cannot be resolved in the extracted scope (a "dangling" reference).
3. **Deduplicate** — ops with the same type and identical tensor shapes are emitted only once.
4. **Patch** the op text:
   - The result SSA is renamed to `%result`.
   - `ins` SSA names become `%arg0, %arg1, …`
   - `outs` SSA names continue the numbering: `%arg<N>, %arg<N+1>, …`
   - All `?` dynamic dimensions are replaced with the concrete `--batch-size` value.
5. **Assemble** a complete benchmark module:
   ```
   #mapN = affine_map<...>    ← referenced affine maps (if any)

   module {
     func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
     func.func @main(%arg0: tensor<...>, ...) -> (tensor<...>, i64)
                attributes {llvm.emit_c_interface} {
       %t0   = call @nanoTime() : () -> i64
       // hoisted arith.constants (linalg.generic only)
       %result = linalg.<op> ins(%arg0, ... : ...) outs(%argN, ... : ...) -> tensor<...>
       %t1   = call @nanoTime() : () -> i64
       %delta = arith.subi %t1, %t0 : i64
       return %result, %delta : tensor<...>, i64
     }
   }
   ```
6. **Write** to `<output-dir>/<model-name>_<op-type>_<index>.mlir`.

---

## Target ops

The following linalg ops are extracted:

| Op | Category |
|---|---|
| `linalg.matmul` | Dense linear algebra |
| `linalg.batch_matmul` | Dense linear algebra |
| `linalg.conv_2d_nchw_fchw` | Convolution |
| `linalg.conv_2d_nhwc_hwcf` | Convolution |
| `linalg.pooling_nchw_max` | Pooling |
| `linalg.pooling_nhwc_max` | Pooling |
| `linalg.pooling_nchw_sum` | Pooling |
| `linalg.pooling_nhwc_sum` | Pooling |
| `linalg.add` | Element-wise |
| `linalg.generic` | Arbitrary structured op |

`linalg.fill`, `linalg.broadcast`, `linalg.transpose`, `linalg.reduce`, and
`linalg.index` are intentionally excluded — they are too trivial to be
interesting scheduling targets.

---

## CLI Reference

```
python data_utils/extract/extract_ops.py [options]
```

| Flag | Required | Default | Description |
|---|---|---|---|
| `--input` | yes | — | Input `*_linalg.mlir` file |
| `--output-dir` | yes | — | Directory to write benchmark files into (created if absent) |
| `--batch-size` | no | `1` | Concrete integer to substitute for `?` dynamic batch dimensions |
| `--model-name` | no | derived from filename | Prefix for output filenames |
| `--min-parallel-loops` | no | `2` | Minimum number of parallel loops required for `linalg.generic` ops to be kept |
| `--clean` | no | `false` | Delete all `.mlir` files in `--output-dir` before extracting |

---

## Usage examples

### Single model file

```bash
python data_utils/extract/extract_ops.py \
    --input  data/nn/generated/code_files/resnet18_linalg.mlir \
    --output-dir data/nn/extracted/resnet18/ \
    --batch-size 1 \
    --model-name resnet18
```

Sample output:
```
Model : resnet18
Written: 23 benchmark file(s) → data/nn/extracted/resnet18/
Skipped: 41 op(s)
Op breakdown: {'conv_2d_nchw_fchw': 16, 'add': 4, 'generic': 3}
```

Each file is named `resnet18_<op-type>_<index>.mlir`, e.g.:
```
resnet18_conv_2d_nchw_fchw_0.mlir
resnet18_conv_2d_nchw_fchw_1.mlir
...
resnet18_generic_0.mlir
```

---

### Batch-process all models in a directory

```bash
for f in data/nn/generated/code_files/*_linalg.mlir; do
    model=$(basename "$f" _linalg.mlir)
    python data_utils/extract/extract_ops.py \
        --input      "$f" \
        --output-dir "data/nn/extracted/$model" \
        --batch-size 1 \
        --model-name "$model"
done
```

---

### Large model files (many operations)

Large models such as BERT, GPT-2, or ViT can produce hundreds of ops.
A few tips:

**1. Use `--clean` to restart a run safely**

If you re-run extraction after modifying the input, stale files from a
previous run are removed first:

```bash
python data_utils/extract/extract_ops.py \
    --input  data/nn/generated/code_files/bert_linalg.mlir \
    --output-dir data/nn/extracted/bert/ \
    --batch-size 1 \
    --clean
```

**2. Raise `--min-parallel-loops` to reduce noise from tiny generics**

Very small element-wise generics (e.g. scalar bias adds) have few parallel
loops and are rarely interesting.  Raising the threshold prunes them:

```bash
python data_utils/extract/extract_ops.py \
    --input  data/nn/generated/code_files/gpt2_linalg.mlir \
    --output-dir data/nn/extracted/gpt2/ \
    --batch-size 1 \
    --min-parallel-loops 4
```

**3. Set `--batch-size` to expose realistic shapes**

By default `?` is replaced with `1`.  For transformer models the batch
dimension is often outer-most; a larger batch size gives more realistic
data (but may make some ops too large to JIT-compile quickly):

```bash
python data_utils/extract/extract_ops.py \
    --input  data/nn/generated/code_files/bert_linalg.mlir \
    --output-dir data/nn/extracted/bert_b8/ \
    --batch-size 8
```

**4. Parallel batch processing across many models**

```bash
for f in data/nn/generated/code_files/*_linalg.mlir; do
    model=$(basename "$f" _linalg.mlir)
    python data_utils/extract/extract_ops.py \
        --input "$f" \
        --output-dir "data/nn/extracted/$model" \
        --batch-size 1 &
done
wait
echo "All models extracted"
```

---

## Output file format

Each output file is a complete, self-contained MLIR module.  It can be
compiled and run directly with `mlir-cpu-runner` or used by the RL
training pipeline.

```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d2)>   ← only present when needed
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(
      %arg0: tensor<1x3x224x224xf32>,   ← all ins tensors (concrete shapes)
      %arg1: tensor<64x3x7x7xf32>,
      %arg2: tensor<1x64x112x112xf32>   ← outs tensor (also an input arg)
  ) -> (tensor<1x64x112x112xf32>, i64) attributes {llvm.emit_c_interface} {
    %t0 = call @nanoTime() : () -> i64
    %result = linalg.conv_2d_nchw_fchw
        ins(%arg0, %arg1 : tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>)
        outs(%arg2 : tensor<1x64x112x112xf32>)
     -> tensor<1x64x112x112xf32>
    %t1 = call @nanoTime() : () -> i64
    %delta = arith.subi %t1, %t0 : i64
    return %result, %delta : tensor<1x64x112x112xf32>, i64
  }
}
```

The `@main` function always returns `(result_tensor, elapsed_nanoseconds)`.

---

## Deduplication

If the same model contains the same op type with identical tensor shapes
more than once (common in ResNets, where the same conv block is repeated),
only the *first* occurrence is emitted.  This prevents the training dataset
from being dominated by repeated identical samples.

---

## Skipped ops — common reasons

| Reason | What to check |
|---|---|
| Dynamic dims remain after substitution | `--batch-size` only replaces `?`; non-batch dynamic dims in weight tensors are unsupported |
| `dense_resource<__elided__>` | Run `strip_mlir.py` *before* extraction to remove elided weights; ops referencing them are dropped |
| Dangling SSA reference | The op body uses a value computed by an earlier op in the original function; reconstruction is not possible |
| Tuple result | Multi-output ops are not supported |
| Too few parallel loops | Increase `--min-parallel-loops` threshold or leave as default |
