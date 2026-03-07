# LLM Transform

Optimize MLIR code by writing MLIR transform schedules, tuning MLIR lowering passes, and configuring LLVM passes and flags. The goal is to minimize execution time relative to PyTorch's optimized implementation.

## Metric

The primary metric is **slowdown compared to PyTorch**: `MLIR_time / PyTorch_time`. A value of 1.0 means parity with PyTorch. The target is 0.5x or lower (i.e. at least 2x faster than PyTorch).

## Project Structure

```
data/
  <name>/                    # Benchmark name (e.g. "matmul", "conv_2d")
    1.mlir                   # Instance 1 of the benchmark
    2.mlir                   # Instance 2, etc.
    sizes.json               # Problem sizes for each instance
resources/
  base_schedule.mlir         # Empty (no-op) transform schedule
  base_passes.txt            # Default MLIR lowering pass pipeline
src/
  mcp_server.py              # MCP server exposing the two tools below
  torch_exec.py              # PyTorch reference execution for comparison
  utils/
    transformation.py        # Core: apply schedule, bufferize, lower, compile
    execution.py             # CLI entry point: transform, run, and measure
  tools/
    plot_performance.py      # Plot slowdown over time per CODE_ID for an experiment
logs/
  jobs/                      # Slurm job output/error logs (not for LLM use)
  stats/                     # Per-experiment statistics (not for LLM use)
  claude_optimization.log    # Claude optimization progress log (write details here) (file must be created by the LLM)
scripts/
  claude.sh                  # Slurm job script: runs Claude sessions with logging
  execute.sh                 # Slurm job script: runs base, optimized, and PyTorch
tmp/                         # Temporary files (not for LLM use)
```

## MLIR Code Format

Each `.mlir` file in `data/` contains a function with a linalg operation tagged `{tag = "operation"}`. This tag is how the transform schedule identifies the target operation. Example (`matmul/2.mlir`):

```mlir
func.func @main(%arg0: tensor<512x512xf64>, %arg1: tensor<512x512xf64>)
    -> (tensor<512x512xf64>, i64) {
    %c0 = arith.constant 0.0 : f64
    %new = tensor.empty() : tensor<512x512xf64>
    %arg2 = linalg.fill ins(%c0 : f64) outs(%new : tensor<512x512xf64>) -> tensor<512x512xf64>
    %0 = call @nanoTime() : () -> i64
    %1 = linalg.matmul {tag = "operation"}
        ins(%arg0, %arg1 : tensor<512x512xf64>, tensor<512x512xf64>)
        outs(%arg2 : tensor<512x512xf64>) -> tensor<512x512xf64>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %1, %3 : tensor<512x512xf64>, i64
}
```

The code is in **tensor** semantics. Bufferization (tensor to memref) is handled automatically by the pipeline.

## Compilation Pipeline

1. **(Optional) Bufferize** — convert tensor semantics to memref (buffer) semantics
2. **Apply transform schedule** — MLIR transform dialect operations (tiling, vectorization, etc.)
3. **Apply MLIR passes** — lowering from linalg/scf/affine to LLVM dialect (see `resources/base_passes.txt`)
4. **Translate to LLVM IR** — `mlir-translate --mlir-to-llvmir`
5. **Optimize LLVM IR** — `opt` with the specified pass pipeline and flags
6. **Compile to object/assembly** — `llc` with the specified codegen flags
7. **Link** — produce a shared library and execute

## MCP Tools

Two tools are available:

### `run_schedule`

Execute the full pipeline and return the measured slowdown vs PyTorch. This is the main evaluation tool.

Parameters:
- `id` (required) — benchmark identifier in the form `"{name}_{instance}"` (e.g. `"matmul_2"`)
- `transform_schedule` (required) — MLIR transform schedule as a string
- `mlir_passes` (required) — MLIR lowering pass pipeline as a string
- `llvm_passes` — LLVM opt pass pipeline (default: `"default<O3>"`)
- `llvm_flags` — comma-separated LLVM CL flags (default: empty)
- `llc_flags` — comma-separated llc codegen flags (default: empty)
- `bufferize_first` — whether to bufferize before applying the schedule (default: `true`)

### `lower_schedule`

Apply the schedule and compile, returning intermediate representations for analysis (no execution).

Same parameters as `run_schedule`. Returns a dictionary with:
- `mlir_transformed` — MLIR after applying the transform schedule
- `llvm` — LLVM IR after lowering
- `llvm_opt` — LLVM IR after opt
- `asm` — generated assembly

## Transform Schedule Format

A transform schedule is an MLIR module using the [transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/). It must contain a named sequence `@__transform_main`. The empty (no-op) schedule is:

```mlir
module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        transform.yield
    }
}
```

To target the operation, match it by its tag:

```mlir
module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %op = transform.structured.match attributes {tag = "operation"} in %arg0
            : (!transform.any_op) -> !transform.any_op
        // Apply transformations to %op here (tile, vectorize, etc.)
        transform.yield
    }
}
```

## Available Benchmarks

| ID | Operation | Sizes |
| --- | --- | --- |
| `matmul_1` | Matrix multiply | M=24576, K=768, N=384 |
| `matmul_2` | Matrix multiply | M=512, K=512, N=512 |
| `matmul_3` | Matrix multiply | M=256, K=512, N=1024 |
