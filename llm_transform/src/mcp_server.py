from pathlib import Path
import re
import subprocess
import tempfile
import time

from fastmcp import FastMCP

from utils.transformation import transform_and_lower

PARENT_DIR = Path(__file__).parents[1]

mcp = FastMCP("mlir-transform")


@mcp.tool()
def run_schedule(
    id: str, transform_schedule: str, mlir_passes: str,
    llvm_passes: str = "default<O3>", llvm_flags: str = "", llc_flags: str = "",
    bufferize_first: bool = True
) -> float:
    """
    Apply the given transformation schedule to the MLIR code associated with the given ID. Lower using
    the specified MLIR and LLVM passes. Execute the resulting LLVM code and return the measured
    slowdown compared to PyTorch optimized code.

    This is the main metric for evaluating the effectiveness of the transformation schedule. The slowdown
    must be minimized as much as possible, ideally reaching around 0.5x or better (i.e. the transformed code
    runs at least 2x faster than PyTorch).

    Args:
        id (str): The unique identifier for the MLIR code to transform. It takes the form "{name}_{instance}", where "name" is the name of the benchmark (e.g. "matmul") and "instance" is the specific instance (e.g. "0", "1", etc.).
        transform_schedule (str): The transformation schedule to apply, specified as a string of MLIR transformations.
        mlir_passes (str, optional): The MLIR passes to apply during lowering. Defaults to BASE_MLIR_PASSES.
        llvm_passes (str, optional): LLVM opt pass pipeline to run before JIT (e.g. "licm,loop-unroll"). Defaults to "default<O3>".
        llvm_flags (str, optional): Comma-separated LLVM CL flags (e.g. "enable-loop-versioning-licm,licm-mssa-max-acc-promotion=1000"). Defaults to empty string.
        llc_flags (str, optional): Comma-separated flags for llc codegen (e.g. "align-loops=32,enable-split-loopiv-heuristic"). Defaults to empty string.
        bufferize_first (bool, optional): Whether to apply bufferization before applying the transformation schedule. Defaults to True.

    Returns:
        dict: the measured slowdown compared to PyTorch optimized code.
    """

    tmp_dir = PARENT_DIR / "tmp"
    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False, dir=tmp_dir) as transform_schedule_tmp, \
         tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, dir=tmp_dir) as mlir_passes_tmp:
        transform_schedule_tmp.write(transform_schedule)
        transform_schedule_tmp.flush()

        mlir_passes_tmp.write(mlir_passes)
        mlir_passes_tmp.flush()

    exec_script_name = "execute"
    exec_script = PARENT_DIR / "scripts" / f"{exec_script_name}.sh"

    cmd = [
        'sbatch', '--parsable', str(exec_script),
        '--id', id,
        '--transform_schedule_file', transform_schedule_tmp.name,
        '--mlir_passes_file', mlir_passes_tmp.name,
        '--llvm_passes', llvm_passes,
    ]

    if llvm_flags:
        cmd.extend(['--llvm_flags', llvm_flags])
    if llc_flags:
        cmd.extend(['--llc_flags', llc_flags])
    if not bufferize_first:
        cmd.append('--no-bufferize')

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_id = result.stdout.strip()

    # Wait for the job to complete and get the output
    while True:
        squeue_result = subprocess.run(['squeue', '-j', job_id, '-h'], capture_output=True, text=True)
        if squeue_result.returncode != 0:
            raise RuntimeError(f"squeue command failed: {squeue_result.stderr}")
        if squeue_result.stdout.strip() == "":
            break
        time.sleep(1)

    # Delete the temporary files
    Path(transform_schedule_tmp.name).unlink()
    Path(mlir_passes_tmp.name).unlink()

    # Read the results of the job
    output_file = PARENT_DIR / "logs" / "jobs" / f"{exec_script_name}_{job_id}.out"
    error_file = PARENT_DIR / "logs" / "jobs" / f"{exec_script_name}_{job_id}.err"

    # Check for errors first
    if error_file.exists() and error_file.stat().st_size > 0:
        with open(error_file, 'r') as f:
            error_content = f.read()
        raise RuntimeError(f"Job {job_id} failed with error:\n{error_content}")

    # Read the output and extract the slowdown compared to PyTorch
    if not output_file.exists():
        raise RuntimeError(f"Output file {output_file} not found for job {job_id}")
    with open(output_file, 'r') as f:
        output_content = f.read().strip()
    last_line = output_content.splitlines()[-1]
    match = re.search(r"Slowdown compared to PyTorch: ([\d.]+)x", last_line)
    if not match:
        raise RuntimeError(f"Unexpected output format in job {job_id} output: {output_content}")
    slowdown = float(match.group(1))

    # Clean up the output files
    output_file.unlink(missing_ok=True)
    error_file.unlink(missing_ok=True)

    return slowdown


@mcp.tool()
def lower_schedule(
    id: str, transform_schedule: str, mlir_passes: str,
    llvm_passes: str = "default<O3>", llvm_flags: str = "", llc_flags: str = "",
    bufferize_first: bool = True
) -> dict[str, str]:
    """
    Apply the given transformation schedule to the MLIR code associated with the given ID. Lower using
    the specified MLIR and LLVM passes. Compile to assembly. Return the transformed MLIR ("mlir_transformed"),
    the resulting LLVM ("llvm"), the optimized LLVM ("llvm_opt"), and the generated assembly ("asm").

    Args:
        id (str): The unique identifier for the MLIR code to transform. It takes the form "{name}_{instance}", where "name" is the name of the benchmark (e.g. "matmul") and "instance" is the specific instance (e.g. "0", "1", etc.).
        transform_schedule (str): The transformation schedule to apply, specified as a string of MLIR transformations.
        mlir_passes (str, optional): The MLIR passes to apply during lowering. Defaults to BASE_MLIR_PASSES.
        llvm_passes (str, optional): LLVM opt pass pipeline to run before JIT (e.g. "licm,loop-unroll"). Defaults to "default<O3>".
        llvm_flags (str, optional): Comma-separated LLVM CL flags (e.g. "enable-loop-versioning-licm,licm-mssa-max-acc-promotion=1000"). Defaults to empty string.
        llc_flags (str, optional): Comma-separated flags for llc codegen (e.g. "align-loops=32,enable-split-loopiv-heuristic"). Defaults to empty string.
        bufferize_first (bool, optional): Whether to apply bufferization before applying the transformation schedule. Defaults to True.

    Returns:
        dict: a dictionary containing:
            - "mlir_transformed": the transformed MLIR code after applying the transformation schedule
            - "llvm": the generated LLVM IR after lowering the transformed MLIR
            - "llvm_opt": the generated LLVM IR after applying llvm opt with the specified passes and flags
            - "asm": the generated assembly code after compiling with llc and the specified flags
    """

    return transform_and_lower(id, transform_schedule, mlir_passes, llvm_passes, llvm_flags, llc_flags, bufferize_first)


@mcp.prompt()
def optimization_prompt() -> str:
    """Provide a detailed prompt for the optimization task"""

    return """Read README.md thoroughly before starting.

## Your Task

For every MLIR benchmark in `data/`, generate optimized configurations that achieve <0.5x slowdown vs PyTorch. A configuration consists of:

1. **Transform schedule** (required) — an MLIR transform dialect module
2. **MLIR pass pipeline** (required) — the lowering pass pipeline string
3. **LLVM passes** (optional) — opt pass pipeline (default: `"default<O3>"`)
4. **LLVM flags** (optional) — comma-separated flags for opt
5. **LLC flags** (optional) — comma-separated flags for llc codegen

Use `run_schedule` to test configurations and `lower_schedule` to inspect intermediate output.

## Target Hardware

**Intel Xeon E5-2680 v4 @ 2.40GHz (Broadwell)**
- 28 cores (2 sockets × 14), 2 NUMA nodes
- L1d: 32KB/core, L2: 256KB/core, L3: 35MB shared/socket
- AVX2 + FMA, NO AVX-512
- 256-bit vectors = 4 doubles or 8 floats

## Workflow

1. **Discover** — List `data/`, read each `sizes.json` and `.mlir` to understand operation types, dimensions, and data types. Read `resources/base_schedule.mlir` and `resources/base_passes.txt` as starting points.

2. **Baseline** — For each benchmark ID, call `run_schedule` with the base schedule and base passes. Record the baseline slowdown.

3. **Iterate** — For each benchmark, repeatedly generate new configurations:
   a. Design a transform schedule and pass pipeline (and optionally LLVM/LLC passes/flags)
   b. Test with `run_schedule`, record the slowdown
   c. If needed, call `lower_schedule` to inspect MLIR, LLVM IR, and assembly
   d. Use insights to generate the next configuration
   e. Log every attempt to `logs/claude_optimization.log`

4. **Converge** — Per benchmark, stop when: slowdown < 0.5x (target met).

## What to Explore in Configurations

**Transform schedules:**
- Tiling (tile sizes tuned to cache hierarchy and problem dimensions)
- Vectorization (match AVX2 width)
- Loop interchange, unrolling, fusion
- Parallelization for large problems

**Pass pipelines:**
- Different orderings and combinations of MLIR lowering passes
- Toggle `bufferize_first`

**LLVM/LLC flags:**
- Different LLVM and codegen passes and flags

Adapt to each operation type and size — what works for a large matmul may not work for a small convolution.

## Web Search

Search aggressively when you need:
- MLIR transform dialect docs/examples (https://mlir.llvm.org/docs/Dialects/Transform/)
- Debugging help for error messages
- Optimization techniques for specific operation types
- LLVM pass and flag documentation

## Progress Logging

Write all progress to `logs/claude_optimization.log` in real-time:

```
=== MLIR Optimization Log ===

## Benchmark Discovery
[benchmarks, sizes, operation types]

## Optimization Progress

### {benchmark_id}
| Iter | Slowdown vs PyTorch | Configuration Summary |
|------|---------------------|-----------------------|
| 0    | 15.2x              | Baseline (no-op schedule, default passes) |
| 1    | 4.1x               | Tiling 32x32, default passes |
| 2    | 1.8x               | Tiling 64x64 + vectorize, -mcpu=broadwell |
| ...  | ...                | ... |

## Best Configurations
[per benchmark: the full schedule, passes, and flags that achieved the best result]

## Final Summary
[key insights, what worked, recommendations]
```

## Guidelines

- Test one change at a time when possible to isolate what helps
- Inspect assembly (via `lower_schedule`) to verify vectorization/tiling take effect
- Comment your schedules to document what each transformation does
- Handle errors gracefully — log them, analyze, try a variation
- Log the **full configuration** (schedule + passes + flags) for your best result per benchmark

## Success Criteria

**Target:** Slowdown vs PyTorch < 0.5x for every benchmark
"""


if __name__ == "__main__":
    mcp.run()
