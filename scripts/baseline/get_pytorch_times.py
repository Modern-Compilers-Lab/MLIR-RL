#!/usr/bin/env python3
"""
Measure PyTorch execution times (eager, torch.jit) for each
benchmark in a directory of .mlir files.

When called with --config, only eval benchmarks (base_eval.json) are
measured — PyTorch times are only used for comparison against RL eval results,
so measuring train benchmarks would waste time.

For each .mlir file the script:
  1. Parses the MLIR to extract operation type and tensor shapes.
  2. Builds the equivalent PyTorch operation.
  3. Measures wall-clock time (nanoseconds) with warm-up, using:
       - eager  : plain PyTorch
       - compile: torch.compile  (requires torch >= 2.0, skipped otherwise)
       - jit    : torch.jit.trace

Output JSON format (matches get_base.py flat format per mode):
  {
    "bench_name": {"eager": <ns>, "compile": <ns>, "jit": <ns>},
    ...
  }

Usage:
  python scripts/baseline/get_pytorch_times.py \\
      --benchmarks-dir data/all \\
      --output results/all/exec_times/pytorch.json   # default if omitted
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

NUM_WARMUP = 3
NUM_TRIALS = 10


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def measure_ns(fn: Callable, warmup: int = NUM_WARMUP, trials: int = NUM_TRIALS) -> int:
    for _ in range(warmup):
        fn()
    start = time.perf_counter_ns()
    for _ in range(trials):
        fn()
    end = time.perf_counter_ns()
    return (end - start) // trials


# ---------------------------------------------------------------------------
# MLIR parsing helpers
# ---------------------------------------------------------------------------

def _normalize(code: str) -> str:
    return re.sub(r'\s+', ' ', code)


def parse_tensor_shape(type_str: str) -> tuple[int, ...]:
    """'tensor<1x64x64x64xf32>' -> (1, 64, 64, 64)"""
    m = re.search(r'tensor<([\dx]+)x\w+>', type_str)
    if not m:
        return ()
    return tuple(int(d) for d in m.group(1).split('x'))


def parse_main_args(code: str) -> list[tuple[int, ...]]:
    """Return shapes of every %argN in the @main function signature."""
    norm = _normalize(code)
    m = re.search(r'func\.func @main\((.+?)\)\s*->', norm)
    if not m:
        return []
    sig = m.group(1)
    shapes = []
    for part in re.split(r'%[a-zA-Z_]\w*\s*:', sig):
        s = parse_tensor_shape(part)
        if s:
            shapes.append(s)
    return shapes


def get_linalg_op(code: str) -> Optional[str]:
    m = re.search(r'linalg\.(\w+)', code)
    return m.group(1) if m else None


def _parse_dense(code: str, attr: str) -> tuple[int, int]:
    """Parse strides= or dilations= from MLIR attributes."""
    # dense<N> -> (N, N)
    m = re.search(rf'{attr}\s*=\s*dense<(\d+)>', code)
    if m:
        v = int(m.group(1))
        return (v, v)
    # dense<[M, N]> -> (M, N)
    m = re.search(rf'{attr}\s*=\s*dense<\[(\d+),\s*(\d+)\]>', code)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (1, 1)


# ---------------------------------------------------------------------------
# PyTorch function builder
# ---------------------------------------------------------------------------

def build_pytorch_fns(
    mlir_code: str,
) -> tuple[Optional[dict[str, Callable]], Optional[str]]:
    """
    Parse an MLIR benchmark and return a dict of callables:
        {"eager": fn, "jit": fn}
    Returns (None, error_message) on failure.
    """
    op = get_linalg_op(mlir_code)
    args = parse_main_args(mlir_code)

    if not op:
        return None, "linalg op not found"
    if not args:
        return None, "cannot parse @main args"

    # Pre-check: skip files with tensors too large to allocate.
    # ~500M f32 elements ≈ 2 GiB per tensor.
    import math
    _MAX_ELEMS = 500_000_000
    for shape in args:
        if math.prod(shape) > _MAX_ELEMS:
            return None, f"tensor too large: {shape}"

    # -----------------------------------------------------------------------
    # conv_2d_nchw_fchw
    # -----------------------------------------------------------------------
    if op == 'conv_2d_nchw_fchw':
        if len(args) < 2:
            return None, "not enough args for conv2d"
        inp    = torch.randn(*args[0])
        weight = torch.randn(*args[1])
        strides   = _parse_dense(mlir_code, 'strides')
        dilations = _parse_dense(mlir_code, 'dilations')

        def eager_fn():
            return F.conv2d(inp, weight, stride=strides, dilation=dilations)

        def _conv_for_trace(x):
            return F.conv2d(x, weight, stride=strides, dilation=dilations)

        jit_fn  = torch.jit.trace(_conv_for_trace, (inp,))
        jit_run = lambda: jit_fn(inp)

    # -----------------------------------------------------------------------
    # matmul  (2-D)
    # -----------------------------------------------------------------------
    elif op == 'matmul':
        if len(args) < 2:
            return None, "not enough args for matmul"
        a = torch.randn(*args[0])
        b = torch.randn(*args[1])

        def eager_fn():
            return torch.mm(a, b)

        def _mm_for_trace(x, y):
            return torch.mm(x, y)

        jit_fn  = torch.jit.trace(_mm_for_trace, (a, b))
        jit_run = lambda: jit_fn(a, b)

    # -----------------------------------------------------------------------
    # batch_matmul  (3-D)
    # -----------------------------------------------------------------------
    elif op == 'batch_matmul':
        if len(args) < 2:
            return None, "not enough args for batch_matmul"
        a = torch.randn(*args[0])
        b = torch.randn(*args[1])

        def eager_fn():
            return torch.bmm(a, b)

        def _bmm_for_trace(x, y):
            return torch.bmm(x, y)

        jit_fn  = torch.jit.trace(_bmm_for_trace, (a, b))
        jit_run = lambda: jit_fn(a, b)

    # -----------------------------------------------------------------------
    # pooling_nchw_max / pooling_nchw_sum
    # -----------------------------------------------------------------------
    elif op in ('pooling_nchw_max', 'pooling_nchw_sum'):
        if len(args) < 2:
            return None, "not enough args for pooling"
        inp    = torch.randn(*args[0])
        kernel = args[1]          # shape (kH, kW) from the kernel tensor arg
        strides = _parse_dense(mlir_code, 'strides')

        if op == 'pooling_nchw_max':
            def eager_fn():
                return F.max_pool2d(inp, kernel_size=kernel, stride=strides)

            def _pool_for_trace(x):
                return F.max_pool2d(x, kernel_size=kernel, stride=strides)
        else:
            def eager_fn():
                return F.avg_pool2d(inp, kernel_size=kernel, stride=strides)

            def _pool_for_trace(x):
                return F.avg_pool2d(x, kernel_size=kernel, stride=strides)

        jit_fn  = torch.jit.trace(_pool_for_trace, (inp,))
        jit_run = lambda: jit_fn(inp)

    # -----------------------------------------------------------------------
    # generic  (reduction — treat as torch.sum on reduction dims)
    # -----------------------------------------------------------------------
    elif op == 'generic':
        if len(args) < 2:
            return None, "not enough args for generic"
        in_shape  = args[0]
        out_shape = args[-1]
        inp = torch.randn(*in_shape)
        reduce_dims = [i for i, (si, so) in enumerate(zip(in_shape, out_shape)) if so == 1]

        if reduce_dims:
            def eager_fn():
                return inp.sum(dim=reduce_dims, keepdim=True)

            def _sum_for_trace(x):
                return x.sum(dim=reduce_dims, keepdim=True)
        else:
            def eager_fn():
                return inp.sum()

            def _sum_for_trace(x):
                return x.sum()

        jit_fn  = torch.jit.trace(_sum_for_trace, (inp,))
        jit_run = lambda: jit_fn(inp)

    # -----------------------------------------------------------------------
    # transpose
    # -----------------------------------------------------------------------
    elif op == 'transpose':
        if len(args) < 1:
            return None, "not enough args for transpose"
        inp = torch.randn(*args[0])

        def eager_fn():
            return inp.T if inp.dim() == 2 else inp.transpose(-2, -1)

        jit_fn  = torch.jit.trace(lambda x: x.transpose(-2, -1), (inp,))
        jit_run = lambda: jit_fn(inp)

    # -----------------------------------------------------------------------
    # broadcast
    # -----------------------------------------------------------------------
    elif op == 'broadcast':
        if len(args) < 2:
            return None, "not enough args for broadcast"
        inp = torch.randn(*args[0])
        target_shape = args[-1]
        # Expand to match target ndim, then broadcast_to
        if inp.ndim < len(target_shape):
            inp = inp.reshape(*inp.shape, *(1,) * (len(target_shape) - inp.ndim))

        def eager_fn():
            return inp.expand(*target_shape)

        jit_fn  = torch.jit.trace(lambda x: x.expand(*target_shape), (inp,))
        jit_run = lambda: jit_fn(inp)

    # -----------------------------------------------------------------------
    # add  (element-wise)
    # -----------------------------------------------------------------------
    elif op == 'add':
        if len(args) < 2:
            return None, "not enough args for add"
        a = torch.randn(*args[0])
        b = torch.randn(*args[1])

        def eager_fn():
            return a + b

        jit_fn  = torch.jit.trace(lambda x, y: x + y, (a, b))
        jit_run = lambda: jit_fn(a, b)

    # -----------------------------------------------------------------------
    # fill  (constant fill)
    # -----------------------------------------------------------------------
    elif op == 'fill':
        if len(args) < 1:
            return None, "not enough args for fill"
        inp = torch.randn(*args[0])

        def eager_fn():
            return inp.fill_(1.0)

        jit_fn  = torch.jit.trace(lambda x: x.fill_(1.0), (inp,))
        jit_run = lambda: jit_fn(inp.clone())

    # -----------------------------------------------------------------------
    # reduce  (generic reduction)
    # -----------------------------------------------------------------------
    elif op == 'reduce':
        if len(args) < 2:
            return None, "not enough args for reduce"
        in_shape  = args[0]
        out_shape = args[-1]
        inp = torch.randn(*in_shape)
        reduce_dims = [i for i, (si, so) in enumerate(zip(in_shape, out_shape)) if so == 1]

        def eager_fn():
            return inp.sum(dim=reduce_dims or None, keepdim=bool(reduce_dims))

        if reduce_dims:
            jit_fn  = torch.jit.trace(lambda x: x.sum(dim=reduce_dims, keepdim=True), (inp,))
        else:
            jit_fn  = torch.jit.trace(lambda x: x.sum(), (inp,))
        jit_run = lambda: jit_fn(inp)

    # -----------------------------------------------------------------------
    # depthwise_conv_2d_nchw_chw
    # -----------------------------------------------------------------------
    elif op == 'depthwise_conv_2d_nchw_chw':
        if len(args) < 2:
            return None, "not enough args for depthwise_conv2d"
        inp    = torch.randn(*args[0])
        weight_raw = torch.randn(*args[1])
        _, in_c, _, _ = inp.shape
        # MLIR weight is (C, kH, kW) — unsqueeze to (C, 1, kH, kW) for PyTorch
        if weight_raw.ndim == 3:
            weight = weight_raw.unsqueeze(1)
        else:
            weight = weight_raw
        strides   = _parse_dense(mlir_code, 'strides')
        dilations = _parse_dense(mlir_code, 'dilations')

        def eager_fn():
            return F.conv2d(inp, weight, stride=strides, dilation=dilations, groups=in_c)

        def _conv_for_trace(x):
            return F.conv2d(x, weight, stride=strides, dilation=dilations, groups=in_c)

        jit_fn  = torch.jit.trace(_conv_for_trace, (inp,))
        jit_run = lambda: jit_fn(inp)

    # -----------------------------------------------------------------------
    # conv_2d_nhwc_hwcf
    # -----------------------------------------------------------------------
    elif op == 'conv_2d_nhwc_hwcf':
        if len(args) < 2:
            return None, "not enough args for conv2d_nhwc"
        inp    = torch.randn(*args[0])
        weight_raw = torch.randn(*args[1])
        # NHWC input (N,H,W,C) → NCHW (N,C,H,W)
        inp = inp.permute(0, 3, 1, 2)
        # HWCF weight (kH,kW,in_c,out_c) → OIHW (out_c,in_c,kH,kW)
        weight = weight_raw.permute(3, 2, 0, 1)
        strides   = _parse_dense(mlir_code, 'strides')
        dilations = _parse_dense(mlir_code, 'dilations')

        def eager_fn():
            return F.conv2d(inp, weight, stride=strides, dilation=dilations)

        def _conv_for_trace(x):
            return F.conv2d(x, weight, stride=strides, dilation=dilations)

        jit_fn  = torch.jit.trace(_conv_for_trace, (inp,))
        jit_run = lambda: jit_fn(inp)

    # -----------------------------------------------------------------------
    # pooling_ncw_* / pooling_nwc_* / pooling_nhwc_* (1D/2D pooling variants)
    # -----------------------------------------------------------------------
    elif op in ('pooling_ncw_max', 'pooling_ncw_sum',
                'pooling_nwc_max', 'pooling_nwc_sum',
                'pooling_nhwc_max', 'pooling_nhwc_sum', 'pooling_nhwc_min'):
        if len(args) < 2:
            return None, "not enough args for pooling"
        inp    = torch.randn(*args[0])
        kernel = args[1]
        strides = _parse_dense(mlir_code, 'strides')

        is_max = 'max' in op
        is_2d  = 'nhwc' in op

        if is_2d:
            pool_fn = F.max_pool2d if is_max else F.avg_pool2d
            pool_kernel = kernel
            pool_stride = strides
        else:
            pool_fn = F.max_pool1d if is_max else F.avg_pool1d
            pool_kernel = kernel[0] if isinstance(kernel, tuple) else kernel
            pool_stride = strides[0] if isinstance(strides, tuple) else strides

        def eager_fn():
            return pool_fn(inp, kernel_size=pool_kernel, stride=pool_stride)

        def _pool_for_trace(x):
            return pool_fn(x, kernel_size=pool_kernel, stride=pool_stride)

        jit_fn  = torch.jit.trace(_pool_for_trace, (inp,))
        jit_run = lambda: jit_fn(inp)

    else:
        return None, f"unsupported linalg op: {op}"

    return {
        "eager":   eager_fn,
        "jit":     jit_run,
    }, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Measure PyTorch eager/compile/jit times for MLIR benchmarks"
    )
    parser.add_argument("--require-base-success", action="store_true", help="Only run benchmarks that have > 0 time in base.json")
    parser.add_argument("--config", default=None,
                        help="Path to config JSON (derives benchmarks-dir and output)")
    parser.add_argument("--benchmarks-dir", default=None,
                        help="Override: directory containing .mlir benchmark files")
    parser.add_argument("--output", default=None,
                        help="Override: output JSON path")
    parser.add_argument("--warmup",  type=int, default=NUM_WARMUP)
    parser.add_argument("--trials",  type=int, default=NUM_TRIALS)
    parser.add_argument("--chunk-index", type=int, default=0, help="0-based index of the chunk to process")
    parser.add_argument("--num-chunks", type=int, default=1, help="Total number of chunks to split the workload into")
    args = parser.parse_args()

    eval_names: set[str] | None = None  # if set, only measure these benchmarks

    if args.config:
        with open(args.config) as _f:
            _cfg = json.load(_f)
        bench_dir   = Path(args.benchmarks_dir or _cfg["benchmarks_folder_path"])
        output_path = (Path(args.output) if args.output else \
                      Path(_cfg["results_dir"]) / "exec_times" / "pytorch.json").resolve()
        # Only measure eval benchmarks — PyTorch times are only compared against RL eval results.
        eval_json = Path(_cfg["results_dir"]) / "exec_times" / "base_eval.json"
        if eval_json.exists():
            with open(eval_json) as _f:
                eval_names = set(json.load(_f).keys())
        else:
            eval_names = None
    else:
        if not args.benchmarks_dir:
            parser.error("Provide --config or --benchmarks-dir")
        bench_dir = Path(args.benchmarks_dir)
        output_path = (Path(args.output) if args.output else \
                      Path("results") / "exec_times" / "pytorch.json").resolve()

    if not bench_dir.is_dir():
        raise SystemExit(f"Not a directory: {bench_dir}")

    all_mlir_files = sorted(bench_dir.rglob("*.mlir"))
    if eval_names is not None:
        mlir_files = [f for f in all_mlir_files if f.stem in eval_names]
        print(f"Found {len(all_mlir_files)} .mlir files in {bench_dir}; "
              f"measuring {len(mlir_files)} eval benchmarks only")
    else:
        mlir_files = all_mlir_files
        print(f"Found {len(mlir_files)} .mlir files in {bench_dir}")
    print("torch.compile is disabled — measuring eager + JIT only")

    results: dict[str, dict] = {}
    skipped = 0

    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
            print(f"Resuming from existing output file ({len(results)} entries): {output_path}", flush=True)
        except Exception as e:
            print(f"Failed to load existing {output_path}: {e}", flush=True)

    # Load base.json to filter out failures if requested
    base_successes = set()
    if args.require_base_success:
        base_path = Path(cfg["results_dir"]) / "exec_times" / "base.json"
        
        if base_path.exists():
            with open(base_path, 'r') as f:
                base_data = json.load(f)
            base_successes = {k for k, v in base_data.items() if isinstance(v, (int, float)) and v > 0}
            print(f"Loaded {len(base_successes)} successful benchmarks from {base_path}")
        else:
            print(f"Warning: --require-base-success passed but {base_path} not found.")

    remaining_files = []
    for f in mlir_files:
        already_done = f.stem in results and results[f.stem].get("eager") not in (None, "N/A") and not str(results[f.stem].get("eager")).startswith("FAILED")
        
        if not already_done:
            if args.require_base_success and f.stem not in base_successes:
                continue # Skip files that failed in MLIR base.json
            remaining_files.append(f)

    print(f"Remaining benchmarks to process: {len(remaining_files)} / {len(mlir_files)}")

    # Partition into chunks
    if args.num_chunks > 1:
        chunk_size = len(remaining_files) // args.num_chunks
        start_idx = args.chunk_index * chunk_size
        end_idx = start_idx + chunk_size if args.chunk_index < args.num_chunks - 1 else len(remaining_files)
        remaining_files = remaining_files[start_idx:end_idx]
        old_out = Path(output_path)
        output_path = old_out.parent / f"{old_out.stem}_chunk{args.chunk_index}{old_out.suffix}"
        print(f"-- Processing chunk {args.chunk_index + 1}/{args.num_chunks} -- Files {start_idx} to {end_idx - 1}")

    def _try_measure(fn, label):
        if fn is None:
            return None
        try:
            return measure_ns(fn, warmup=args.warmup, trials=args.trials)
        except Exception as exc:
            return f"FAILED: {exc}"

    def process_file(args_tuple):
        i, mlir_file = args_tuple
        bench_name = mlir_file.stem
        code = mlir_file.read_text()

        fns, err = build_pytorch_fns(code)
        if fns is None:
            return i, bench_name, None, err

        entry = {}
        entry["eager"]   = _try_measure(fns["eager"],   "eager")
        entry["jit"]     = _try_measure(fns["jit"],     "jit")

        if (entry["eager"] is None or str(entry["eager"]).startswith("FAILED")) and (entry["jit"] is None or str(entry["jit"]).startswith("FAILED")):
            return i, bench_name, None, "all modes failed"

        return i, bench_name, entry, None

    process_args = [(i, f) for i, f in enumerate(remaining_files)]

    # Process sequentially to avoid OOM from multiple large tensors
    for arg in process_args:
        i, f = arg
        bench_name = f.stem
        try:
            _, _, entry, err = process_file(arg)
        except Exception as e:
            err = f"Uncaught Error: {e}"
            entry = None

        if err:
            print(f"[{i+1}/{len(remaining_files)}] SKIP {bench_name}: {err}", flush=True)
            skipped += 1
        else:
            formatted_entry = {}
            for k, v in entry.items():
                if isinstance(v, int):
                    formatted_entry[k] = v
                else:
                    formatted_entry[k] = "N/A"
            results[bench_name] = formatted_entry
            print(f"[{i+1}/{len(remaining_files)}] {bench_name}: "
                  f"eager={formatted_entry.get('eager')}ns  "
                  f"jit={formatted_entry.get('jit')}ns", flush=True)

            if (i+1) % 10 == 0:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f_out:
                    json.dump(results, f_out, indent=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} measured, {skipped} skipped.", flush=True)
    print(f"Output: {output_path}", flush=True)


if __name__ == "__main__":
    main()
