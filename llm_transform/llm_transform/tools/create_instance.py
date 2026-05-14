"""Create a new instance of an existing benchmark in the data/ directory.

Writes a new ``<id>.mlir`` under ``data/<benchmark>/`` and updates
``sizes.json``. The id auto-picks the next free integer (override with ``--id``).

Sizes: always check ``data/<benchmark>/sizes.json`` for the expected
parameters. Pass ``key=value`` pairs when entries are dicts, or bare
positional ints when entries are lists.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def out_dim(in_dim: int, k: int, s: int) -> int:
    return (in_dim - k) // s + 1


def render_matmul(s: dict) -> str:
    M, K, N = s["M"], s["K"], s["N"]
    return f"""func.func private @nanoTime() -> i64 attributes {{llvm.emit_c_interface}}
func.func @main(%arg0: tensor<{M}x{K}xf64>, %arg1: tensor<{K}x{N}xf64>) -> (tensor<{M}x{N}xf64>, i64) attributes {{llvm.emit_c_interface}} {{
    %c0 = arith.constant 0.0 : f64
    %new = tensor.empty() : tensor<{M}x{N}xf64>
    %arg2 = linalg.fill ins(%c0 : f64) outs(%new : tensor<{M}x{N}xf64>) -> tensor<{M}x{N}xf64>
    %0 = call @nanoTime() : () -> i64
    %1 = linalg.matmul {{tag = "operation"}} ins(%arg0, %arg1 : tensor<{M}x{K}xf64>, tensor<{K}x{N}xf64>) outs(%arg2 : tensor<{M}x{N}xf64>) -> tensor<{M}x{N}xf64>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %1, %3 : tensor<{M}x{N}xf64>, i64
}}
"""


def render_add(s: list) -> str:
    shape = "x".join(str(d) for d in s)
    return f"""module {{
  func.func private @nanoTime() -> i64 attributes {{llvm.emit_c_interface}}
  func.func @main(%arg0: tensor<{shape}xf64>, %arg1: tensor<{shape}xf64>) -> (tensor<{shape}xf64>, i64) attributes {{llvm.emit_c_interface}} {{
    %c0 = arith.constant 0.0 : f64
    %new = tensor.empty() : tensor<{shape}xf64>
    %arg2 = linalg.fill ins(%c0 : f64) outs(%new : tensor<{shape}xf64>) -> tensor<{shape}xf64>
    %0 = call @nanoTime() : () -> i64
    %1 = linalg.add {{tag = "operation"}} ins(%arg0, %arg1 : tensor<{shape}xf64>, tensor<{shape}xf64>) outs(%arg2 : tensor<{shape}xf64>) -> tensor<{shape}xf64>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %1, %3 : tensor<{shape}xf64>, i64
  }}
}}
"""


def render_conv_2d(s: dict) -> str:
    N, C, H, W = s["N"], s["C"], s["H"], s["W"]
    F, KH, KW, S = s["F"], s["KH"], s["KW"], s["S"]
    OH, OW = out_dim(H, KH, S), out_dim(W, KW, S)
    return f"""module {{
  func.func private @nanoTime() -> i64 attributes {{llvm.emit_c_interface}}
  func.func @main(%arg0: tensor<{N}x{C}x{H}x{W}xf64>, %arg1: tensor<{F}x{C}x{KH}x{KW}xf64>) -> (tensor<{N}x{F}x{OH}x{OW}xf64>, i64) attributes {{llvm.emit_c_interface}} {{
    %c0 = arith.constant 0.0 : f64
    %new = tensor.empty() : tensor<{N}x{F}x{OH}x{OW}xf64>
    %arg2 = linalg.fill ins(%c0 : f64) outs(%new : tensor<{N}x{F}x{OH}x{OW}xf64>) -> tensor<{N}x{F}x{OH}x{OW}xf64>
    %0 = call @nanoTime() : () -> i64
    %1 = linalg.conv_2d_nchw_fchw {{tag = "operation", dilations = dense<1> : tensor<2xi64>, strides = dense<{S}> : tensor<2xi64>}} ins(%arg0, %arg1 : tensor<{N}x{C}x{H}x{W}xf64>, tensor<{F}x{C}x{KH}x{KW}xf64>) outs(%arg2 : tensor<{N}x{F}x{OH}x{OW}xf64>) -> tensor<{N}x{F}x{OH}x{OW}xf64>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %1, %3 : tensor<{N}x{F}x{OH}x{OW}xf64>, i64
  }}
}}
"""


def render_pooling(s: dict) -> str:
    N, C, H, W = s["N"], s["C"], s["H"], s["W"]
    KH, KW, S = s["KH"], s["KW"], s["S"]
    OH, OW = out_dim(H, KH, S), out_dim(W, KW, S)
    return f"""module {{
  func.func private @nanoTime() -> i64 attributes {{llvm.emit_c_interface}}
  func.func @main(%arg0: tensor<{N}x{C}x{H}x{W}xf64>, %arg1: tensor<{KH}x{KW}xf64>) -> (tensor<{N}x{C}x{OH}x{OW}xf64>, i64) attributes {{llvm.emit_c_interface}} {{
    %c0 = arith.constant 0.0 : f64
    %new = tensor.empty() : tensor<{N}x{C}x{OH}x{OW}xf64>
    %arg2 = linalg.fill ins(%c0 : f64) outs(%new : tensor<{N}x{C}x{OH}x{OW}xf64>) -> tensor<{N}x{C}x{OH}x{OW}xf64>
    %0 = call @nanoTime() : () -> i64
    %1 = linalg.pooling_nchw_max {{tag = "operation", dilations = dense<1> : tensor<2xi64>, strides = dense<{S}> : tensor<2xi64>}} ins(%arg0, %arg1 : tensor<{N}x{C}x{H}x{W}xf64>, tensor<{KH}x{KW}xf64>) outs(%arg2 : tensor<{N}x{C}x{OH}x{OW}xf64>) -> tensor<{N}x{C}x{OH}x{OW}xf64>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %1, %3 : tensor<{N}x{C}x{OH}x{OW}xf64>, i64
  }}
}}
"""


# Each entry: (required keys or None for positional list, renderer, sizes-as-list flag).
BENCHMARKS = {
    "matmul":  (["M", "K", "N"], render_matmul, False),
    "add":     (4, render_add, True),
    "conv_2d": (["N", "C", "H", "W", "F", "KH", "KW", "S"], render_conv_2d, False),
    "pooling": (["N", "C", "H", "W", "KH", "KW", "S"], render_pooling, False),
}


def parse_sizes(bench: str, raw: list[str]):
    spec, _, as_list = BENCHMARKS[bench]
    if as_list:
        if len(raw) != spec:
            raise ValueError(f"benchmark '{bench}' expects {spec} positional sizes, got {len(raw)}")
        return [int(x) for x in raw]
    sizes = {}
    for tok in raw:
        if "=" not in tok:
            raise ValueError(f"expected key=value, got '{tok}'")
        k, v = tok.split("=", 1)
        sizes[k] = int(v)
    missing = [k for k in spec if k not in sizes]
    extra = [k for k in sizes if k not in spec]
    if missing:
        raise ValueError(f"missing sizes: {missing}")
    if extra:
        raise ValueError(f"unexpected sizes: {extra}")
    return sizes


def next_instance_id(bench_dir: Path) -> int:
    ids = [int(p.stem) for p in bench_dir.glob("*.mlir") if p.stem.isdigit()]
    return max(ids, default=0) + 1


def main():
    p = argparse.ArgumentParser(description="Create a new benchmark instance.")
    p.add_argument("benchmark", choices=sorted(BENCHMARKS), help="Benchmark name")
    p.add_argument("sizes", nargs="+",
                   help="Sizes matching data/<benchmark>/sizes.json: key=value pairs or positional ints.")
    p.add_argument("--id", type=int, default=None,
                   help="Instance id (default: next free integer).")
    args = p.parse_args()

    bench_dir = DATA_DIR / args.benchmark
    if not bench_dir.is_dir():
        sys.exit(f"benchmark dir not found: {bench_dir}")

    sizes = parse_sizes(args.benchmark, args.sizes)
    _, render, _ = BENCHMARKS[args.benchmark]

    sizes_path = bench_dir / "sizes.json"
    sizes_json = json.loads(sizes_path.read_text()) if sizes_path.exists() else {}

    inst_id = args.id if args.id is not None else next_instance_id(bench_dir)
    mlir_path = bench_dir / f"{inst_id}.mlir"

    mlir_path.write_text(render(sizes))
    sizes_json[str(inst_id)] = sizes
    sizes_path.write_text(json.dumps(sizes_json, indent=4) + "\n")

    print(f"Created {mlir_path} (id={args.benchmark}_{inst_id})")
    print(f"Updated {sizes_path}")


if __name__ == "__main__":
    main()
