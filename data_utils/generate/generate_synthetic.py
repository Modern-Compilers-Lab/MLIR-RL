#!/usr/bin/env python3
"""
generate_synthetic.py
---------------------
Generate synthetic MLIR benchmarks in the legacy-compatible flat-file format.

Outputs individual .mlir files:
  - single_bench_{N}.mlir  →  data/all/code_files/single_bench/
  - bench_{N}.mlir          →  data/all/code_files/bench/

Single ops are drawn from LINALG_OPERATION_GENERATORS.
Bench blocks are produced by randomSubGraph (chains of 5 ops with
consumer→producer links and intermediate buffer allocations), then
inlined into the legacy @main wrapper.

IDs continue sequentially from the maximum found across all known roots
using id_allocator.py.
"""

from __future__ import annotations

import argparse
import os
import re
import string
import sys
import traceback
import yaml
from random import choice, randint, seed as _set_seed
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Maximum tensor elements allowed in any generated benchmark.
# ~500M f32 elements ≈ 2 GiB. The NN model max is ~348M (1.3 GiB).
MAX_TENSOR_ELEMENTS = 500_000_000


# ---------------------------------------------------------------------------
# Shape extraction (mirrors mlir_generators.getShapes_Args, self-contained)
# ---------------------------------------------------------------------------

def _remove_duplicate_args(args: list[str], shapes: list[str]):
    seen: set[tuple[str, str]] = set()
    out_args: list[str] = []
    out_shapes: list[str] = []
    for a, s in zip(args, shapes):
        pair = (a.strip(), s.strip())
        if pair not in seen:
            seen.add(pair)
            out_args.append(a.strip())
            out_shapes.append(s.strip())
    return out_args, out_shapes


def _get_shapes_args(operation: str):
    """Extract (arg_names, type_strings) from a linalg op text."""
    ins_outs_pattern = r"(?:ins|outs)\s*\(([^())]+)\)"
    fields = re.findall(ins_outs_pattern, operation)

    if not fields:
        alt = re.findall(r"(?:\(([^(]+)\))(?:\s*\->\s*([^(]+))", operation)
        if not alt:
            return [], []
        fields = [alt[0]]
        args, shapes = [], []
        for f in fields[0][0].split(", "):
            shapes.append(f)
        try:
            raw_args = re.findall(r"(?:@\w+\(([^)]+))", operation)[0].split(",")
            args = [a.strip() for a in raw_args]
        except (IndexError, AttributeError):
            args = [f"%arg{i}" for i in range(len(shapes))]
        shapes = [s.strip() for s in shapes]
    else:
        args, shapes = [], []
        for field in fields:
            args_field, shapes_field = field.split(":")
            args += args_field.split(",")
            shapes += shapes_field.split(",")
        args = [a.strip() for a in args]
        shapes = [s.strip() for s in shapes]
        args, shapes = _remove_duplicate_args(args, shapes)

    return args, shapes


# ---------------------------------------------------------------------------
# Legacy wrapping: single op
# ---------------------------------------------------------------------------

def _extract_result_type(op_text: str) -> str:
    """Extract the result type from '-> type' in the operation text."""
    # Handle multi-line ops: check last line first
    last_line = op_text.split("\n")[-1]
    m = re.search(r"->\s*(tensor<[^>]+(?:>[^>]*)*>)", last_line)
    if m:
        return m.group(1)
    return ""


def _legacy_wrap_single(operation: str, maps: str = "", additional_function: str = "") -> str:
    """Wrap a single linalg op in legacy benchmark format.

    Module includes @nanoTime, @printI64, @printF32, @printNewline.
    @main receives ALL tensor args (ins + outs init tensors),
    times one execution, returns (result, elapsed_ns).
    """
    args, shapes = _get_shapes_args(operation)

    if len(args) < 1 or len(shapes) < 1:
        raise ValueError("Cannot extract sufficient args/shapes from operation")

    result_type = _extract_result_type(operation)
    if not result_type and shapes:
        result_type = shapes[-1]  # fallback

    # shapes = [ins_types..., outs_types...]
    # ALL of these become @main arguments
    all_arg_types = list(shapes)
    all_args = args

    # Build rename map: original SSA → %argN
    arg_map: dict[str, str] = {}
    for i, orig in enumerate(all_args):
        arg_map[orig.strip()] = f"%arg{i}"

    op_text = operation

    # Add %result = prefix if the op doesn't already have an SSA assignment
    if not re.match(r'\s*%\w+\s*=\s*linalg\.', op_text) and \
       not re.match(r'\s*%\w+\s*=\s*func\.call', op_text):
        op_text = f"%result = {op_text}"
    else:
        # Rename existing SSA assignment to %result
        op_text = re.sub(r'^(\s*)%\w+(\s*=\s*)', r'\1%result\2', op_text, count=1)

    # Rename arg SSA refs (longest first to avoid prefix collisions)
    for orig in sorted(arg_map, key=len, reverse=True):
        op_text = re.sub(
            r"(?<!\w)" + re.escape(orig) + r"(?!\w)",
            arg_map[orig],
            op_text,
        )

    # Extract helper function definitions from additional_function (e.g. softmax)
    helper_funcs: list[str] = []
    if additional_function:
        # Find all func.func private @XXX definitions and extract their full bodies
        # using brace-matching (handles nested {} in linalg.generic bodies).
        helper_pattern = re.compile(
            r'func\.func\s+private\s+@(\w+)\s*\([^{]*\)\s*->\s*\S+',
            re.DOTALL,
        )
        for m in helper_pattern.finditer(additional_function):
            start = m.start()
            brace_pos = additional_function.find('{', m.end())
            if brace_pos == -1:
                continue
            depth = 0
            end_pos = brace_pos
            for i in range(brace_pos, len(additional_function)):
                if additional_function[i] == '{':
                    depth += 1
                elif additional_function[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break
            if end_pos > brace_pos:
                helper_funcs.append(additional_function[start:end_pos])

    # Assemble module
    header_lines: list[str] = []
    if maps and maps.strip():
        header_lines.extend(maps.strip().split("\n"))

    body: list[str] = []
    body.append('module attributes {torch.debug_module_name = "Net"} {')
    body.append("  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}")
    body.append("  func.func private @printI64(i64)")
    body.append("  func.func private @printF32(f32)")
    body.append("  func.func private @printNewline()")

    # Include helper function definitions
    for hf in helper_funcs:
        for line in hf.strip().split("\n"):
            body.append(f"  {line.strip()}")

    arg_decls = ", ".join(
        f"%arg{i}: {all_arg_types[i]}" for i in range(len(all_arg_types))
    )
    body.append(
        f"  func.func @main({arg_decls}) -> ({result_type}, i64)"
        f' attributes {{llvm.emit_c_interface}} {{'
    )
    body.append(f"    %__t_start = call @nanoTime() : () -> i64")

    # Indent op text
    for line in op_text.split("\n"):
        stripped = line.rstrip()
        if stripped:
            body.append(f"    {stripped}")

    body.append(f"    %__t_end = call @nanoTime() : () -> i64")
    body.append(f"    %__delta = arith.subi %__t_end, %__t_start : i64")
    body.append(f"    return %result, %__delta : {result_type}, i64")
    body.append(f"  }}")
    body.append(f"}}")

    return "\n".join(header_lines + body)


# ---------------------------------------------------------------------------
# Legacy wrapping: bench block (from randomSubGraph)
# ---------------------------------------------------------------------------

def _legacy_wrap_bench(operation_text: str, maps: str, additional_function: str) -> str:
    """Wrap a randomSubGraph result into an inlined legacy bench block.

    Parses the @myFunction definition and inlines its body into @main,
    preserving bufferization.alloc_tensor calls for intermediate tensors.
    SSA variables in the inlined body are renamed to avoid conflicts with
    the timing SSA names used by the wrapper.
    """
    # --- Parse the call site ---
    call_match = re.search(
        r'func\.call @(\w+)\(([^)]*)\)\s*:\s*\(([^)]*)\)\s*->\s*(\S+)',
        operation_text,
    )
    if not call_match:
        raise ValueError(f"Cannot parse subgraph call:\n{operation_text}")

    func_name = call_match.group(1)
    call_args = [a.strip() for a in call_match.group(2).split(",") if a.strip()]

    # Parse call arg types (handle nested < >)
    call_arg_types: list[str] = []
    depth = 0
    current = ""
    for ch in call_match.group(3):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "," and depth == 0:
            call_arg_types.append(current.strip())
            current = ""
            continue
        current += ch
    if current.strip():
        call_arg_types.append(current.strip())

    result_type = call_match.group(4)

    # --- Parse the @myFunction definition ---
    func_match = re.search(
        rf'func\.func private @{re.escape(func_name)}\(([^)]*)\)\s*->\s*(\S+)\s*\{{',
        additional_function,
    )
    if not func_match:
        raise ValueError(f"Cannot find @{func_name} definition in additional_function")

    # Find matching closing brace for the function body
    func_body_start = func_match.end()
    brace_depth = 0
    func_body_end = func_body_start
    for i in range(func_body_start, len(additional_function)):
        if additional_function[i] == "{":
            brace_depth += 1
        elif additional_function[i] == "}":
            if brace_depth == 0:
                func_body_end = i
                break
            brace_depth -= 1

    func_body = additional_function[func_body_start:func_body_end].strip()
    func_params_str = func_match.group(1)

    # --- Extract params ---
    func_params = []
    for p in func_params_str.split(","):
        p = p.strip()
        if ":" in p:
            name, ptype = p.split(":", 1)
            func_params.append((name.strip(), ptype.strip()))
        else:
            func_params.append((p, ""))

    # --- SSA renaming to avoid conflicts ---
    # Reserved names used by the wrapper
    RESERVED = {"%__t_start", "%__t_end", "%__delta", "%result"}
    # Collect all SSA names defined in the body
    all_ssa_in_body = set(re.findall(r'%\w+', func_body))

    # Build rename map: params → %argN, plus conflict-avoidance for other SSAs
    ssa_rename: dict[str, str] = {}
    for i, (pname, _) in enumerate(func_params):
        if i < len(call_args):
            ssa_rename[pname.strip()] = f"%arg{i}"

    # Rename any body SSA that conflicts with reserved wrapper names
    conflict_counter = 1000
    for ssa in sorted(all_ssa_in_body, key=len, reverse=True):
        if ssa in ssa_rename:
            continue  # already mapped
        if ssa in RESERVED:
            ssa_rename[ssa] = f"%__s{conflict_counter}"
            conflict_counter += 1

    # Apply renames (longest first to avoid prefix collisions)
    body_text = func_body
    for orig in sorted(ssa_rename, key=len, reverse=True):
        body_text = re.sub(
            r"(?<!\w)" + re.escape(orig) + r"(?!\w)",
            ssa_rename[orig],
            body_text,
        )

    # Extract the return statement to get result SSA and type
    return_match = re.search(r'return\s+(%\w+)\s*:\s*(\S+)', body_text)
    if return_match:
        result_ssa = return_match.group(1)
        return_type = return_match.group(2)
        body_text = body_text[:return_match.start()].rstrip() + "\n" + body_text[return_match.end():]
        body_text = body_text.rstrip()
        # Rename the result SSA to %result
        if result_ssa != "%result":
            body_text = re.sub(
                r"(?<!\w)" + re.escape(result_ssa) + r"(?!\w)",
                "%result",
                body_text,
            )
    else:
        return_type = result_type

    # --- Extract helper function definitions (e.g. softmax) from additional_function ---
    # These are func.func private @XYZ(...) definitions that are NOT @myFunction.
    # The @myFunction body inlines them via func.call, so we must include them.
    helper_funcs: list[str] = []
    helper_pattern = re.compile(
        r'func\.func\s+private\s+@(\w+)\s*\([^{]*\)\s*->\s*\S+',
        re.DOTALL,
    )
    for m in helper_pattern.finditer(additional_function):
        helper_name = m.group(1)
        if helper_name == func_name:
            continue  # skip @myFunction, already inlined
        start = m.start()
        brace_pos = additional_function.find('{', m.end())
        if brace_pos == -1:
            continue
        depth = 0
        end_pos = brace_pos
        for i in range(brace_pos, len(additional_function)):
            if additional_function[i] == '{':
                depth += 1
            elif additional_function[i] == '}':
                depth -= 1
                if depth == 0:
                    end_pos = i + 1
                    break
        if end_pos > brace_pos:
            helper_funcs.append(additional_function[start:end_pos])

    # --- Assemble the module ---
    output: list[str] = []
    if maps and maps.strip():
        output.extend(maps.strip().split("\n"))

    output.append('module attributes {torch.debug_module_name = "Net"} {')
    output.append("  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}")
    output.append("  func.func private @printI64(i64)")
    output.append("  func.func private @printF32(f32)")
    output.append("  func.func private @printNewline()")

    # Include helper function definitions (e.g. softmax)
    for hf in helper_funcs:
        # Indent to match module scope (2 spaces)
        for line in hf.strip().split("\n"):
            output.append(f"  {line.strip()}")

    arg_decls = ", ".join(
        f"%arg{i}: {call_arg_types[i]}" for i in range(len(call_arg_types))
    )
    output.append(
        f"  func.func @main({arg_decls}) -> ({return_type}, i64)"
        f' attributes {{llvm.emit_c_interface}} {{'
    )
    output.append(f"    %__t_start = call @nanoTime() : () -> i64")

    # Append inlined body (already indented with 4 spaces from the function def)
    for line in body_text.split("\n"):
        stripped = line.rstrip()
        if stripped:
            output.append(f"    {stripped}")

    output.append(f"    %__t_end = call @nanoTime() : () -> i64")
    output.append(f"    %__delta = arith.subi %__t_end, %__t_start : i64")
    output.append(f"    return %result, %__delta : {return_type}, i64")
    output.append(f"  }}")
    output.append(f"}}")

    return "\n".join(output)


# ---------------------------------------------------------------------------
# Generator helpers
# ---------------------------------------------------------------------------

def _unpack_generator_result(res: Any):
    """Normalise generator return into (raw_op, maps, additional_function)."""
    maps = ""
    additional_function = ""
    if isinstance(res, tuple):
        raw_operation, extra = res
        if isinstance(extra, tuple) and len(extra) == 2:
            maps, additional_function = extra
        elif isinstance(extra, str):
            maps = extra
    else:
        raw_operation = res
    return raw_operation, maps, additional_function


def _load_shape_config(config_path: str | None):
    """Load optional YAML config that extends mlir_generators shape globals."""
    from data_utils.generate.mlir_generators import (
        BATCH_SIZES, HEIGHTS, CHANNELS, KERNELS, DILATIONS, STRIDES, SIZES,
    )
    if config_path and os.path.isfile(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        shapes = cfg.get("SHAPES", {})
        for pool_name, pool in [
            ("BATCH_SIZES", BATCH_SIZES),
            ("HEIGHTS", HEIGHTS),
            ("CHANNELS", CHANNELS),
            ("KERNELS", KERNELS),
            ("DILATIONS", DILATIONS),
            ("STRIDES", STRIDES),
            ("SIZES", SIZES),
        ]:
            extra = shapes.get(pool_name, [])
            if extra:
                pool.extend(extra)


def _generate_single(operation_name: str, generator) -> str | None:
    """Generate one single-op benchmark; return MLIR text or None on failure."""
    try:
        res = generator()
        operation, maps, additional_function = _unpack_generator_result(res)
        mlir = _legacy_wrap_single(operation, maps, additional_function)
        if mlir is None:
            return None
        if _exceeds_tensor_limit(mlir):
            return None  # silently retry
        return mlir
    except Exception:
        traceback.print_exc()
        return None


def _generate_bench() -> str | None:
    """Generate one bench block using randomSubGraph; return MLIR text or None."""
    from data_utils.generate.mlir_generators import randomSubGraph

    try:
        res = randomSubGraph()
        if isinstance(res, tuple) and len(res) == 2:
            operation = res[0]
            extra = res[1]
            if isinstance(extra, tuple) and len(extra) == 2:
                maps, additional_function = extra
            elif isinstance(extra, str):
                maps = extra
                additional_function = ""
            else:
                maps = ""
                additional_function = ""
        else:
            operation = res
            maps = ""
            additional_function = ""

        mlir = _legacy_wrap_bench(operation, maps, additional_function)
        if mlir is None:
            return None
        if _exceeds_tensor_limit(mlir):
            return None  # silently retry
        return mlir
    except Exception:
        traceback.print_exc()
        return None


def _exceeds_tensor_limit(mlir_text: str) -> bool:
    """True if any tensor in *mlir_text* exceeds MAX_TENSOR_ELEMENTS."""
    import math
    for m in re.finditer(r'tensor<([\dx]+)x[fi]\d+>', mlir_text):
        try:
            dims = [int(x) for x in m.group(1).split('x')]
            if math.prod(dims) > MAX_TENSOR_ELEMENTS:
                return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------------------
# ID allocation
# ---------------------------------------------------------------------------

def _get_next_ids():
    """Return (next_single_id, next_bench_id)."""
    from data_utils.generate.id_allocator import build_id_space_state

    state = build_id_space_state(
        legacy_single_dir=os.path.join(PROJECT_ROOT, "data", "all", "code_files", "single_bench"),
        legacy_bench_dir=os.path.join(PROJECT_ROOT, "data", "all", "code_files", "bench"),
        synthetic_single_dir=os.path.join(PROJECT_ROOT, "data", "nn", "synthetic", "single"),
        synthetic_bench_dir=os.path.join(PROJECT_ROOT, "data", "nn", "synthetic", "benchs"),
    )
    return state.single_next, state.bench_next


def _write_file(directory: str, filename: str, content: str) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic MLIR benchmarks in legacy format."
    )
    parser.add_argument("--num-single", type=int, default=100,
                        help="Number of single-op benchmarks (default: 100).")
    parser.add_argument("--num-bench", type=int, default=50,
                        help="Number of bench block benchmarks (default: 50).")
    parser.add_argument("--output-single-dir",
                        default=os.path.join(PROJECT_ROOT, "data", "all", "code_files", "single_bench"))
    parser.add_argument("--output-bench-dir",
                        default=os.path.join(PROJECT_ROOT, "data", "all", "code_files", "bench"))
    parser.add_argument("--config", default=None,
                        help="Optional YAML config extending shape pools.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("--ops", nargs="+", default=None,
                        help="Specific generator names (default: all in LINALG_OPERATION_GENERATORS).")
    parser.add_argument("--skip-single", action="store_true", default=False)
    parser.add_argument("--skip-bench", action="store_true", default=False)
    args = parser.parse_args()

    if args.seed is not None:
        _set_seed(args.seed)

    _load_shape_config(args.config)

    from data_utils.generate.mlir_generators import LINALG_OPERATION_GENERATORS as GEN_MAP

    # Select generators
    if args.ops:
        active_gens = {n: GEN_MAP[n] for n in args.ops if n in GEN_MAP}
        if not active_gens:
            print(f"ERROR: None of the requested ops found: {args.ops}")
            sys.exit(1)
    else:
        active_gens = dict(GEN_MAP)

    gen_names = sorted(active_gens)

    single_next, bench_next = _get_next_ids()
    print(f"Next single ID : {single_next}")
    print(f"Next bench  ID : {bench_next}")
    print(f"Generators     : {gen_names}")

    # --- Single ops ---
    sid = single_next
    bid = bench_next

    if not args.skip_single and args.num_single > 0:
        print(f"\n=== Generating {args.num_single} single-op benchmarks ===")
        written = 0
        attempts = 0
        max_attempts = args.num_single * 20

        while written < args.num_single and attempts < max_attempts:
            attempts += 1
            op_name = choice(gen_names)
            gen = active_gens[op_name]
            mlir = _generate_single(op_name, gen)
            if mlir is None:
                continue
            _write_file(args.output_single_dir, f"single_bench_{sid}.mlir", mlir)
            written += 1
            sid += 1
            if written % 10 == 0:
                print(f"  {written}/{args.num_single} singles written")

        print(f"  Done: {written} singles → {args.output_single_dir}")

    # --- Bench blocks ---
    if not args.skip_bench and args.num_bench > 0:
        print(f"\n=== Generating {args.num_bench} bench blocks ===")
        written = 0
        attempts = 0
        max_attempts = args.num_bench * 20

        while written < args.num_bench and attempts < max_attempts:
            attempts += 1
            mlir = _generate_bench()
            if mlir is None:
                continue
            _write_file(args.output_bench_dir, f"bench_{bid}.mlir", mlir)
            written += 1
            bid += 1
            if written % 10 == 0:
                print(f"  {written}/{args.num_bench} benche written")

        print(f"  Done: {written} benche → {args.output_bench_dir}")

    print(f"\nDone. Next available IDs: single={sid}, bench={bid}")


if __name__ == "__main__":
    main()
