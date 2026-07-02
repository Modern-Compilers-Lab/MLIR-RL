#!/usr/bin/env python3
"""
extract_blocks.py
-----------------
Extract sequence blocks (benchs) from full-model MLIR using consumer->producer
graph paths emitted by AST dumper.

Outputs:
- benchmark MLIR files
- extraction manifest JSON

Notes:
- No execution-time measurement is performed here.
- Dynamic batch handling defaults to heuristic selection from block complexity.
- Optional fallback is explicit: --batch-fallback mean
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from data_utils.extract.batch_policy import (
    BatchPolicyError,
    block_complexity_from_op_counts,
    select_batches,
)


# Import helper utilities from the existing single-op extractor to avoid drift.
from data_utils.extract.extract_ops import (  # pylint: disable=import-error
    _parse_section,
    _extract_result_type,
    _replace_section_inplace,
    _specialize,
    _has_dynamic,
    _get_referenced_maps,
    _get_needed_constants,
    _has_elided_tensor,
    _normalized_op_signature,
    _count_reduction_loops,
    _find_closing_paren,
    _collect_generic,
)


@dataclass
class ParsedOperation:
    tag: str
    operation_name: str
    op_count: dict[str, int]
    producers: list[tuple[str, int]]
    op_text: Optional[str] = None


@dataclass
class ParsedAst:
    full_code: str
    operations: dict[str, ParsedOperation]
    operation_tags: list[str]
    map_defs: dict[str, str]
    constants: dict[str, str]


class BlockBuildSkip(RuntimeError):
    """Known, non-fatal reason for skipping one candidate block."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


_HEAVY_OP_PATTERNS = re.compile(
    r"linalg\.(matmul|batch_matmul|conv_2d|pooling_)",
)


def _block_has_heavy_op(op_texts: list[str]) -> bool:
    """Return True if any op in *op_texts* is a non-trivial scheduling target.
    
    Heavy: matmul, batch_matmul, conv_2d, pooling, or generic with reduction loops.
    Light: element-wise linalg.generic (zero reduction loops).
    """
    for text in op_texts:
        if _HEAVY_OP_PATTERNS.search(text):
            return True
        if "linalg.generic" in text and _count_reduction_loops(text) > 0:
            return True
    return False


def _collect_linalg_ops_from_full_code(full_code: str) -> list[str]:
    """Collect linalg op text snippets in lexical order from full_code."""
    lines = full_code.splitlines()
    i = 0
    op_texts: list[str] = []

    while i < len(lines):
        line = lines[i]

        if "linalg." not in line or "linalg.yield" in line:
            i += 1
            continue

        if re.search(r"linalg\.generic\b", line):
            op_text, end_i = _collect_generic(lines, i)
        elif re.search(r"linalg\.\w+\b", line):
            op_text = line
            end_i = i
        else:
            i += 1
            continue

        op_texts.append(op_text)
        i = end_i + 1

    return op_texts


def _run_ast_dumper(file_path: str) -> str:
    ast_dumper = os.getenv("AST_DUMPER_BIN_PATH")
    if not ast_dumper:
        raise RuntimeError(
            "AST_DUMPER_BIN_PATH is not set. Please configure it before block extraction."
        )

    proc = subprocess.run(
        f"{ast_dumper} {file_path}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    out = proc.stdout.decode("utf-8")
    err = proc.stderr.decode("utf-8")

    if proc.returncode != 0:
        raise RuntimeError(f"AST dumper failed for {file_path}: {err}")
    if "#BEGIN_GRAPH" not in out:
        raise RuntimeError(f"AST dumper output missing #BEGIN_GRAPH for {file_path}")

    return out


def _parse_op_count(ops_count_str: str) -> dict[str, int]:
    op_count: dict[str, int] = {}
    for line in ops_count_str.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split(" ")
        if len(parts) == 2:
            op_count[parts[0]] = int(parts[1])
    return op_count


def _parse_ast(raw_ast_info: str) -> ParsedAst:
    if "########################################" in raw_ast_info:
        info, full_code = raw_ast_info.split("########################################", 1)
    else:
        info, full_code = raw_ast_info, ""

    operations_lines, graph_str = info.split("#BEGIN_GRAPH", 1)

    map_defs: dict[str, str] = {}
    constants: dict[str, str] = {}
    for line in full_code.splitlines():
        m = re.match(r"^(#\w+)\s*=\s*affine_map<", line)
        if m:
            map_defs[m.group(1)[1:]] = line
            continue

        m = re.match(r"\s+(%\w+)\s*=\s*(arith\.constant\s+.+)", line)
        if m:
            constants[m.group(1)] = f"    {m.group(1)} = {m.group(2)}"

    operations: dict[str, ParsedOperation] = {}
    operation_tags: list[str] = []

    blocks = [b.strip() for b in operations_lines.split("#START_OPERATION") if b.strip()]
    for block in blocks:
        rest, operation_tag = block.split("#START_TAG", 1)
        operation_tag = operation_tag.strip().split("\n")[0]

        operation_name, rest = rest.split("#START_VECTORIZABLE", 1)
        operation_name = operation_name.strip()

        _vectorizable, rest = rest.split("#START_NESTED_LOOPS", 1)
        _nested_loops, rest = rest.split("#START_LOAD_DATA", 1)
        _loads, rest = rest.split("#START_STORE_DATA", 1)
        _stores, ops_count_str = rest.split("#START_OP_COUNT", 1)

        operation_tags.append(operation_tag)
        operations[operation_tag] = ParsedOperation(
            tag=operation_tag,
            operation_name=operation_name,
            op_count=_parse_op_count(ops_count_str),
            producers=[],
        )

    graph_clean = graph_str.replace("#END_GRAPH", "")
    for line in graph_clean.strip().split("\n"):
        if not line.strip() or " --> " not in line:
            continue

        left, right = line.split(" --> ")
        producer, _res_idx = left.split(" ")
        consumer, op_idx = right.split(" ")

        if consumer in operations:
            operations[consumer].producers.append((producer, int(op_idx)))

    # Stabilize ordering by operand index then producer tag.
    for op in operations.values():
        op.producers = sorted(op.producers, key=lambda p: (p[1], p[0]))

    # Bind each operation tag to its exact linalg op text in lexical order.
    op_texts = _collect_linalg_ops_from_full_code(full_code)
    for pos, tag in enumerate(operation_tags):
        idx = pos
        m = re.match(r"^operation_(\d+)$", tag)
        if m:
            idx = int(m.group(1))
        if 0 <= idx < len(op_texts):
            operations[tag].op_text = op_texts[idx]

    return ParsedAst(
        full_code=full_code,
        operations=operations,
        operation_tags=operation_tags,
        map_defs=map_defs,
        constants=constants,
    )


def _enumerate_backward_paths(
    operations: dict[str, ParsedOperation],
    start_tag: str,
    max_depth: int,
    max_paths: int,
) -> list[list[str]]:
    """Enumerate simple consumer->producer paths from start_tag."""
    paths: list[list[str]] = []
    stack: list[list[str]] = [[start_tag]]

    while stack and len(paths) < max_paths:
        path = stack.pop()
        current = path[-1]

        if len(path) >= 2:
            paths.append(path)

        if len(path) >= max_depth:
            continue

        for producer_tag, _ in operations[current].producers:
            if producer_tag not in operations:
                continue
            if producer_tag in path:
                continue
            stack.append(path + [producer_tag])

    return paths


def _window_paths(path: list[str], window_size: int, stride: int) -> list[list[str]]:
    if len(path) < window_size:
        return []

    windows = []
    last_start = len(path) - window_size

    start = 0
    while start <= last_start:
        windows.append(path[start:start + window_size])
        start += stride

    if windows and windows[-1][0] != path[last_start]:
        windows.append(path[last_start:last_start + window_size])

    return windows


def _replace_result_name(op_text: str, new_result_ssa: str) -> str:
    return re.sub(r"^(\s*)%\w+(\s*=\s*)", rf"\1{new_result_ssa}\2", op_text, count=1)


def _patch_block_op(
    op_text: str,
    ins_arg_names: list[str],
    ins_types: list[str],
    outs_arg_names: list[str],
    outs_types: list[str],
    result_type: str,
    result_ssa: str,
) -> str:
    text = _replace_result_name(op_text, result_ssa)
    text = _replace_section_inplace(text, "ins", ins_arg_names, ins_types)
    text = _replace_section_inplace(text, "outs", outs_arg_names, outs_types)

    if "\n" in text:
        lines = text.split("\n")
        lines[-1] = re.sub(r"->\s*tensor<[^>]+>", f"-> {result_type}", lines[-1])
        text = "\n".join(lines)
    else:
        text = re.sub(r"->\s*tensor<[^>]+>\s*$", f"-> {result_type}", text)

    return text


def _build_block_module(
    affine_maps: list[str],
    constants: list[str],
    function_args: list[tuple[str, str]],
    op_texts: list[str],
    return_ssa: str,
    return_type: str,
) -> str:
    header = "\n".join(affine_maps) + "\n" if affine_maps else ""
    args_decl = ", ".join([f"{a}: {t}" for a, t in function_args])

    const_block = ""
    if constants:
        const_block = "\n" + "\n".join([f"    {c.strip()}" for c in constants])

    body_ops = []
    for op in op_texts:
        dedented = textwrap.dedent(op)
        body_ops.append("\n".join([("    " + line) if line.strip() else "" for line in dedented.split("\n")]))
    body_ops_text = "\n".join(body_ops)

    return (
        f"{header}"
        f"module {{\n"
        f"  func.func private @nanoTime() -> i64 attributes {{llvm.emit_c_interface}}\n"
        f"  func.func @main({args_decl}) -> ({return_type}, i64) attributes {{llvm.emit_c_interface}} {{\n"
        f"    %t0 = call @nanoTime() : () -> i64"
        f"{const_block}\n"
        f"{body_ops_text}\n"
        f"    %t1 = call @nanoTime() : () -> i64\n"
        f"    %delta = arith.subi %t1, %t0 : i64\n"
        f"    return {return_ssa}, %delta : {return_type}, i64\n"
        f"  }}\n"
        f"}}\n"
    )


def _build_block_benchmark(
    parsed: ParsedAst,
    block_tags_consumer_to_producer: list[str],
    selected_batch: int,
    model_name: str,
    block_index: int,
    skip_pure_elementwise: bool = False,
) -> tuple[str, dict]:
    """Build one block benchmark MLIR and return (mlir_text, metadata)."""
    # Producer->consumer execution order.
    ordered_tags = list(reversed(block_tags_consumer_to_producer))

    op_texts_patched: list[str] = []
    function_args: list[tuple[str, str]] = []
    dedup_arg_names: set[str] = set()

    all_affine_maps: list[str] = []
    all_constants: list[str] = []

    previous_tag: Optional[str] = None
    previous_result_ssa: Optional[str] = None
    previous_result_type: Optional[str] = None

    block_op_counts: list[dict[str, int]] = []
    had_dynamic = False

    for op_idx, tag in enumerate(ordered_tags):
        op_meta = parsed.operations[tag]
        block_op_counts.append(op_meta.op_count)

        op_text = op_meta.op_text
        if not op_text:
            raise BlockBuildSkip("missing_op_text")

        ins_args, ins_types = _parse_section(op_text, "ins")
        outs_args, outs_types = _parse_section(op_text, "outs")
        result_type_raw = _extract_result_type(op_text)
        if not result_type_raw and outs_types:
            result_type_raw = outs_types[-1]
        if not result_type_raw:
            raise BlockBuildSkip("missing_result_type")

        raw_types = ins_types + outs_types + [result_type_raw]
        if _has_dynamic(raw_types):
            had_dynamic = True

        spec_ins = [_specialize(t, selected_batch) for t in ins_types]
        spec_outs = [_specialize(t, selected_batch) for t in outs_types]
        spec_result = _specialize(result_type_raw, selected_batch)

        if _has_dynamic(spec_ins + spec_outs + [spec_result]):
            # Unresolved dynamic symbols remain.
            raise BlockBuildSkip("unresolved_dynamic_shape")

        # Build ins arg names for this op.
        new_ins_names: list[str] = []
        for i, _ in enumerate(ins_args):
            connect_prev = False
            if previous_tag is not None:
                # current operation is consumer of previous_tag at producer operand index.
                for prod_tag, operand_idx in op_meta.producers:
                    if prod_tag == previous_tag and operand_idx == i:
                        connect_prev = True
                        break

            if connect_prev:
                if previous_result_ssa is None or previous_result_type is None:
                    raise BlockBuildSkip("missing_previous_result")
                if previous_result_type != spec_ins[i]:
                    # Type mismatch on path edge.
                    raise BlockBuildSkip("path_edge_type_mismatch")
                new_ins_names.append(previous_result_ssa)
            else:
                arg_type = spec_ins[i]
                # Scalar types (f32, i64, etc.) are hoisted as arith.constants
                # inside @main instead of being @main arguments.
                if (arg_type.startswith('tensor') or arg_type.startswith('memref')
                        or arg_type == 'index'):
                    arg_name = f"%arg_{op_idx}_in_{i}"
                    new_ins_names.append(arg_name)
                    if arg_name not in dedup_arg_names:
                        function_args.append((arg_name, arg_type))
                        dedup_arg_names.add(arg_name)
                else:
                    const_name = f"%const_{op_idx}_in_{i}"
                    new_ins_names.append(const_name)
                    if arg_type in ('f32', 'f64', 'f16', 'bf16'):
                        const_line = f"{const_name} = arith.constant 2.00000e+00 : {arg_type}"
                    elif arg_type in ('i64', 'i32', 'i16', 'i8', 'i1'):
                        const_line = f"{const_name} = arith.constant 2 : {arg_type}"
                    else:
                        const_line = f"{const_name} = arith.constant 2.00000e+00 : f32"
                    if const_line not in all_constants:
                        all_constants.append(const_line)

        new_out_names: list[str] = []
        for i, _ in enumerate(outs_args):
            out_name = f"%arg_{op_idx}_out_{i}"
            new_out_names.append(out_name)
            if out_name not in dedup_arg_names:
                function_args.append((out_name, spec_outs[i]))
                dedup_arg_names.add(out_name)

        result_ssa = f"%v{op_idx}"
        patched = _patch_block_op(
            op_text,
            ins_arg_names=new_ins_names,
            ins_types=spec_ins,
            outs_arg_names=new_out_names,
            outs_types=spec_outs,
            result_type=spec_result,
            result_ssa=result_ssa,
        )

        ref_maps = _get_referenced_maps(patched, parsed.map_defs)
        for m in ref_maps:
            if m not in all_affine_maps:
                all_affine_maps.append(m)

        needed_consts = _get_needed_constants(patched, parsed.constants)
        if _has_elided_tensor(needed_consts):
            raise BlockBuildSkip("elided_tensor_constant")
        for c in needed_consts:
            if c not in all_constants:
                all_constants.append(c)

        op_texts_patched.append(patched)

        previous_tag = tag
        previous_result_ssa = result_ssa
        previous_result_type = spec_result

    if previous_result_ssa is None or previous_result_type is None:
        raise BlockBuildSkip("missing_final_result")

    if skip_pure_elementwise and not _block_has_heavy_op(op_texts_patched):
        raise BlockBuildSkip("pure_elementwise")

    signature_blob = "|".join([
        model_name,
        ",".join(ordered_tags),
        str(selected_batch),
        "\n".join([_normalized_op_signature(op) for op in op_texts_patched]),
    ])

    block_complexity = block_complexity_from_op_counts(block_op_counts)

    metadata = {
        "model": model_name,
        "block_index": block_index,
        "window_size": len(block_tags_consumer_to_producer),
        "path_consumer_to_producer": block_tags_consumer_to_producer,
        "path_producer_to_consumer": ordered_tags,
        "selected_batch": selected_batch,
        "had_dynamic": had_dynamic,
        "complexity": block_complexity,
        "signature": signature_blob,
    }

    mlir = _build_block_module(
        affine_maps=all_affine_maps,
        constants=all_constants,
        function_args=function_args,
        op_texts=op_texts_patched,
        return_ssa=previous_result_ssa,
        return_type=previous_result_type,
    )
    return mlir, metadata


def _collect_candidate_windows(
    parsed: ParsedAst,
    window_size: int,
    stride: int,
    max_depth: int,
    max_paths: int,
) -> list[list[str]]:
    """Collect unique consumer->producer windows."""
    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    for start_tag in parsed.operation_tags:
        if start_tag not in parsed.operations:
            continue

        paths = _enumerate_backward_paths(parsed.operations, start_tag, max_depth, max_paths)
        for path in paths:
            for window in _window_paths(path, window_size, stride):
                key = tuple(window)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(window)

    return unique


def extract_blocks_from_file(
    input_path: str,
    output_dir: str,
    model_name: Optional[str],
    window_size: int,
    stride: int,
    max_depth: int,
    max_paths: int,
    batch_candidates: list[int],
    batch_fallback: Optional[str],
    manifest_path: Optional[str],
    skip_pure_elementwise: bool = False,
) -> tuple[int, int]:
    """Extract blocks from one MLIR file. Returns (written, skipped)."""
    os.makedirs(output_dir, exist_ok=True)

    if model_name is None:
        model_name = os.path.basename(input_path).replace("_linalg.mlir", "").replace(".mlir", "")

    raw_ast = _run_ast_dumper(input_path)
    parsed = _parse_ast(raw_ast)

    windows = _collect_candidate_windows(
        parsed,
        window_size=window_size,
        stride=stride,
        max_depth=max_depth,
        max_paths=max_paths,
    )

    if windows:
        complexities = []
        for window in windows:
            ordered = list(reversed(window))
            op_counts = [parsed.operations[t].op_count for t in ordered if t in parsed.operations]
            complexities.append(block_complexity_from_op_counts(op_counts))

        batch_selection = select_batches(
            complexities=complexities,
            batch_candidates=batch_candidates,
            fallback=batch_fallback,
        )
        batch_selection_method = batch_selection.method
        batch_selection_details = batch_selection.details
        selected_batches = batch_selection.selected_batches
    else:
        batch_selection_method = "heuristic"
        batch_selection_details = {"note": "no_candidate_windows"}
        selected_batches = []

    written = 0
    skipped = 0
    skip_reason_counts: dict[str, int] = {}
    skip_reason_examples: dict[str, dict] = {}
    entries = []

    for i, window in enumerate(windows):
        selected_batch = selected_batches[i]
        try:
            mlir, meta = _build_block_benchmark(
                parsed=parsed,
                block_tags_consumer_to_producer=window,
                selected_batch=selected_batch,
                model_name=model_name,
                block_index=i,
                skip_pure_elementwise=skip_pure_elementwise,
            )
        except BlockBuildSkip as exc:
            skipped += 1
            skip_reason_counts[exc.reason] = skip_reason_counts.get(exc.reason, 0) + 1
            if exc.reason not in skip_reason_examples:
                skip_reason_examples[exc.reason] = {
                    "block_index": i,
                    "path_consumer_to_producer": window,
                    "selected_batch": selected_batch,
                }
            continue

        filename = f"{model_name}_block_{i}.mlir"
        out_path = os.path.join(output_dir, filename)

        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(mlir)

        written += 1
        meta["output_file"] = os.path.abspath(out_path)
        meta["batch_policy_method"] = batch_selection_method
        entries.append(meta)

    if windows and written == 0:
        raise RuntimeError(
            f"No blocks were generated for model '{model_name}'. "
            f"Skip reasons: {skip_reason_counts}"
        )

    if manifest_path:
        parent = os.path.dirname(manifest_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        manifest = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "input_file": os.path.abspath(input_path),
            "model": model_name,
            "window_size": window_size,
            "stride": stride,
            "traversal": "consumer_to_producer_graph_paths",
            "timing_fill": {
                "status": "pending_supervised_fill",
                "execution_times_file": None,
            },
            "batch_policy": {
                "method": batch_selection_method,
                "details": batch_selection_details,
                "batch_candidates": sorted({int(v) for v in batch_candidates if int(v) > 0}),
                "batch_fallback": batch_fallback,
            },
            "counts": {
                "candidate_windows": len(windows),
                "written": written,
                "skipped": skipped,
                "skip_reasons": skip_reason_counts,
            },
            "skip_reason_examples": skip_reason_examples,
            "entries": entries,
        }

        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract graph-path blocks from full-model MLIR")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Single input .mlir file")
    group.add_argument("--input-dir", help="Directory with *_linalg.mlir files")

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--max-paths", type=int, default=5000)

    parser.add_argument("--batch-policy", choices=["heuristic"], default="heuristic")
    parser.add_argument("--batch-candidates", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--batch-fallback", choices=["mean"], default=None)
    parser.add_argument(
        "--skip-pure-elementwise", action="store_true", default=False,
        help="Skip blocks where every op is an element-wise linalg.generic "
             "(no matmul, conv2d, pooling, or reduction generic)."
    )

    parser.add_argument("--manifest-dir", default=None)

    args = parser.parse_args()

    if args.batch_policy != "heuristic":
        raise SystemExit("Only heuristic batch policy is currently supported")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input:
        model_name = args.model_name
        manifest = None
        if args.manifest_dir:
            model_key = model_name or os.path.basename(args.input).replace("_linalg.mlir", "").replace(".mlir", "")
            manifest = os.path.join(args.manifest_dir, f"{model_key}_blocks_manifest.json")

        written, skipped = extract_blocks_from_file(
            input_path=args.input,
            output_dir=args.output_dir,
            model_name=model_name,
            window_size=args.window_size,
            stride=args.stride,
            max_depth=args.max_depth,
            max_paths=args.max_paths,
            batch_candidates=args.batch_candidates,
            batch_fallback=args.batch_fallback,
            manifest_path=manifest,
            skip_pure_elementwise=args.skip_pure_elementwise,
        )
        print(f"written={written} skipped={skipped}")
        return

    input_files = sorted(
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith("_linalg.mlir") or f.endswith(".mlir")
    )
    if not input_files:
        raise SystemExit(f"No .mlir files found in {args.input_dir}")

    total_written = 0
    total_skipped = 0
    for path in input_files:
        model = os.path.basename(path).replace("_linalg.mlir", "").replace(".mlir", "")
        model_output_dir = os.path.join(args.output_dir, model)
        os.makedirs(model_output_dir, exist_ok=True)

        manifest = None
        if args.manifest_dir:
            manifest = os.path.join(args.manifest_dir, f"{model}_blocks_manifest.json")

        written, skipped = extract_blocks_from_file(
            input_path=path,
            output_dir=model_output_dir,
            model_name=model,
            window_size=args.window_size,
            stride=args.stride,
            max_depth=args.max_depth,
            max_paths=args.max_paths,
            batch_candidates=args.batch_candidates,
            batch_fallback=args.batch_fallback,
            manifest_path=manifest,
            skip_pure_elementwise=args.skip_pure_elementwise,
        )

        total_written += written
        total_skipped += skipped
        print(f"[{model}] written={written} skipped={skipped}")

    print(f"total_written={total_written} total_skipped={total_skipped}")


if __name__ == "__main__":
    main()
