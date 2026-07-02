"""
optimize_full_model.py
----------------------
End-to-end full-model RL optimization pipeline.

For each full neural network model in data/nn/raw_bench/*_linalg.mlir:
  1. Preprocess: inject {tag} attributes via AST dumper
  2. Add @nanoTime() timing wrapper
  3. Measure MLIR baseline execution time (no transforms)
  4. For each operation: run trained RL agent (greedy) to get optimal schedule
  5. Apply all schedules to the full model in-place
  6. Wrap optimized model with timing
  7. Measure MLIR RL (optimized) execution time
  8. Record results

Usage:
  source ~/envs/mlir/bin/activate && set -a && source .env && set +a
  export AUTOSCHEDULER_IMPL=rl_autoschedular_v4_5 CONFIG_FILE_PATH=config/v4_5.json
  python scripts/full_model/optimize_full_model.py \
      --checkpoint results/experiment3/v4_5_agent/run_0/models/model_715.pt \
      --output results/full_model_results.json
"""

import os
import sys
import json
import re
import argparse
import subprocess
import traceback
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Environment must be loaded before any utils.config imports
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()
load_dotenv(".env.debug", override=False)

if not os.getenv("CONFIG_FILE_PATH"):
    print("Error: CONFIG_FILE_PATH not set. Source .env and export CONFIG_FILE_PATH.")
    sys.exit(1)

from utils.config import Config
from utils.implementation import get_autoschedular_impl, import_autoschedular_module

# ---------------------------------------------------------------------------
# Dynamic imports based on implementation
# ---------------------------------------------------------------------------
IMPL = get_autoschedular_impl()

HiearchyModel = import_autoschedular_module("model", IMPL).HiearchyModel
ExecutionMod = import_autoschedular_module("execution", IMPL)
Execution = ExecutionMod.Execution
ActionSpace = import_autoschedular_module("actions", IMPL).ActionSpace
Observation = import_autoschedular_module("observation", IMPL).Observation
extract_bench_features_from_code = import_autoschedular_module("state", IMPL).extract_bench_features_from_code
OperationState = import_autoschedular_module("state", IMPL).OperationState
BenchmarkFeatures = import_autoschedular_module("state", IMPL).BenchmarkFeatures
OperationFeatures = import_autoschedular_module("state", IMPL).OperationFeatures
NestedLoopFeatures = import_autoschedular_module("state", IMPL).NestedLoopFeatures
IteratorType = import_autoschedular_module("state", IMPL).IteratorType
OperationType = import_autoschedular_module("state", IMPL).OperationType
# TiledFusion for producer update
TiledFusion = import_autoschedular_module("actions.tiled_fusion", IMPL).TiledFusion

_config = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# AST dumper utilities (inline to avoid import issues from scripts/)
# ---------------------------------------------------------------------------

def run_ast_dumper(file_path: str) -> str:
    ast_dumper = os.getenv("AST_DUMPER_BIN_PATH")
    if not ast_dumper:
        raise RuntimeError("AST_DUMPER_BIN_PATH is not set.")
    proc = subprocess.run(
        f"{ast_dumper} {file_path}",
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
    )
    out = proc.stdout.decode("utf-8")
    err = proc.stderr.decode("utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"AST dumper failed: {err}")
    return out


def extract_tagged_code(raw_ast_info: str) -> str:
    if "########################################" in raw_ast_info:
        _, tagged = raw_ast_info.split("########################################", 1)
        return tagged.strip()
    raise RuntimeError("AST dumper output missing '########################################' separator.")


# ---------------------------------------------------------------------------
# Timing wrapper (inline)
# ---------------------------------------------------------------------------

def add_timing_wrapper(code: str) -> str:
    """Wrap @main body with @nanoTime() calls, change return to (tensor, i64).

    Places %t0 at the start of @main body and %t1 right before the return,
    so the entire function body is timed.
    """
    # The AST dumper may produce {-# dialect_resources #-} blocks.
    # We strip these, process the MLIR, and re-insert them inside the
    # module before the closing }.
    resource_block = ""
    resource_match = re.search(r'\n\{-.*?#-\}', code, re.DOTALL)
    if resource_match:
        resource_block = resource_match.group(0)
        code = code[:resource_match.start()] + code[resource_match.end():]

    # Match @main with optional parenthesized multi-value return: -> type or -> (type1, type2, ...)
    main_match = re.search(
        r'(func\.func\s+@main\s*\(([^)]*)\)\s*->\s*((?:\([^)]+\)|\S+))\s*\{)\n?', code
    )
    if not main_match:
        raise ValueError("Could not find func.func @main(…) -> … {")

    args_str = main_match.group(2).strip()
    return_type_str = main_match.group(3).strip()
    # Strip outer parens if multi-value: "(type1, type2)" -> "type1, type2"
    if return_type_str.startswith("(") and return_type_str.endswith(")"):
        return_type_str = return_type_str[1:-1].strip()

    # Find matching closing brace for @main
    body_start = main_match.end()
    depth = 1
    pos = body_start
    while depth > 0 and pos < len(code):
        if code[pos] == '{':
            depth += 1
        elif code[pos] == '}':
            depth -= 1
        pos += 1
    body_end = pos - 1

    body = code[body_start:body_end]

    # Find the return statement (handles single and multi-value returns)
    lines = body.split('\n')
    return_idx = None
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith('return ') and ':' in stripped:
            return_idx = i
            break
    if return_idx is None:
        raise ValueError("Could not find 'return %val : type' in @main body")

    return_line = lines[return_idx].strip()
    ret_content = return_line[len('return '):].strip()
    # Split on last ':' to separate values from types
    ret_vals, ret_types = ret_content.rsplit(':', 1)
    ret_vals = ret_vals.strip()
    ret_types = ret_types.strip()
    # Normalize: remove outer parens from multi-value returns if present
    if ret_types.startswith("(") and ret_types.endswith(")"):
        ret_types = ret_types[1:-1].strip()

    # Build timed body: @nanoTime at start, body, @nanoTime before return
    body_before_return = '\n'.join(lines[:return_idx])
    indent = "    "

    timed_body = (
        f'{indent}%t0 = func.call @nanoTime() : () -> i64\n'
        f'{body_before_return}\n'
        f'{indent}%t1 = func.call @nanoTime() : () -> i64\n'
        f'{indent}%delta = arith.subi %t1, %t0 : i64\n'
        f'{indent}return {ret_vals}, %delta : {ret_types}, i64'
    )

    new_main = (
        f'func.func @main({args_str}) -> ({return_type_str}, i64) '
        f'attributes {{llvm.emit_c_interface}} {{\n'
        f'{timed_body}\n'
        f'}}'
    )

    pre = code[:main_match.start()]
    post = code[pos:]

    # Add @nanoTime declaration
    nano_decl = '  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}\n'
    if '@nanoTime' not in pre:
        # Insert after "module {" or "module attributes {...} {"
        if '{' in pre:
            first_brace = pre.index('{')
            pre = pre[:first_brace + 1] + '\n' + nano_decl + pre[first_brace + 1:]
        else:
            pre = nano_decl + pre

    # Re-insert resource block OUTSIDE the module (as in original format)
    if resource_block:
        result = pre + new_main + post + resource_block + '\n'
    else:
        result = pre + new_main + post

    return result


# ---------------------------------------------------------------------------
# Greedy inference: get optimal action sequence for one operation
# ---------------------------------------------------------------------------

def infer_greedy_schedule(
    model: HiearchyModel,
    bench_features: BenchmarkFeatures,
    operation_tag: str,
) -> list:
    """Run greedy RL inference for a single operation.

    Returns list of Action objects representing the optimal schedule.
    """
    operation_features = bench_features.operations[operation_tag].copy()

    for action in operation_features.pre_actions:
        operation_features = action.update_features(operation_features)

    producer_tag = None
    producer_operand_idx = None
    producer_features = None
    if operation_features.producers:
        producer_tag = operation_features.producers[-1][0]
        producer_operand_idx = min(
            idx for t, idx in operation_features.producers if t == producer_tag
        )
        producer_features = bench_features.operations[producer_tag].copy()

    try:
        bench_idx = list(bench_features.operations.keys()).index(operation_tag)
    except ValueError:
        bench_idx = 0

    state = OperationState(
        bench_idx=bench_idx,
        bench_name=bench_features.bench_name,
        operation_tag=operation_tag,
        original_operation_features=bench_features.operations[operation_tag].copy(),
        operation_features=operation_features,
        producer_tag=producer_tag,
        producer_operand_idx=producer_operand_idx,
        producer_features=producer_features,
        transformation_history=[[]],
        terminal=False,
    )

    while not state.terminal:
        obs = Observation.from_state(state).to(DEVICE)

        with torch.no_grad():
            actions_index, _, _ = model.sample(obs, greedy=True)

        try:
            action = ActionSpace.action_by_index(actions_index[0], state)
        except Exception as e:
            print(f"      ERROR creating action: {e}")
            break

        try:
            state.record_action(action)
        except Exception:
            pass

        if isinstance(action, TiledFusion):
            try:
                action.update_producer_features(state, bench_features)
            except Exception:
                pass

        state.operation_features = action.update_features(state.operation_features)
        state.terminal = action.terminal or state.step_count >= _config.truncate

    return state.transformation_history[0]


# ---------------------------------------------------------------------------
# Apply schedules to code
# ---------------------------------------------------------------------------

def apply_schedules_to_code(code: str, schedules: dict[str, list]) -> str:
    """Apply all schedules via a single batched in-process MLIR transform.

    Builds ONE combined transform dialect script for all actions, runs it
    in-process (no subprocess). Falls back to per-op batching if the combined
    script fails (e.g. fragile vectorization for some ops).
    """
    # Step 1: Handle vectorization preprocessing (modifies code string)
    for op_tag, actions in schedules.items():
        for action in actions:
            if type(action).__name__ == 'Vectorization':
                for pre_fn in getattr(action, 'preprocessing', []):
                    try:
                        code = pre_fn(code)
                    except Exception:
                        pass

    # Step 2: Build combined transform script
    body = _build_combined_transform(schedules)
    if not body:
        return code

    from mlir.ir import Context, Module
    from mlir.dialects.transform import interpreter

    full_transform = (
        '\nmodule attributes {transform.with_named_sequence} {\n'
        '  transform.named_sequence @__transform_main'
        '(%arg1: !transform.any_op {transform.readonly}) {\n'
        + '\n'.join('    ' + l for l in body) + '\n'
        '    transform.yield\n'
        '  }\n'
        '}\n'
    )

    try:
        with Context():
            m = Module.parse(code)
            t = Module.parse(full_transform)
            interpreter.apply_named_sequence(
                m, t.body.operations[0], t
            )
            return str(m)
    except Exception as batch_err:
        print(f"    Batched transform failed ({batch_err}), falling back to per-op ...")
        return _apply_schedules_per_op(code, schedules)


def _build_combined_transform(schedules: dict[str, list]) -> list[str]:
    """Build a single combined transform dialect body from all schedules.
    Returns list of transform dialect operation lines."""
    body = []
    step_counter = [0]  # mutable counter for unique SSA names

    def _next_id() -> str:
        sid = f"s{step_counter[0]}"
        step_counter[0] += 1
        return sid

    for op_tag, actions in schedules.items():
        for action in actions:
            cls_name = type(action).__name__

            if cls_name == 'NoTransformation':
                continue

            elif cls_name == 'Tiling':
                sizes = action.parameters
                nz = [s for s in sizes if s != 0]
                if not nz:
                    continue
                sid = _next_id()
                n = len(nz)
                r = ', '.join(['!transform.any_op'] * n)
                body.append(
                    f'%op{sid} = transform.structured.match '
                    f'attributes{{tag = "{op_tag}"}} in %arg1 : '
                    f'(!transform.any_op) -> !transform.any_op'
                )
                body.append(
                    f'%t{sid}, %l{sid}:{n} = '
                    f'transform.structured.tile_using_for %op{sid} '
                    f'tile_sizes {list(sizes)} : (!transform.any_op) -> '
                    f'(!transform.any_op, {r})'
                )

            elif cls_name == 'Interchange':
                perm = action.parameters
                if perm == list(range(len(perm))):
                    continue
                sid = _next_id()
                body.append(
                    f'%op{sid} = transform.structured.match '
                    f'attributes{{tag = "{op_tag}"}} in %arg1 : '
                    f'(!transform.any_op) -> !transform.any_op'
                )
                body.append(
                    f'%g{sid} = transform.structured.generalize '
                    f'%op{sid} : (!transform.any_op) -> !transform.any_op'
                )
                body.append(
                    f'%i{sid} = transform.structured.interchange '
                    f'%g{sid} iterator_interchange = {list(perm)} : '
                    f'(!transform.any_op) -> !transform.any_op'
                )
                body.append(
                    f'%n{sid} = transform.param.constant '
                    f'"{op_tag}" -> !transform.any_param'
                )
                body.append(
                    f'transform.annotate %i{sid} "tag" = %n{sid} : '
                    f'!transform.any_op, !transform.any_param'
                )

            elif cls_name == 'Vectorization':
                sid = _next_id()
                body.append(
                    f'%op{sid} = transform.structured.match '
                    f'attributes{{tag = "{op_tag}"}} in %arg1 : '
                    f'(!transform.any_op) -> !transform.any_op'
                )
                body.append(
                    f'transform.structured.vectorize %op{sid} : '
                    f'(!transform.any_op) -> !transform.any_op'
                )

            elif cls_name == 'TiledParallelization':
                sid = _next_id()
                sizes = action.parameters
                body.append(
                    f'%op{sid} = transform.structured.match '
                    f'attributes{{tag = "{op_tag}"}} in %arg1 : '
                    f'(!transform.any_op) -> !transform.any_op'
                )
                body.append(
                    f'%tp{sid}, %f{sid} = '
                    f'transform.structured.tile_using_forall %op{sid} '
                    f'tile_sizes {list(sizes)} : (!transform.any_op) -> '
                    f'(!transform.any_op, !transform.any_op)'
                )

            elif cls_name == 'TiledFusion':
                consumer = op_tag
                producer = getattr(action, 'producer_tag', None)
                if not producer:
                    continue
                new_producer = getattr(action, 'new_producer_tag',
                                        f'{producer}_{consumer}')
                sid = _next_id()
                body.append(
                    f'%op_c{sid} = transform.structured.match '
                    f'attributes{{tag = "{consumer}"}} in %arg1 : '
                    f'(!transform.any_op) -> !transform.any_op'
                )
                body.append(
                    f'%tc{sid}, %fc{sid} = '
                    f'transform.structured.tile_using_forall %op_c{sid} '
                    f'tile_sizes {list(action.parameters)} : '
                    f'(!transform.any_op) -> '
                    f'(!transform.any_op, !transform.any_op)'
                )
                body.append(
                    f'%op_p{sid} = transform.structured.match '
                    f'attributes{{tag = "{producer}"}} in %arg1 : '
                    f'(!transform.any_op) -> !transform.any_op'
                )
                body.append(
                    f'%fu{sid}, %co{sid} = '
                    f'transform.structured.fuse_into_containing_op '
                    f'%op_p{sid} into %fc{sid} : '
                    f'(!transform.any_op, !transform.any_op) -> '
                    f'(!transform.any_op, !transform.any_op)'
                )
                body.append(
                    f'%ft{sid} = transform.param.constant '
                    f'"{new_producer}" -> !transform.any_param'
                )
                body.append(
                    f'transform.annotate %fu{sid} "tag" = %ft{sid} : '
                    f'!transform.any_op, !transform.any_param'
                )

    return body


def _apply_schedules_per_op(code: str, schedules: dict[str, list]) -> str:
    """Fallback: apply schedules per-op (actions batched per operation)."""
    from mlir.ir import Context, Module
    from mlir.dialects.transform import interpreter

    for op_tag, actions in schedules.items():
        body = _build_op_body(op_tag, actions)
        if not body:
            continue

        transform_str = (
            '\nmodule attributes {transform.with_named_sequence} {\n'
            '  transform.named_sequence @__transform_main'
            '(%arg1: !transform.any_op {transform.readonly}) {\n'
            + '\n'.join('    ' + l for l in body) + '\n'
            '    transform.yield\n'
            '  }\n'
            '}\n'
        )

        try:
            with Context():
                m = Module.parse(code)
                t = Module.parse(transform_str)
                interpreter.apply_named_sequence(
                    m, t.body.operations[0], t
                )
                code = str(m)
        except Exception as e:
            print(f"      WARNING: actions for {op_tag} failed: {e}")
    return code


def _build_op_body(op_tag: str, actions: list) -> list[str]:
    """Build transform body for a single operation's action sequence."""
    body = []
    step_counter = [0]

    def _next_id() -> str:
        sid = f"{op_tag}_{step_counter[0]}"
        step_counter[0] += 1
        return sid

    for action in actions:
        cls_name = type(action).__name__

        if cls_name == 'NoTransformation':
            continue

        elif cls_name == 'Tiling':
            sizes = action.parameters
            nz = [s for s in sizes if s != 0]
            if not nz:
                continue
            sid = _next_id()
            n = len(nz)
            r = ', '.join(['!transform.any_op'] * n)
            body.append(
                f'%op{sid} = transform.structured.match '
                f'attributes{{tag = "{op_tag}"}} in %arg1 : '
                f'(!transform.any_op) -> !transform.any_op'
            )
            body.append(
                f'%t{sid}, %l{sid}:{n} = '
                f'transform.structured.tile_using_for %op{sid} '
                f'tile_sizes {list(sizes)} : (!transform.any_op) -> '
                f'(!transform.any_op, {r})'
            )

        elif cls_name == 'Interchange':
            perm = action.parameters
            if perm == list(range(len(perm))):
                continue
            sid = _next_id()
            body.append(
                f'%op{sid} = transform.structured.match '
                f'attributes{{tag = "{op_tag}"}} in %arg1 : '
                f'(!transform.any_op) -> !transform.any_op'
            )
            body.append(
                f'%g{sid} = transform.structured.generalize '
                f'%op{sid} : (!transform.any_op) -> !transform.any_op'
            )
            body.append(
                f'%i{sid} = transform.structured.interchange '
                f'%g{sid} iterator_interchange = {list(perm)} : '
                f'(!transform.any_op) -> !transform.any_op'
            )
            body.append(
                f'%n{sid} = transform.param.constant '
                f'"{op_tag}" -> !transform.any_param'
            )
            body.append(
                f'transform.annotate %i{sid} "tag" = %n{sid} : '
                f'!transform.any_op, !transform.any_param'
            )

        elif cls_name == 'Vectorization':
            sid = _next_id()
            body.append(
                f'%op{sid} = transform.structured.match '
                f'attributes{{tag = "{op_tag}"}} in %arg1 : '
                f'(!transform.any_op) -> !transform.any_op'
            )
            body.append(
                f'transform.structured.vectorize %op{sid} : '
                f'(!transform.any_op) -> !transform.any_op'
            )

        elif cls_name == 'TiledParallelization':
            sid = _next_id()
            sizes = action.parameters
            body.append(
                f'%op{sid} = transform.structured.match '
                f'attributes{{tag = "{op_tag}"}} in %arg1 : '
                f'(!transform.any_op) -> !transform.any_op'
            )
            body.append(
                f'%tp{sid}, %f{sid} = '
                f'transform.structured.tile_using_forall %op{sid} '
                f'tile_sizes {list(sizes)} : (!transform.any_op) -> '
                f'(!transform.any_op, !transform.any_op)'
            )

        elif cls_name == 'TiledFusion':
            consumer = op_tag
            producer = getattr(action, 'producer_tag', None)
            if not producer:
                continue
            new_producer = getattr(
                action, 'new_producer_tag', f'{producer}_{consumer}'
            )
            sid = _next_id()
            body.append(
                f'%op_c{sid} = transform.structured.match '
                f'attributes{{tag = "{consumer}"}} in %arg1 : '
                f'(!transform.any_op) -> !transform.any_op'
            )
            body.append(
                f'%tc{sid}, %fc{sid} = '
                f'transform.structured.tile_using_forall %op_c{sid} '
                f'tile_sizes {list(action.parameters)} : '
                f'(!transform.any_op) -> '
                f'(!transform.any_op, !transform.any_op)'
            )
            body.append(
                f'%op_p{sid} = transform.structured.match '
                f'attributes{{tag = "{producer}"}} in %arg1 : '
                f'(!transform.any_op) -> !transform.any_op'
            )
            body.append(
                f'%fu{sid}, %co{sid} = '
                f'transform.structured.fuse_into_containing_op '
                f'%op_p{sid} into %fc{sid} : '
                f'(!transform.any_op, !transform.any_op) -> '
                f'(!transform.any_op, !transform.any_op)'
            )
            body.append(
                f'%ft{sid} = transform.param.constant '
                f'"{new_producer}" -> !transform.any_param'
            )
            body.append(
                f'transform.annotate %fu{sid} "tag" = %ft{sid} : '
                f'!transform.any_op, !transform.any_param'
            )

    return body


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def measure_exec_time(exec_instance, code: str, bench_name: str) -> tuple[int, bool]:
    """Measure execution time. Tries v4_5 bindings first, falls back to
    mlir-cpu-runner cmd if bindings time out (common for large models)."""
    try:
        result = exec_instance.execute_code(code, bench_name, [])
        et = result[0]
        ok = result[1]
        if ok and et > 0:
            return et, True
        err_msg = result[3] if len(result) >= 4 else "unknown error"
        if "timed out" in str(err_msg).lower():
            print(f"      v4_5 execution timed out, trying cmd fallback ...")
            return _measure_with_cmd_fallback(code, bench_name)
        print(f"      Execution returned ok=False: {err_msg[:200]}")
        return 0, False
    except Exception as e:
        msg = str(e)
        if "timed out" in msg.lower():
            print(f"      v4_5 execution timed out, trying cmd fallback ...")
            return _measure_with_cmd_fallback(code, bench_name)
        print(f"      Execution exception: {e}")
        return 0, False


def _measure_with_cmd_fallback(code: str, bench_name: str) -> tuple[int, bool]:
    """Execute MLIR code in an isolated multiprocessing subprocess with a
    generous timeout. Runs mlir-opt (one-shot-bufferize) + LLVM lowering +
    JIT all inside the worker (so the full timeout applies to everything).
    Uses EXEC_TIMEOUT_CMD (default 7200s)."""
    import multiprocessing
    import ctypes
    import ctypes.util

    timeout_s = int(os.environ.get("EXEC_TIMEOUT_CMD", 7200))
    llvm_build = os.getenv("LLVM_BUILD_PATH", "")

    def worker(code_str: str, result_dict: dict) -> None:
        try:
            import tempfile, subprocess
            import numpy as np
            from mlir.ir import Context, Module, MemRefType, IntegerType, F64Type, F32Type
            from mlir.execution_engine import ExecutionEngine
            from mlir.runtime import (
                get_ranked_memref_descriptor, make_nd_memref_descriptor,
                as_ctype, ranked_memref_to_numpy,
            )
            from mlir.passmanager import PassManager
            from mlir.dialects.func import FuncOp

            # Step 1: mlir-opt one-shot-bufferize (stop before LLVM lowering)
            buf_pipeline = (
                f"{llvm_build}/bin/mlir-opt "
                f"-loop-invariant-code-motion -canonicalize "
                f"-eliminate-empty-tensors -empty-tensor-to-alloc-tensor "
                f"-one-shot-bufferize='bufferize-function-boundaries "
                f"function-boundary-type-conversion=identity-layout-map' "
            )
            with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tf:
                tf.write(code_str)
                tmp_path = tf.name
            buf_path = tmp_path.replace(".mlir", "_buf.mlir")
            buf_cmd = f"{buf_pipeline} {tmp_path} -o {buf_path}"
            buf_proc = subprocess.run(buf_cmd, shell=True, capture_output=True, text=True, timeout=7200)
            os.remove(tmp_path)
            if buf_proc.returncode != 0:
                result_dict["success"] = False
                result_dict["error"] = f"mlir-opt bufferize failed: {buf_proc.stderr[:300]}"
                return
            with open(buf_path) as f:
                bufferized = f.read()
            os.remove(buf_path)

            # Step 2: LLVM lowering + JIT via PassManager + ExecutionEngine
            pass_pipeline = """builtin.module(
                canonicalize,
                convert-linalg-to-loops,
                scf-forall-to-parallel,
                convert-scf-to-openmp,
                expand-strided-metadata,
                finalize-memref-to-llvm,
                convert-scf-to-cf,
                lower-affine,
                convert-openmp-to-llvm,
                convert-vector-to-llvm,
                convert-math-to-llvm,
                convert-math-to-libm,
                finalize-memref-to-llvm,
                convert-func-to-llvm,
                convert-index-to-llvm,
                convert-arith-to-llvm,
                convert-cf-to-llvm,
                reconcile-unrealized-casts,
                canonicalize,
                cse
            )"""

            with Context():
                module = Module.parse(bufferized)
                pm = PassManager.parse(pass_pipeline)

                main_func = next(
                    op for op in module.body.operations
                    if isinstance(op, FuncOp) and op.name.value == "main"
                )

                def _dtype(memref_type):
                    et = memref_type.element_type
                    if isinstance(et, F32Type):
                        return np.float32
                    if isinstance(et, F64Type):
                        return np.float64
                    if isinstance(et, IntegerType):
                        return {32: np.int32, 64: np.int64}.get(et.width, np.int32)
                    return np.float32

                inputs = []
                for inp_type in main_func.type.inputs:
                    arr = np.zeros(inp_type.shape, dtype=_dtype(inp_type))
                    inputs.append(arr)

                res_types = main_func.type.results
                out_fields = []
                for i, ot in enumerate(res_types[:-1]):
                    desc_t = make_nd_memref_descriptor(ot.rank, as_ctype(_dtype(ot)))
                    out_fields.append((f"out_{i}", desc_t))

                class OutStruct(ctypes.Structure):
                    _fields_ = [*out_fields, ("delta", ctypes.c_int64)]

                outs = OutStruct()
                for i, (fname, ftype) in enumerate(out_fields):
                    setattr(outs, fname, ftype())

                args = [ctypes.pointer(ctypes.pointer(outs))]
                for arr in inputs:
                    args.append(ctypes.pointer(ctypes.pointer(
                        get_ranked_memref_descriptor(arr)
                    )))

                pm.run(module.operation)
                engine = ExecutionEngine(
                    module, opt_level=3,
                    shared_libs=os.getenv("MLIR_SHARED_LIBS", "").split(","),
                )

                libc = ctypes.CDLL(ctypes.util.find_library("c"))
                libc.free.argtypes = [ctypes.c_void_p]
                libc.free.restype = None

                def _free_outputs():
                    for fname, desc_t in out_fields:
                        memref = getattr(outs, fname)
                        al = getattr(memref, "allocated", None)
                        if al and ctypes.cast(al, ctypes.c_void_p).value:
                            libc.free(ctypes.cast(al, ctypes.c_void_p))

                engine.invoke("main", *args)  # warmup
                _free_outputs()
                engine.invoke("main", *args)  # measured run
                _free_outputs()

                result_dict["delta"] = outs.delta
                result_dict["success"] = True
        except Exception as e:
            result_dict["success"] = False
            result_dict["error"] = str(e)

    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    proc = multiprocessing.Process(target=worker, args=(code, result_dict))
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        print(f"      Multiprocess fallback timed out after {timeout_s}s")
        return 0, False

    if result_dict.get("success"):
        delta = result_dict.get("delta", 0)
        print(f"      Multiprocess fallback succeeded: {delta:,} ns")
        return delta, True

    error = result_dict.get("error", "unknown")
    print(f"      Multiprocess fallback failed: {error[:200]}")
    print(f"      Trying mlir-cpu-runner CMD pipeline (one-shot-bufferize) ...")
    return _measure_with_cmd_pipeline(code, bench_name)


def _measure_with_cmd_pipeline(code: str, bench_name: str) -> tuple[int, bool]:
    """Final fallback: mlir-opt (one-shot-bufferize) | mlir-cpu-runner.
    Replicates v4_5's __execute_code_with_cmd pass pipeline exactly."""
    import tempfile
    import subprocess
    import signal

    timeout_s = int(os.environ.get("EXEC_TIMEOUT_CMD", 7200))
    llvm_build = os.getenv("LLVM_BUILD_PATH", "")
    shared_libs = os.getenv("MLIR_SHARED_LIBS", "")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tf:
        tf.write(code)
        tmp_path = tf.name
        tf.flush()  # ensure data is on disk before mlir-opt reads it

    try:
        command_1 = (
            f"{llvm_build}/bin/mlir-opt "
            f"-loop-invariant-code-motion -canonicalize "
            f"-eliminate-empty-tensors -empty-tensor-to-alloc-tensor "
            f"-one-shot-bufferize='bufferize-function-boundaries "
            f"function-boundary-type-conversion=identity-layout-map' "
            f"-convert-vector-to-scf -convert-linalg-to-loops "
            f"-buffer-deallocation-pipeline -scf-forall-to-parallel "
            f"-convert-scf-to-openmp -expand-strided-metadata "
            f"-finalize-memref-to-llvm -convert-scf-to-cf -lower-affine "
            f"-convert-arith-to-llvm -convert-openmp-to-llvm "
            f"-convert-vector-to-llvm -convert-cf-to-llvm "
            f"-convert-func-to-llvm -convert-math-to-llvm "
            f"-convert-math-to-libm -finalize-memref-to-llvm "
            f"-reconcile-unrealized-casts -canonicalize -cse"
        )
        command_2 = (
            f"{llvm_build}/bin/mlir-cpu-runner -e main "
            f"-entry-point-result=void -shared-libs={shared_libs}"
        )
        os.environ["OMP_NUM_THREADS"] = "8"

        opt_path = tmp_path.replace(".mlir", "_opt.mlir")
        opt_cmd = f"{command_1} {tmp_path} -o {opt_path}"
        opt_proc = subprocess.run(opt_cmd, shell=True, capture_output=True, text=True, timeout=timeout_s)
        if opt_proc.returncode != 0:
            print(f"      CMD pipeline mlir-opt failed: {opt_proc.stderr[:300]}")
            return 0, False
        full_cmd = f"{command_2} {opt_path}"
        proc = subprocess.Popen(
            full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, preexec_fn=os.setpgrp,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait()
            print(f"      CMD pipeline timed out after {timeout_s}s")
            return 0, False

        if proc.returncode == 0 and stdout.strip():
            et_ns = int(stdout.strip().split('\n')[-1])
            print(f"      CMD pipeline succeeded: {et_ns:,} ns")
            return et_ns, True

        err_sample = stderr[:300] if stderr else f"rc={proc.returncode}"
        print(f"      CMD pipeline failed: {err_sample}")
        return 0, False
    except Exception as e:
        print(f"      CMD pipeline error: {e}")
        return 0, False
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        opt_path = tmp_path.replace(".mlir", "_opt.mlir")
        if os.path.exists(opt_path):
            os.remove(opt_path)


def _save_results(output_path: str, results: dict) -> None:
    """Save results incrementally so progress is not lost on crash."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


def _save_checkpoint(ckpt_path, baseline_ns, schedules, ops):
    """Save checkpoint so step 5 can resume if it crashes."""
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt_path, "w") as f:
        json.dump({
            "baseline_ns": baseline_ns,
            "ops": ops,
            "schedules": {t: [str(a) for a in acts] for t, acts in schedules.items()},
        }, f)


def _load_checkpoint(ckpt_path):
    """Load checkpoint. Returns (baseline_ns, schedules_action_strs, ops) or None."""
    if not ckpt_path.exists():
        return None
    with open(ckpt_path) as f:
        data = json.load(f)
    return data.get("baseline_ns"), data.get("schedules", {}), data.get("ops", [])


def _extract_features_resilient(
    tagged_code: str, bench_name: str, root_exec_time: int
) -> tuple:
    """Extract features from tagged MLIR code, skipping ops that exceed
    dimension limits. Workaround for state.py KeyError when graph edges
    reference operations dropped by max_num_loops / max_num_stores_loads
    / max_num_load_store_dim limits."""
    raw_ast = run_ast_dumper_on_code(tagged_code)
    info, _ = raw_ast.split("########################################", 1)
    if "#BEGIN_GRAPH" not in info:
        raise RuntimeError("No graph in AST dumper output")

    ops_lines, graph_str = info.split("#BEGIN_GRAPH", 1)
    graph_clean = graph_str.replace("#END_GRAPH", "")
    graph_edges = [
        (line.split(" --> ")[0].split(" "), line.split(" --> ")[1].split(" "))
        for line in graph_clean.strip().split("\n") if " --> " in line
    ]

    # Parse each operation block and check dimension limits
    ops_blocks = [b.strip() for b in ops_lines.split("#START_OPERATION") if b.strip()]
    valid_tags = set()
    ops_data = {}

    cfg = Config()
    for block in ops_blocks:
        rest, tag = block.split("#START_TAG", 1)
        tag = tag.strip().split("\n")[0]

        # Check nested loops
        _opname, rest2 = rest.split("#START_VECTORIZABLE", 1)
        _vec, rest3 = rest2.split("#START_NESTED_LOOPS", 1)
        nested_str, rest4 = rest3.split("#START_LOAD_DATA", 1)
        nloops = len([l for l in nested_str.strip().split("\n") if l.strip()])
        if nloops > cfg.max_num_loops:
            continue

        # Check load/store dimensions
        loads_str, rest5 = rest4.split("#START_STORE_DATA", 1)
        stores_str, _opcount_str = rest5.split("#START_OP_COUNT", 1)
        max_load_dim = max((len(ld.split(", ")) for ld in loads_str.strip().split("\n") if ld.strip()), default=0)
        max_store_dim = max((len(sd.split(", ")) for sd in stores_str.strip().split("\n") if sd.strip()), default=0)
        if max_load_dim > cfg.max_num_load_store_dim or max_store_dim > cfg.max_num_load_store_dim:
            continue

        valid_tags.add(tag)
        ops_data[tag] = {"tag": tag, "nloops": nloops}

    # Filter graph edges to only valid tags
    valid_edges = [
        ((p, ri), (c, oi))
        for (p, ri), (c, oi) in graph_edges
        if p[0] in valid_tags and c[0] in valid_tags
    ]

    print(f"        Resilient: {len(valid_tags)}/{len(ops_blocks)} ops within limits, "
          f"{len(valid_edges)}/{len(graph_edges)} graph edges valid")

    # Build feature dict manually from AST dumper output.
    # Do NOT call extract_bench_features_from_code — it would trigger the same KeyError.
    bench_features = BenchmarkFeatures(
        bench_name=bench_name,
        code=tagged_code,
        operation_tags=sorted(valid_tags),
        operations={},
        root_exec_time=root_exec_time,
    )

    # Parse operations manually from AST blocks, skipping oversized ones
    import re as _re

    def _get_op_type(opname: str):
        for ot in OperationType:
            if ot.value and ot.value in opname:
                return ot
        return OperationType.unknown

    for block in ops_blocks:
        rest, tag = block.split("#START_TAG", 1)
        tag = tag.strip().split("\n")[0]
        if tag not in valid_tags:
            continue

        opname, rest2 = rest.split("#START_VECTORIZABLE", 1)
        opname = opname.strip()
        op_type = _get_op_type(opname)

        vectorizable_str, rest3 = rest2.split("#START_NESTED_LOOPS", 1)
        vectorizable = vectorizable_str.strip() == "true"

        nested_str, rest4 = rest3.split("#START_LOAD_DATA", 1)
        nested_loops = []
        for nl_str in nested_str.strip().split("\n"):
            if not nl_str.strip():
                continue
            parts = nl_str.strip().split(" ")
            if len(parts) >= 5:
                nested_loops.append(NestedLoopFeatures(
                    arg=f"%{parts[0]}", lower_bound=int(parts[1]),
                    upper_bound=int(parts[2]), step=int(parts[3]),
                    iterator_type=IteratorType(parts[4]),
                ))

        loads_str, rest5 = rest4.split("#START_STORE_DATA", 1)
        loads_str = _re.sub(r'd\d+', lambda m: f'%{m.group()}', loads_str)
        load_data = [ld.split(", ") for ld in loads_str.strip().split("\n") if ld.strip()]

        stores_str, opcount_str = rest5.split("#START_OP_COUNT", 1)
        stores_str = _re.sub(r'd\d+', lambda m: f'%{m.group()}', stores_str)
        store_data = [sd.split(", ") for sd in stores_str.strip().split("\n") if sd.strip()]

        op_count = {}
        for oc_str in opcount_str.strip().split("\n"):
            if not oc_str.strip():
                continue
            parts = oc_str.strip().split(" ")
            if len(parts) == 2:
                op_count[parts[0]] = int(parts[1])

        bench_features.operations[tag] = OperationFeatures(
            operation_name=opname, operation_type=op_type,
            op_count=op_count, load_data=load_data, store_data=store_data,
            nested_loops=nested_loops, producers=[], consumers=[],
            vectorizable=vectorizable, pre_actions=[],
        )

    # Build producer/consumer edges for valid ops only
    for (producer, res_idx), (consumer, op_idx) in valid_edges:
        ptag = producer[0]
        ctag = consumer[0]
        if ptag in bench_features.operations and ctag in bench_features.operations:
            bench_features.operations[ctag].producers.append((ptag, int(op_idx)))
            bench_features.operations[ptag].consumers.append((ctag, int(res_idx)))

    ops_list = sorted(valid_tags)
    return bench_features, ops_list


def run_ast_dumper_on_code(code: str) -> str:
    """Run AST dumper on a code string via stdin."""
    proc = subprocess.run(
        f'{os.getenv("AST_DUMPER_BIN_PATH")} -',
        shell=True, input=code.encode("utf-8"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
    )
    out = proc.stdout.decode("utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"AST dumper failed: {proc.stderr.decode('utf-8')}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full-model RL optimization pipeline.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model .pt file.")
    parser.add_argument("--models-dir", default="data/nn/raw_bench", help="Dir with *_linalg.mlir files.")
    parser.add_argument("--output", default="results/full_model_results.json", help="Results JSON path.")
    parser.add_argument("--models", nargs="*", default=None, help="Specific model names (stem, without _linalg).")
    parser.add_argument("--tag-dir", default="data/nn/tagged", help="Dir for intermediate tagged files.")
    parser.add_argument("--skip-tagging", action="store_true", help="Skip tagging if files exist.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available.")
    args = parser.parse_args()

    # --- Load model ---
    print(f"Loading model: {args.checkpoint}")
    model = HiearchyModel().to(DEVICE)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()
    print(f"Model loaded (impl: {IMPL}).")

    # --- Init execution ---
    exec_engine = Execution("")
    print("Execution engine ready.")

    # --- Discover models ---
    models_dir = Path(args.models_dir)
    if args.models:
        model_files = [models_dir / f"{m}_linalg.mlir" for m in args.models]
    else:
        model_files = sorted(models_dir.glob("*_linalg.mlir"))
    print(f"\nFound {len(model_files)} model(s).")

    # --- Process ---
    results = {}
    tag_dir = Path(args.tag_dir)
    tag_dir.mkdir(parents=True, exist_ok=True)

    for model_path in model_files:
        model_name = model_path.stem.replace("_linalg", "")
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        # [1] Tagging
        tagged_path = tag_dir / f"{model_name}_tagged.mlir"
        if not args.skip_tagging or not tagged_path.exists():
            print("  [1/5] Tagging ...")
            try:
                raw = run_ast_dumper(str(model_path))
                tagged_code = extract_tagged_code(raw)
                with open(tagged_path, "w") as f:
                    f.write(tagged_code)
                print(f"        → {tagged_path}")
            except Exception as e:
                print(f"        ERROR: {e}")
                results[model_name] = {"error": f"tagging: {e}"}
                _save_results(args.output, results)
                continue
        else:
            print("  [1/5] Using cached tagged file.")
            with open(tagged_path) as f:
                tagged_code = f.read()

        # [2] Wrap for baseline timing
        print("  [2/5] Wrapping for MLIR baseline ...")
        try:
            wrapped_base = add_timing_wrapper(tagged_code)
        except Exception as e:
            print(f"        ERROR: {e}")
            results[model_name] = {"error": f"wrap: {e}"}
            _save_results(args.output, results)
            continue

        # [3] MLIR baseline
        print("  [3/5] Measuring MLIR baseline ...")
        baseline_ns, ok = measure_exec_time(exec_engine, wrapped_base, f"model-{model_name}")
        if not ok:
            print(f"        FAILED")
            results[model_name] = {"error": "baseline_exec_failed"}
            _save_results(args.output, results)
            continue
        print(f"        {baseline_ns:,} ns ({baseline_ns/1e6:.2f} ms)")

        # [4] Extract features + RL inference per op
        print("  [4/5] RL agent inference per operation ...")
        schedules = {}
        _ckpt_path = tag_dir / f"{model_name}_checkpoint.json"

        if args.resume and _ckpt_path.exists():
            print("        Resuming from checkpoint ...")
            bl_ck, sched_strs, ops_ck = _load_checkpoint(_ckpt_path)
            if sched_strs:
                ops = ops_ck
                # Reconstruct actions from strings
                for tag, action_strs in sched_strs.items():
                    actions = []
                    for s in action_strs:
                        try:
                            symbol = s[0]  # e.g. 'I', 'T', 'V', 'NT'
                            params_str = s[2:-1] if '(' in s else ''
                            params = [int(x) for x in params_str.split(',')] if params_str else []
                            action_type = ActionSpace.action_type_by_symbol(symbol)
                            actions.append(action_type(params, operation_tag=tag))
                        except Exception:
                            pass
                    if actions:
                        schedules[tag] = actions
                print(f"        Loaded {len(schedules)} ops from checkpoint.")
            else:
                print("        Corrupt checkpoint, re-running inference.")

        if not schedules:
            try:
                bench_features = extract_bench_features_from_code(
                    bench_name=f"model-{model_name}",
                    code=tagged_code,
                    root_execution_time=baseline_ns,
                )
                ops = bench_features.operation_tags

                if not ops:
                    print("        No linalg ops found — skipping.")
                    results[model_name] = {
                        "model": model_name, "baseline_ns": baseline_ns,
                        "optimized_ns": baseline_ns, "speedup": 1.0, "num_ops": 0,
                    }
                    _save_results(args.output, results)
                    continue

                print(f"        {len(ops)} operations found.")
                schedules = {}
                for tag in ops:
                    try:
                        actions = infer_greedy_schedule(model, bench_features, tag)
                        schedules[tag] = actions
                        action_strs = [str(a) for a in actions]
                        print(f"        {tag}: {action_strs}")
                    except Exception as e:
                        print(f"        ERROR {tag}: {e}")
                        traceback.print_exc()
                        schedules[tag] = []
            except KeyError as ke:
                print(f"        KeyError in feature extraction ({ke}), retrying with resilient extractor ...")
                try:
                    bench_features, ops = _extract_features_resilient(
                        tagged_code, f"model-{model_name}", baseline_ns
                    )
                    if not ops:
                        results[model_name] = {"model": model_name, "baseline_ns": baseline_ns,
                                                "optimized_ns": baseline_ns, "speedup": 1.0, "num_ops": 0}
                        _save_results(args.output, results)
                        continue
                    print(f"        {len(ops)} operations found (resilient).")
                    schedules = {}
                    for tag in ops:
                        try:
                            actions = infer_greedy_schedule(model, bench_features, tag)
                            schedules[tag] = actions
                            action_strs = [str(a) for a in actions]
                            print(f"        {tag}: {action_strs}")
                        except Exception as e:
                            print(f"        ERROR {tag}: {e}")
                            traceback.print_exc()
                            schedules[tag] = []
                except Exception as e2:
                    print(f"        Resilient extractor also failed: {e2}")
                    results[model_name] = {"error": f"features: {e2}"}
                    _save_results(args.output, results)
                    continue
            except Exception as e:
                print(f"        ERROR extracting features: {e}")
                traceback.print_exc()
                results[model_name] = {"error": f"features: {e}"}
                _save_results(args.output, results)
                continue

        if not any(schedules.values()):
            print("        No valid schedules.")
            results[model_name] = {
                "model": model_name, "baseline_ns": baseline_ns,
                "optimized_ns": baseline_ns, "speedup": 1.0, "num_ops": len(ops),
            }
            _save_results(args.output, results)
            continue

        # Checkpoint: save schedules so step 5 can resume if it crashes
        _ckpt_path = tag_dir / f"{model_name}_checkpoint.json"
        _save_checkpoint(_ckpt_path, baseline_ns, schedules, ops)

        # [5] Apply schedules + wrap + measure optimized
        print("  [5/5] Applying schedules and measuring ...")
        try:
            opt_code = apply_schedules_to_code(tagged_code, schedules)
            opt_wrapped = add_timing_wrapper(opt_code)
            opt_ns, opt_ok = measure_exec_time(
                exec_engine, opt_wrapped, f"model-{model_name}-opt"
            )
            speedup = (baseline_ns / opt_ns) if (opt_ok and baseline_ns > 0) else 1.0
            if opt_ok:
                print(f"        MLIR baseline: {baseline_ns:,} ns")
                print(f"        MLIR RL (opt): {opt_ns:,} ns")
                print(f"        Speedup:       {speedup:.4f}x")
            else:
                print(f"        MLIR RL (opt): FAILED")
                opt_ns = 0
        except Exception as e:
            print(f"        ERROR: {e}")
            traceback.print_exc()
            opt_ns = 0
            opt_ok = False

        # Clean up checkpoint on success
        if _ckpt_path.exists():
            _ckpt_path.unlink()
            speedup = 1.0

        results[model_name] = {
            "model": model_name,
            "baseline_ns": baseline_ns,
            "optimized_ns": opt_ns,
            "speedup": speedup,
            "num_ops": len(ops),
            "schedules": {t: [str(a) for a in acts] for t, acts in schedules.items()},
        }
        _save_results(args.output, results)

    # --- Summary ---
    print(f"\n{'Model':<30} {'Baseline (ns)':>18} {'Optimized (ns)':>18} {'Speedup':>10}")
    print("-" * 78)
    for name, d in results.items():
        if "error" in d:
            print(f"{name:<30} {'ERROR: ' + d['error'][:40]}")
        else:
            print(f"{name:<30} {d['baseline_ns']:>18,} {d['optimized_ns']:>18,} {d['speedup']:>10.4f}x")


if __name__ == "__main__":
    main()
