"""
optimize_model_via_blocks.py
-----------------------------
Block-based model optimization with multi-checkpoint comparison.
Parallelizes block evaluation across multiple CPU workers.

Usage:
  source ~/envs/mlir/bin/activate && set -a && source .env && set +a
  export AUTOSCHEDULER_IMPL=rl_autoschedular_v4_5 CONFIG_FILE_PATH=config/full_model_optim.json
  python scripts/full_model/optimize_model_via_blocks.py \
      --checkpoints model_700.pt model_800.pt ... model_1999.pt \
      --model distilbert --workers 64
"""

import os
import sys
import json
import argparse
import traceback
import multiprocessing
from pathlib import Path
from typing import Optional, List

import torch
import numpy as np

from dotenv import load_dotenv
load_dotenv()
load_dotenv(".env.debug", override=False)

if not os.getenv("CONFIG_FILE_PATH"):
    print("Error: CONFIG_FILE_PATH not set.")
    sys.exit(1)

from utils.config import Config
from utils.implementation import get_autoschedular_impl, import_autoschedular_module

IMPL = get_autoschedular_impl()
HiearchyModel = import_autoschedular_module("model", IMPL).HiearchyModel
Execution = import_autoschedular_module("execution", IMPL).Execution
ActionSpace = import_autoschedular_module("actions", IMPL).ActionSpace
Observation = import_autoschedular_module("observation", IMPL).Observation
extract_bench_features_from_code = import_autoschedular_module("state", IMPL).extract_bench_features_from_code
OperationState = import_autoschedular_module("state", IMPL).OperationState
BenchmarkFeatures = import_autoschedular_module("state", IMPL).BenchmarkFeatures
TiledFusion = import_autoschedular_module("actions.tiled_fusion", IMPL).TiledFusion

from data_utils.extract.extract_blocks import extract_blocks_from_file

_config = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COMPUTE_HEAVY_OPS = {"matmul", "batch_matmul", "conv_2d", "conv_1d", "conv_3d",
                     "batch_matmul_transpose_a", "batch_matmul_transpose_b",
                     "matmul_transpose_a", "matmul_transpose_b",
                     "batch_reduce_matmul", "depthwise_conv_2d"}

import re

def _has_compute_heavy(code: str) -> bool:
    for op in COMPUTE_HEAVY_OPS:
        if f"linalg.{op}" in code:
            return True
    # Also detect linalg.generic with reduction iterator (lowered matmul/contraction)
    if "linalg.generic" in code:
        match = re.search(r'iterator_types\s*=\s*\[([^\]]+)\]', code)
        if match and '"reduction"' in match.group(1):
            return True
    return False


def infer_greedy_schedule(model, bench_features, operation_tag):
    operation_features = bench_features.operations[operation_tag].copy()
    for action in operation_features.pre_actions:
        operation_features = action.update_features(operation_features)

    producer_tag = None
    producer_operand_idx = None
    producer_features = None
    if operation_features.producers:
        producer_tag = operation_features.producers[-1][0]
        producer_operand_idx = min(idx for t, idx in operation_features.producers if t == producer_tag)
        producer_features = bench_features.operations[producer_tag].copy()

    try:
        bench_idx = list(bench_features.operations.keys()).index(operation_tag)
    except ValueError:
        bench_idx = 0

    state = OperationState(
        bench_idx=bench_idx, bench_name=bench_features.bench_name,
        operation_tag=operation_tag,
        original_operation_features=bench_features.operations[operation_tag].copy(),
        operation_features=operation_features,
        producer_tag=producer_tag, producer_operand_idx=producer_operand_idx,
        producer_features=producer_features,
        transformation_history=[[]], terminal=False,
    )

    max_steps = _config.truncate
    steps = 0
    while not state.terminal and steps < max_steps:
        steps += 1
        obs = Observation.from_state(state).to(DEVICE)
        with torch.no_grad():
            actions_index, _, _ = model.sample(obs, greedy=True)
        try:
            action = ActionSpace.action_by_index(actions_index[0], state)
        except Exception:
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
        try:
            state.operation_features = action.update_features(state.operation_features)
        except Exception:
            break
        state.terminal = action.terminal or state.step_count >= _config.truncate

    return state.transformation_history[0]


def apply_actions_to_code(code, actions, op_tag, timeout=30):
    _body = _build_block_action_body(op_tag, actions)
    if not _body:
        return code

    _transform_str = (
        '\nmodule attributes {transform.with_named_sequence} {\n'
        '  transform.named_sequence @__transform_main'
        '(%arg1: !transform.any_op {transform.readonly}) {\n'
        + '\n'.join('    ' + l for l in _body) + '\n'
        '    transform.yield\n'
        '  }\n'
        '}\n'
    )

    import tempfile, pickle, subprocess, sys, os as _os
    tmp = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    pickle.dump((code, _transform_str), tmp)
    tmp.close()

    _out_path = f"{tmp.name}.out"
    worker = (
        'import pickle, sys;'
        'from mlir.ir import Context, Module;'
        'from mlir.dialects.transform import interpreter;'
        f'code, ts = pickle.load(open("{tmp.name}", "rb"));'
        'try:'
        '  with Context():'
        '    m = Module.parse(code);'
        '    t = Module.parse(ts);'
        '    interpreter.apply_named_sequence(m, t.body.operations[0], t);'
        f'    pickle.dump(str(m), open("{_out_path}", "wb"));'
        'except:'
        f'  pickle.dump(None, open("{_out_path}", "wb"));'
    )

    try:
        subprocess.run([sys.executable, '-c', worker],
            timeout=timeout, capture_output=True,
            env={**_os.environ, 'PYTHONUNBUFFERED': '1'})
        if _os.path.exists(f"{tmp.name}.out"):
            with open(f"{tmp.name}.out", "rb") as f:
                result = pickle.load(f)
            return result if result is not None else code
        return code
    except subprocess.TimeoutExpired:
        return code
    except Exception:
        return code
    finally:
        for p in [tmp.name, f"{tmp.name}.out"]:
            if _os.path.exists(p):
                _os.unlink(p)


def _build_block_action_body(op_tag: str, actions: list) -> list[str]:
    body = []
    for step, action in enumerate(actions):
        sid = f"{op_tag}_{step}"
        cls_name = type(action).__name__

        if cls_name == 'NoTransformation':
            continue

        elif cls_name == 'Tiling':
            sizes = action.parameters
            nz = [s for s in sizes if s != 0]
            if not nz:
                continue
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


def _worker_process_block(args_tuple):
    """Process one block across all checkpoints. Runs in a worker subprocess."""
    block_name, code, baseline, ckpt_info_list, impl = args_tuple

    Execution = import_autoschedular_module("execution", impl).Execution
    HiearchyModel = import_autoschedular_module("model", impl).HiearchyModel
    ActionSpace = import_autoschedular_module("actions", impl).ActionSpace
    Observation = import_autoschedular_module("observation", impl).Observation
    extract_bench_features_from_code = import_autoschedular_module("state", impl).extract_bench_features_from_code
    OperationState = import_autoschedular_module("state", impl).OperationState
    TiledFusion = import_autoschedular_module("actions.tiled_fusion", impl).TiledFusion
    _config = Config()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        bench_features = extract_bench_features_from_code(
            bench_name=block_name, code=code,
            root_execution_time=baseline,
        )
    except Exception:
        return {block_name: {n: baseline for n, _, _ in ckpt_info_list}}

    if not bench_features.operation_tags:
        return {block_name: {n: baseline for n, _, _ in ckpt_info_list}}

    # Get schedules for each checkpoint's model
    all_schedules = {}
    for ckpt_name, ckpt_path, _ in ckpt_info_list:
        model = models_dict[ckpt_name]
        schedules = {}
        for tag in bench_features.operation_tags:
            try:
                operation_features = bench_features.operations[tag].copy()
                for action in operation_features.pre_actions:
                    operation_features = action.update_features(operation_features)

                producer_tag = None
                producer_operand_idx = None
                producer_features = None
                if operation_features.producers:
                    producer_tag = operation_features.producers[-1][0]
                    producer_operand_idx = min(idx for t, idx in operation_features.producers if t == producer_tag)
                    producer_features = bench_features.operations[producer_tag].copy()

                try:
                    bench_idx = list(bench_features.operations.keys()).index(tag)
                except ValueError:
                    bench_idx = 0

                state = OperationState(
                    bench_idx=bench_idx, bench_name=bench_features.bench_name,
                    operation_tag=tag,
                    original_operation_features=bench_features.operations[tag].copy(),
                    operation_features=operation_features,
                    producer_tag=producer_tag, producer_operand_idx=producer_operand_idx,
                    producer_features=producer_features,
                    transformation_history=[[]], terminal=False,
                )

                actions_list = []
                while not state.terminal:
                    obs = Observation.from_state(state).to(_device)
                    with torch.no_grad():
                        actions_index, _, _ = model.sample(obs, greedy=True)
                    try:
                        action = ActionSpace.action_by_index(actions_index[0], state)
                    except Exception:
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
                    actions_list.append(action)

                schedules[tag] = actions_list
            except Exception:
                schedules[tag] = []
        all_schedules[ckpt_name] = schedules

    # Apply schedules and measure for each checkpoint
    exec_engine = Execution("")
    result = {}
    for ckpt_name, ckpt_path, _ in ckpt_info_list:
        opt_code = code
        for tag, actions in all_schedules[ckpt_name].items():
            opt_code = apply_actions_to_code(opt_code, actions, tag)

        opt_ns = 0
        try:
            r = exec_engine.execute_code(opt_code, f"{block_name}_{ckpt_name}_opt", [])
            if r[1] and r[0] > 0:
                opt_ns = r[0]
        except Exception:
            pass
        result[ckpt_name] = opt_ns if opt_ns > 0 else baseline

    return {block_name: result}


def _worker_init(ckpt_info_list, impl):
    """Load all RL models (one per checkpoint) once per worker process."""
    global models_dict
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HiearchyModel = import_autoschedular_module("model", impl).HiearchyModel
    models_dict = {}
    for ckpt_name, ckpt_path, _ in ckpt_info_list:
        model = HiearchyModel().to(_device)
        ckpt = torch.load(ckpt_path, map_location=_device, weights_only=False)
        model.load_state_dict(ckpt)
        model.eval()
        models_dict[ckpt_name] = model


def measure_exec(exec_engine, code, bench_name):
    try:
        r = exec_engine.execute_code(code, bench_name, [])
        if r[1] and r[0] > 0:
            return r[0]
    except Exception:
        pass
    return 0


def process_model(model_name: str, checkpoint_paths: List[str], blocks_dir_base: str,
                  models_dir: str, window_size: int = 10, stride: int = 5,
                  skip_extraction: bool = False, workers: int = 1,
                  baselines_json: Optional[str] = None) -> dict:
    model_input = Path(models_dir) / f"{model_name}_linalg.mlir"
    blocks_dir = Path(blocks_dir_base) / model_name

    block_files = sorted(blocks_dir.glob("*.mlir"))
    if skip_extraction and block_files:
        print(f"  Skipping extraction: {len(block_files)} blocks already exist")
        written, skipped = len(block_files), 0
    else:
        print(f"  Extracting blocks (window={window_size}, stride={stride}) ...")
        written, skipped = extract_blocks_from_file(
            input_path=str(model_input), output_dir=str(blocks_dir),
            model_name=model_name, window_size=window_size, stride=stride,
            max_depth=10, max_paths=50,
            batch_candidates=[1, 2, 4, 8, 16, 32, 64],
            batch_fallback=None, manifest_path=None,
        )
        print(f"  Extracted {written} blocks ({skipped} skipped).")

    block_files = sorted(blocks_dir.glob("*.mlir"))
    if not block_files:
        return {"model": model_name, "error": "no blocks extracted"}

    heavy_files = []
    skipped_generic = 0
    for bf in block_files:
        with open(bf) as f:
            code = f.read()
        if _has_compute_heavy(code):
            heavy_files.append(bf)
        else:
            skipped_generic += 1
    print(f"  Filtered: {len(heavy_files)} compute-heavy / {len(block_files)} total ({skipped_generic} generic skipped)")

    if not heavy_files:
        return {"model": model_name, "error": "no compute-heavy blocks after filter"}

    # --- Step 3: Load baselines from pre-measured JSON (no re-measurement) ---
    print(f"  Loading baselines from {baselines_json} ...")
    with open(baselines_json) as f:
        all_baselines = json.load(f)

    model_agg_baseline = all_baselines.get(model_name, 0)
    if model_agg_baseline == 0:
        print(f"  WARNING: No pre-measured baseline for {model_name}. Will measure inline.")

    # Read all block codes
    block_codes = {}
    for bf in heavy_files:
        bn = bf.stem
        with open(bf) as f:
            block_codes[bn] = f.read()

    valid_blocks = list(block_codes.keys())
    total_baseline = model_agg_baseline
    print(f"  Baselines: {len(valid_blocks)} blocks, total={total_baseline:,} ns ({total_baseline/1e6:.2f} ms)")

    ckpt_info_list = []
    for ckpt_path in checkpoint_paths:
        ckpt_name = Path(ckpt_path).stem
        ckpt_info_list.append((ckpt_name, ckpt_path, 0))

    checkpoint_results = {}
    for ckpt_name, _, _ in ckpt_info_list:
        checkpoint_results[ckpt_name] = {
            "baseline_ns": total_baseline,
            "optimized_ns": 0,
            "speedup": 1.0,
            "blocks_used": len(valid_blocks),
            "errors": 0,
        }

    # Estimate per-block baseline for fallback (aggregate / num blocks)
    per_block_baseline = total_baseline / len(valid_blocks) if valid_blocks else 0

    if workers > 1:
        print(f"  Parallel optimization: {workers} workers, {len(valid_blocks)} blocks, {len(checkpoint_paths)} checkpoints ...")

        worker_args = []
        for block_name in valid_blocks:
            code = block_codes[block_name]
            worker_args.append((block_name, code, per_block_baseline, ckpt_info_list, IMPL))

        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=workers, initializer=_worker_init, initargs=(ckpt_info_list, IMPL)) as pool:
            for i, result in enumerate(pool.imap_unordered(_worker_process_block, worker_args)):
                for block_name, ckpt_times in result.items():
                    for ckpt_name, opt_ns in ckpt_times.items():
                        checkpoint_results[ckpt_name]["optimized_ns"] += opt_ns
                        if opt_ns <= 0:
                            checkpoint_results[ckpt_name]["errors"] += 1

                if (i + 1) % 50 == 0:
                    print(f"    [{i+1}/{len(valid_blocks)}] blocks processed")
    else:
        print(f"  Sequential optimization: {len(valid_blocks)} blocks, {len(checkpoint_paths)} checkpoints ...")

        for ckpt_idx, (ckpt_path, ckpt_name) in enumerate([(p, Path(p).stem) for p in checkpoint_paths]):
            print(f"  --- Checkpoint: {ckpt_name} ({ckpt_idx+1}/{len(checkpoint_paths)}) ---")

            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = HiearchyModel().to(_device)
            ckpt = torch.load(ckpt_path, map_location=_device, weights_only=False)
            model.load_state_dict(ckpt)
            model.eval()

            total_optimized = 0
            total_transformations = 0
            errors = 0

            for i, block_name in enumerate(valid_blocks):
                code = block_codes[block_name]
                baseline = per_block_baseline

                try:
                    bench_features = extract_bench_features_from_code(
                        bench_name=f"{model_name}_{block_name}", code=code,
                        root_execution_time=baseline,
                    )
                except Exception:
                    errors += 1
                    total_optimized += baseline
                    continue

                if not bench_features.operation_tags:
                    total_optimized += baseline
                    continue

                schedules = {}
                for tag in list(bench_features.operation_tags):
                    try:
                        actions = infer_greedy_schedule(model, bench_features, tag)
                        schedules[tag] = actions
                    except Exception:
                        schedules[tag] = []

                opt_code = code
                for tag, actions in schedules.items():
                    opt_code = apply_actions_to_code(opt_code, actions, tag)

                total_transformations += sum(
                    1 for actions in schedules.values()
                    for a in actions
                    if type(a).__name__ != 'NoTransformation'
                )

                opt_ns = baseline
                try:
                    exec_engine = Execution("")
                    r = exec_engine.execute_code(opt_code, f"{model_name}_{block_name}_{ckpt_name}_opt", [])
                    if r[1] and r[0] > 0:
                        opt_ns = r[0]
                except Exception:
                    pass
                total_optimized += opt_ns

                if (i + 1) % 100 == 0:
                    current = total_baseline / (total_optimized if total_optimized > 0 else 1)
                    print(f"    [{i+1}/{len(valid_blocks)}] partial sp={current:.4f}x tf={total_transformations} ({errors} err)")

            checkpoint_results[ckpt_name]["optimized_ns"] = total_optimized
            checkpoint_results[ckpt_name]["errors"] = errors
            checkpoint_results[ckpt_name]["transformations"] = total_transformations
            speedup = total_baseline / total_optimized if total_optimized > 0 else 1.0
            print(f"    FINAL: {total_baseline:,} -> {total_optimized:,} = {speedup:.4f}x ({total_transformations} tf)")

    for ckpt_name in checkpoint_results:
        cr = checkpoint_results[ckpt_name]
        cr["speedup"] = cr["baseline_ns"] / cr["optimized_ns"] if cr["optimized_ns"] > 0 else 1.0

    return {
        "model": model_name,
        "method": "block_based_compute_heavy",
        "window_size": window_size,
        "stride": stride,
        "total_blocks_extracted": len(block_files),
        "compute_heavy_blocks": len(valid_blocks),
        "generic_skipped": skipped_generic,
        "total_baseline_ns": total_baseline,
        "checkpoints": checkpoint_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Block-based model optimization — multi-checkpoint compare.")
    parser.add_argument("--checkpoints", nargs="+", default=None, help="Checkpoint .pt files to compare")
    parser.add_argument("--single-checkpoint", default=None, help="Single checkpoint .pt file (trigger per-block output + resume)")
    parser.add_argument("--model", required=True, help="Model name (stem without _linalg)")
    parser.add_argument("--models-dir", default="data/nn/raw_bench")
    parser.add_argument("--blocks-dir", default="data/nn/blocks")
    parser.add_argument("--output", required=True, help="Results JSON path")
    parser.add_argument("--markers-dir", default=None, help="Markers directory for resume (single-ckpt mode)")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--skip-extraction", action="store_true", help="Skip block extraction if blocks already exist")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1 = sequential)")
    parser.add_argument("--baselines-json", default=None, help="Pre-measured baselines JSON (skips baseline measurement)")
    args = parser.parse_args()

    if args.single_checkpoint:
        _run_single_checkpoint(args)
        return

    if not args.checkpoints:
        parser.error("Either --checkpoints or --single-checkpoint required")

    print(f"Model: {args.model}")
    print(f"Checkpoints: {len(args.checkpoints)}")
    print(f"Workers: {args.workers}")
    print(f"Window={args.window_size}, stride={args.stride}")
    if args.skip_extraction:
        print("Skip extraction: enabled")
    if args.baselines_json:
        print(f"Baselines: loading from {args.baselines_json}")
    print()

    result = process_model(
        model_name=args.model,
        checkpoint_paths=args.checkpoints,
        blocks_dir_base=args.blocks_dir,
        models_dir=args.models_dir,
        window_size=args.window_size,
        stride=args.stride,
        skip_extraction=args.skip_extraction,
        workers=args.workers,
        baselines_json=args.baselines_json,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {args.output}")

    if "checkpoints" in result:
        print(f"\n{'Checkpoint':<20} {'Baseline':>16} {'Optimized':>16} {'Speedup':>10} {'TF':>6} {'Blocks':>8} {'Errors':>8}")
        print("-" * 86)
        for ckpt_name, cr in result["checkpoints"].items():
            print(f"{ckpt_name:<20} {cr['baseline_ns']:>16,} {cr['optimized_ns']:>16,} "
                  f"{cr['speedup']:>10.4f}x {cr.get('transformations', 0):>6} {cr['blocks_used']:>8} {cr['errors']:>8}")


# ===========================================================================
# Single-checkpoint mode — per-block output, resume with markers
# ===========================================================================

def _run_single_checkpoint(args):
    ckpt_path = Path(args.single_checkpoint)
    blocks_dir = Path(args.blocks_dir) / args.model
    block_files = sorted(blocks_dir.glob("*.mlir"))
    if not block_files:
        print(f"No blocks found in {blocks_dir}")
        return

    # Load all per-block baselines
    if not args.baselines_json:
        print("ERROR: --baselines-json required in single-ckpt mode")
        sys.exit(1)
    with open(args.baselines_json) as f:
        all_baselines = json.load(f)

    # Load existing output (resume)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {}
    if output_path.exists():
        try:
            output = json.loads(output_path.read_text())
        except Exception:
            output = {}

    # Markers dir for per-block resume
    markers_dir = Path(args.markers_dir or str(output_path.parent / "markers"))
    model_markers = markers_dir / args.model
    model_markers.mkdir(parents=True, exist_ok=True)

    # Determine pending blocks
    pending = []
    skipped_cached = 0
    for bf in block_files:
        bn = bf.stem
        marker = model_markers / f"{bn}.json"
        if bn in output or marker.exists():
            # Load from marker if not in output yet
            if bn not in output and marker.exists():
                try:
                    output[bn] = json.loads(marker.read_text())
                except Exception:
                    pending.append(bf)
                    continue
            skipped_cached += 1
            continue
        pending.append(bf)

    total = len(block_files)
    print(f"Model: {args.model}")
    print(f"Blocks: {total} total, {skipped_cached} cached, {len(pending)} pending")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Output: {output_path}")
    print(f"Markers: {model_markers}")
    print()

    if not pending:
        print("All blocks already evaluated. Done.")
        return

    # Load RL model once
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rl_model = HiearchyModel().to(_device)
    ckpt = torch.load(str(ckpt_path), map_location=_device, weights_only=False)
    rl_model.load_state_dict(ckpt)
    rl_model.eval()

    exec_engine = Execution("")

    heavy_count = 0
    generic_count = 0
    error_count = 0

    def _save_atomic(obj, path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(obj, indent=2) + "\n")
        tmp.rename(path)

    for i, bf in enumerate(pending):
        bn = bf.stem
        with open(bf) as f:
            code = f.read()

        baseline_ns = all_baselines.get(bn, 0)
        if baseline_ns <= 0:
            print(f"  [{i+1}/{len(pending)}] {bn}: no baseline → skip")
            result = {"baseline_ns": 0, "optimized_ns": -1, "error": "no baseline"}
            output[bn] = result
            _save_atomic(result, model_markers / f"{bn}.json")
            _save_atomic(output, output_path)
            error_count += 1
            continue

        if _has_compute_heavy(code):
            # --- RL optimization ---
            try:
                bench_features = extract_bench_features_from_code(
                    bench_name=f"{args.model}_{bn}", code=code,
                    root_execution_time=baseline_ns,
                )
            except Exception:
                print(f"  [{i+1}/{len(pending)}] {bn}: AST dumper fail → baseline")
                result = {"baseline_ns": baseline_ns, "optimized_ns": baseline_ns}
                output[bn] = result
                _save_atomic(result, model_markers / f"{bn}.json")
                _save_atomic(output, output_path)
                error_count += 1
                continue

            if not bench_features.operation_tags:
                print(f"  [{i+1}/{len(pending)}] {bn}: no ops → baseline")
                result = {"baseline_ns": baseline_ns, "optimized_ns": baseline_ns}
                output[bn] = result
                _save_atomic(result, model_markers / f"{bn}.json")
                _save_atomic(output, output_path)
                continue

            schedules = {}
            for tag in list(bench_features.operation_tags):
                try:
                    actions = infer_greedy_schedule(rl_model, bench_features, tag)
                    schedules[tag] = actions
                except Exception:
                    schedules[tag] = []

            opt_code = code
            tf_count = 0
            for tag, actions in schedules.items():
                try:
                    opt_code = apply_actions_to_code(opt_code, actions, tag)
                    tf_count += sum(1 for a in actions if type(a).__name__ != 'NoTransformation')
                except Exception:
                    pass

            # Execute
            opt_ns = baseline_ns
            try:
                r = exec_engine.execute_code(opt_code, f"{args.model}_{bn}_opt", [])
                if r[1] and r[0] > 0:
                    opt_ns = r[0]
            except Exception:
                pass

            result = {"baseline_ns": baseline_ns, "optimized_ns": opt_ns}
            sp = baseline_ns/opt_ns if opt_ns > 0 else 1.0
            marker = "++" if sp > 1.01 else ("--" if sp < 0.99 else " =")
            print(f"  [{i+1}/{len(pending)}] {bn}: heavy  {baseline_ns:>12,} → {opt_ns:>12,} ns  ({tf_count} tf)  {sp:.3f}x {marker}")
            heavy_count += 1
        else:
            # Non-heavy: speedup = 1.0
            result = {"baseline_ns": baseline_ns, "optimized_ns": baseline_ns}
            generic_count += 1

        output[bn] = result
        _save_atomic(result, model_markers / f"{bn}.json")
        _save_atomic(output, output_path)

    # Final save
    _save_atomic(output, output_path)

    # Summary
    total_baseline = sum(v["baseline_ns"] for v in output.values() if v.get("baseline_ns", 0) > 0)
    total_optimized = sum(v["optimized_ns"] for v in output.values() if v.get("optimized_ns", 0) > 0)
    speedup = total_baseline / total_optimized if total_optimized > 0 else 1.0
    print(f"\n  Summary: {heavy_count} heavy, {generic_count} generic, {error_count} errors")
    print(f"  Total: {total_baseline:,} → {total_optimized:,} = {speedup:.4f}x")
    print(f"  Saved → {output_path}")


if __name__ == "__main__":
    main()
