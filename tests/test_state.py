# test_state.py
from dotenv import load_dotenv
load_dotenv(override=True)

import os
import torch
from rl_autoschedular.state import (
    extract_bench_features_from_file,
    OperationState,
)
from rl_autoschedular.observation import (
    Observation,
    OpFeatures,
    ActionHistory,
    NumLoops,
    ActionMask,
)


def main():
    bench_name = "matmul"
    file_path = "./tests/benchmarks/matmul.mlir"  # path to your mlir file
    root_exec_time = 1000      # dummy baseline time (ns)

    print("=== Extracting Benchmark Features ===")
    bench_features = extract_bench_features_from_file(
        bench_name, file_path, root_exec_time
    )
    print("Benchmark name:", bench_features.bench_name)
    print("Root exec time:", bench_features.root_exec_time)
    print("Operation tags:", bench_features.operation_tags)
    print("Number of operations:", len(bench_features.operations))

    for tag, op in bench_features.operations.items():
        print(f"\n--- Operation: {tag} ---")
        print("Type:", op.operation_type)
        print("Vectorizable:", op.vectorizable)
        print("Op counts:", op.op_count)
        print("Load data:", op.load_data)
        print("Store data:", op.store_data)
        print("Nested loops:")
        for loop in op.nested_loops:
            print(f"  {loop.arg} from {loop.lower_bound} to {loop.upper_bound} "
                  f"step {loop.step} [{loop.iterator_type}]")

    print("\n=== Testing OperationState ===")
    # Just pick the first operation
    first_tag = bench_features.operation_tags[0]
    first_op = bench_features.operations[first_tag]

    op_state = OperationState(
        bench_name=bench_features.bench_name,
        operation_tag=first_tag,
        operation_features=first_op,
        validated_code=bench_features.code,
        transformed_code=first_op.raw_operation,
        step_count=0,
        exec_time=bench_features.root_exec_time,
        transformation_history=[[]],
        tmp_file="tmp.mlir",
        terminal=False
    )

    print("OperationState created:")
    print(" Bench:", op_state.bench_name)
    print(" Tag:", op_state.operation_tag)
    print(" Exec time:", op_state.exec_time)
    print(" Terminal:", op_state.terminal)

    print("\n=== Copy test ===")
    op_state_copy = op_state.copy()
    print("Copied state same tag?", op_state_copy.operation_tag == op_state.operation_tag)

    print("\n=== Observation Tests ===")
    obs = Observation.from_state(op_state)
    print("Observation shape:", obs.shape)
    print("Total observation size (expected):", Observation.cumulative_sizes()[-1])

    # Extract each part
    op_features = Observation.get_part(obs, OpFeatures)
    action_hist = Observation.get_part(obs, ActionHistory)
    num_loops = Observation.get_part(obs, NumLoops)
    action_mask = Observation.get_part(obs, ActionMask)

    print(" OpFeatures shape:", op_features.shape)
    print(" ActionHistory shape:", action_hist.shape)
    print(" NumLoops:", num_loops.item() if isinstance(num_loops, torch.Tensor) else num_loops)
    print(" ActionMask shape:", action_mask.shape)

    # Check consistency
    combined = Observation.get_parts(obs, OpFeatures, ActionHistory, NumLoops, ActionMask)
    print(" Combined parts shape:", combined.shape)
    print(" Matches full obs?", combined.shape[1] == obs.shape[1])


if __name__ == "__main__":
    # Ensure AST_DUMPER_BIN_PATH is set
    if "AST_DUMPER_BIN_PATH" not in os.environ:
        print("ERROR: Please set AST_DUMPER_BIN_PATH to your ast_dumper binary path.")
    else:
        main()
