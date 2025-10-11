# tests/test_transforms.py
from dotenv import load_dotenv
load_dotenv(override=True)

import os
from rl_autoschedular.transforms import (
    transform_dialect_TP,
    transform_dialect_tile,
    transform_dialect_interchange,
    transform_dialect_vectorize,
)

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
    file_path = "./tests/benchmarks/matmul.mlir"
    tmp_file = "./tests/tmp.mlir"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")

    with open(file_path, "r") as f:
        code = f.read()

    bench_name = "matmul"
    root_exec_time = 1000      # dummy baseline time (ns)
    ''''''''''''''
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

    

    op_tag = first_tag  # must match the tag used in your MLIR benchmark

    print("\n=== Original Code ===")
    print(code)

    # 1. Tiling + Parallelization
    transformed_tp = transform_dialect_TP(
        code, op_tag, tiling_sizes=[4, 4], tmp_file_path=tmp_file
    )
    print("\n=== After TP (Tiling + Parallelization) ===")
    print(transformed_tp)

    '''
    # 2. Tiling (for-loops)
    transformed_tile = transform_dialect_tile(
        transformed_tp, op_tag, tiling_size=[2, 2], tmp_file_path=tmp_file
    )
    print("\n=== After Tile ===")
    print(transformed_tile)

    # 3. Interchange
    transformed_interchange = transform_dialect_interchange(
        transformed_tile, op_tag, interchange_list=[1, 0], tmp_file_path=tmp_file
    )
    print("\n=== After Interchange ===")
    print(transformed_interchange)

    # 4. Vectorize
    transformed_vectorize = transform_dialect_vectorize(
        transformed_interchange, op_tag, tmp_file_path=tmp_file
    )
    print("\n=== After Vectorize ===")
    print(transformed_vectorize)
    '''

if __name__ == "__main__":
    main()
