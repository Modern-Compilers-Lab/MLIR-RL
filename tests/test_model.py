# test_model.py
from dotenv import load_dotenv
load_dotenv(override=True)

import os
import torch

from rl_autoschedular.observation import Observation, OpFeatures, ActionHistory, NumLoops, ActionMask
from rl_autoschedular.state import extract_bench_features_from_file, OperationState
from rl_autoschedular.model import PolicyModel   # adjust if path is different
from rl_autoschedular.actions import ActionSpace


def main():
    bench_name = "matmul"
    file_path = "./tests/benchmarks/matmul.mlir"  # path to your mlir file
    root_exec_time = 1000

    print("=== Load benchmark and make state ===")
    bench_features = extract_bench_features_from_file(
        bench_name, file_path, root_exec_time
    )
    first_tag = bench_features.operation_tags[0]
    first_op = bench_features.operations[first_tag]

    state = OperationState(
        bench_name=bench_name,
        operation_tag=first_tag,
        operation_features=first_op,
        validated_code=bench_features.code,
        transformed_code=first_op.raw_operation,
        step_count=0,
        exec_time=bench_features.root_exec_time,
        transformation_history=[[]],
        tmp_file="tmp.mlir",
        terminal=False,
    )

    obs = Observation.from_state(state)

    print("\n=== Build PolicyModel ===")
    obs_parts = [OpFeatures, ActionHistory]
    policy = PolicyModel(obs_parts)

    # random weights are already in place by default initialization
    dists = policy(obs)

    print("\n=== Policy Outputs ===")
    print("Number of distributions:", len(dists))
    for i, dist in enumerate(dists):
        if dist is None:
            print(f" Head {i}: None (no params)")
        else:
            print(f" Head {i}: dist type={type(dist).__name__}, batch shape={dist.batch_shape}, event shape={dist.event_shape}")

    print("\n=== Sample from Policy ===")
    index = ActionSpace.sample(obs, dists, dists)  # reuse same dists as eps_dists
    
    # action by index
    action_index = ActionSpace.action_by_index(index[0],state) 
    print("Sampled index:", index)
    print("Sampled action:", action_index)
    
    
    


if __name__ == "__main__":
    main()
