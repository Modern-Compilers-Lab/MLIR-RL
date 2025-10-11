# test_action_space.py
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
from rl_autoschedular.actions import ActionSpace

from rl_autoschedular.actions.tiled_parallelization import TiledParallelization





def main():
    bench_name = "matmul"
    file_path = "./tests/benchmarks/matmul.mlir"
    root_exec_time = 1000  # dummy baseline

    print("=== Extract Benchmark Features ===")
    bench_features = extract_bench_features_from_file(
        bench_name, file_path, root_exec_time
    )
    first_tag = bench_features.operation_tags[0]
    first_op = bench_features.operations[first_tag]

    state = OperationState(
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

    print("\n=== ActionSpace Basic Info ===")
    print("Supported actions:", [a.__name__ for a in ActionSpace.supported_actions])
    print("Size:", ActionSpace.size())
    print("Cumulative param sizes:", ActionSpace.cumulative_params_sizes())
    print("Cumulative mask sizes:", ActionSpace.cumulative_mask_sizes())
    print("Cumulative history sizes:", ActionSpace.cumulative_history_sizes())

    print("\n=== Action Lookup ===")
    for i, act in enumerate(ActionSpace.supported_actions):
        print(f"Index {i}: {act.__name__}, number={ActionSpace.action_number(act)}, symbol={act.symbol}")

    # Try lookup by symbol
    sym = ActionSpace.supported_actions[0].symbol
    print(f"Symbol '{sym}' resolves to:", ActionSpace.action_type_by_symbol(sym).__name__)
    print(f"Symbol '{sym}' has number:", ActionSpace.action_number_by_symbol(sym))

    print("\n=== Action By Index ===")
    cum_sizes = ActionSpace.cumulative_params_sizes()
    for i, act in enumerate(ActionSpace.supported_actions):
        # construct a dummy index tensor with selection + params
        index = torch.zeros(cum_sizes[-1], dtype=torch.long)
        index[0] = i  # select action
        action = ActionSpace.action_by_index(index, state)
        print(f"Constructed action {action} from index {i}")

    print("\n=== Action Mask ===")
    mask = ActionSpace.action_mask(state)
    print("Action mask:", mask.tolist(), " length:", len(mask))
    
    '''
    # what does the class say about the current state?
    print("\n=== TP Mask ===")
    tp_action_mask = TiledParallelization.action_mask(state)
    print("TP mask:", tp_action_mask.tolist())
    '''
    
    


    print("\n=== Action History ===")
    history = ActionSpace.action_history(state)
    print("History tensor:", history.tolist() if history.numel() else "empty")

    print("\n=== Observation Integration ===")
    obs = Observation.from_state(state)
    action_mask = Observation.get_part(obs, ActionMask)
    print("ActionMask from obs shape:", action_mask.shape)

    print("\n=== Distributions ===")
    selection_logits = torch.randn(1, ActionSpace.size())
    actions_logits = [torch.randn(1, size) if size > 0 else None
                      for size in [a.mask_size() for a in ActionSpace.supported_actions]]
    
    print('Actions logits sizes:', [logits.shape if logits is not None else None for logits in actions_logits])
    dists = ActionSpace.distributions(obs, selection_logits, *actions_logits)
    
    print("Number of distributions:", len(dists))
    print('Dists sizes:', dists.shape)

    print("\n=== Uniform Distributions ===")
    uniform_dists = ActionSpace.uniform_distributions(obs)
    print("Number of uniform dists:", len(uniform_dists))

    print("\n=== Distributions Stats ===")
    index = ActionSpace.sample(obs, dists, uniform_dists)  # sample an index
    logp, entropy = ActionSpace.distributions_stats(dists, index, uniform_dists, eps=0.1)
    print("Log prob:", logp)
    print("Entropy:", entropy)

    print("\n=== Sampling ===")
    sample1 = ActionSpace.sample(obs, dists, uniform_dists, greedy=True)
    sample2 = ActionSpace.sample(obs, dists, uniform_dists, uniform=True)
    sample3 = ActionSpace.sample(obs, dists, uniform_dists)
    print("Greedy sample:", sample1.tolist())
    print("Uniform sample:", sample2.tolist())
    print("Random sample:", sample3.tolist())
    


if __name__ == "__main__":
    main()
