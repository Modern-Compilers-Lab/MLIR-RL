from rl_autoschedular.env import Env
from rl_autoschedular.model import apply_masks, extract_masks, indices_to_raw_actions
import torch
import math
from torch.distributions import Categorical, Distribution, Uniform
from rl_autoschedular import config as cfg
from tqdm import tqdm


N = cfg.num_transformations
L = cfg.max_num_loops
TS = cfg.num_tile_sizes
match cfg.interchange_mode:
    case 'enumerate':
        interchange_mask = 3 * L - 6
    case 'pointers':
        interchange_mask = L
    case 'continuous':
        interchange_mask = 0
action_mask_size = N + 2 * L * (TS + 1) + interchange_mask


def create_uniform_distributions(obs: torch.Tensor, num_loops: list[int]) -> tuple[Distribution, Distribution, Distribution, Distribution]:
    """Create uniform distributions for the actions.

    Args:
        obs (torch.Tensor): The input tensor.

    Returns:
        tuple[Distribution, Distribution, Distribution, Distribution]: The uniform distributions for the transformations, parallelizations, tilings, and interchanges.
    """
    batch_size = obs.shape[0]
    action_mask = obs[:, -(action_mask_size):].bool()

    transformation_logits = torch.zeros((batch_size, N), dtype=torch.float32)
    parallelization_logits = torch.zeros((batch_size, L, TS + 1), dtype=torch.float32)
    tiling_logits = torch.zeros((batch_size, L, TS + 1), dtype=torch.float32)
    match cfg.interchange_mode:
        case 'enumerate':
            interchange_logits = torch.zeros((batch_size, 3 * L - 6), dtype=torch.float32)
        case 'pointers':
            interchange_logits = torch.zeros((batch_size, L), dtype=torch.float32)
        case 'continuous':
            interchange_logits = torch.zeros((batch_size, 1), dtype=torch.float32)

    # Apply masks on logits
    transformation_logits, parallelization_logits, tiling_logits, interchange_logits = apply_masks(transformation_logits, parallelization_logits, tiling_logits, interchange_logits, *extract_masks(action_mask))

    # Create distributions with the masked probabilities
    transformation_dist = Categorical(logits=transformation_logits)
    parallelization_dist = Categorical(logits=parallelization_logits)
    tiling_dist = Categorical(logits=tiling_logits)
    if cfg.interchange_mode != 'continuous':
        interchange_dist = Categorical(logits=interchange_logits)
    else:
        total_count = torch.tensor([math.factorial(loops) for loops in num_loops], dtype=torch.float64)
        interchange_dist = Uniform(0.0, total_count)

    return transformation_dist, parallelization_dist, tiling_dist, interchange_dist


if __name__ == "__main__":
    env = Env(is_training=True)
    print(f"Environments initialized: {env.tmp_file}")

    pbar = tqdm(unit="bench")
    while True:
        state, obs = env.reset()
        bench_done = False
        while not bench_done:
            num_loops = len(state.operation_features.nested_loops)
            transformation_eps_dist, parallelization_eps_dist, tiling_eps_dist, interchange_eps_dist = create_uniform_distributions(obs, [num_loops])
            transformation_index = transformation_eps_dist.sample()
            parallelization_index = parallelization_eps_dist.sample()
            tiling_index = tiling_eps_dist.sample()
            interchange_index = interchange_eps_dist.sample().long()
            actions = indices_to_raw_actions(transformation_index, parallelization_index, tiling_index, interchange_index, [num_loops])
            next_state, next_obs, _, op_done, _ = env.step(state, actions[0])
            if op_done:
                next_state, next_obs, bench_done = env.get_next_op_state(next_state)
            state = next_state
            obs = next_obs
        pbar.update(1)
