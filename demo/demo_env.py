import os
from utils.environment import ParallelEnv
from utils.ppo_model import HiearchyModel as MyModel
from utils.consts import (
    MAX_NUM_STORES_LOADS,
    MAX_NUM_LOOPS,
    MAX_NUM_LOAD_STORE_DIM
)

import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


CONFIG = {
    'len_trajectory': 64,
    'ppo_batch_size': 64,
    'steps':10000,
    'ppo_epochs':4,
    'logs':True,
    'entropy_coef':0.01,
    'lr':0.001,
    'truncate':5,
    'json_file': os.path.abspath("generated_data/bigger_input_nn_(32x230x230x3)operations.json"),
}

env = ParallelEnv(
    json_file=CONFIG["json_file"],
    num_env=1,
    truncate=CONFIG["truncate"],
    reset_repeat=1,
    step_repeat=1,
)



# Define the model:
input_dim = MAX_NUM_LOOPS + MAX_NUM_LOOPS*MAX_NUM_LOAD_STORE_DIM*MAX_NUM_STORES_LOADS + MAX_NUM_LOOPS*MAX_NUM_LOAD_STORE_DIM + 5 + \
    MAX_NUM_LOOPS*3*CONFIG["truncate"]
print('input_dim:', input_dim)

model = MyModel(
    input_dim=input_dim,
    num_loops=MAX_NUM_LOOPS
)

# model.load_state_dict(torch.load('models/demo_model_bigger_nn.pt'))
model.load_state_dict(torch.load('models/demo_ppo_model.pt'))



benchmark_operations = [
    ('linalg.matmul ins(%collapsed, %27 : tensor<32x10816xf32>, tensor<10816x120xf32>) outs(%29 : tensor<32x120xf32>) -> tensor<32x120xf32>', 9),
    ('linalg.matmul ins(%32, %34 : tensor<32x120xf32>, tensor<120x84xf32>) outs(%36 : tensor<32x84xf32>) -> tensor<32x84xf32>', 13),
    ('linalg.matmul ins(%39, %41 : tensor<32x84xf32>, tensor<84x10xf32>) outs(%43 : tensor<32x10xf32>) -> tensor<32x10xf32>', 17),

    ('linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%0, %1 : tensor<32x3x230x230xf32>, tensor<64x3x7x7xf32>) outs(%12 : tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xf32>', 1),
    ('linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%18, %3 : tensor<32x64x56x56xf32>, tensor<16x64x5x5xf32>) outs(%20 : tensor<32x16x52x52xf32>) -> tensor<32x16x52x52xf32>', 5),

    ('linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%14, %17 : tensor<32x64x112x112xf32>, tensor<2x2xf32>) outs(%16 : tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf32>', 3),
    ('linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%22, %17 : tensor<32x16x52x52xf32>, tensor<2x2xf32>) outs(%24 : tensor<32x16x26x26xf32>) -> tensor<32x16x26x26xf32>', 7),

    ('linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13 : tensor<32x64x112x112xf32>) outs(%11 : tensor<32x64x112x112xf32>) {\n^bb0(%in: f32, %out: f32):\n  %cst_1 = arith.constant 0.000000e+00 : f32\n  %46 = arith.cmpf ugt, %in, %cst_1 : f32\n  %47 = arith.select %46, %in, %cst_1 : f32\n  linalg.yield %47 : f32\n} -> tensor<32x64x112x112xf32>', 2),
    ('linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<120x10816xf32>) outs(%26 : tensor<10816x120xf32>) {\n^bb0(%in: f32, %out: f32):\n  linalg.yield %in : f32\n} -> tensor<10816x120xf32>', 8),
]



for operation, i in benchmark_operations:

    print(f'Operation ({i}):', env.env.operations_files[i][0])

    # Reset the environement with the specific operation
    state, obs = env.reset(i)
    obs = torch.cat(obs).to(device)

    while True:

        with torch.no_grad():
            # Select the action using the model
            action, action_log_p, values, entropy = model.sample(obs)

        # Apply the action and get the next state
        next_obs, reward, terminated, truncated, next_state, final_state = env.step(state, action, model)

        done = terminated[0] or truncated[0]
        if done:
            final_state = final_state[0]
            speedup_metric = final_state.root_exec_time /  final_state.exec_time
            print('Base execution time:', final_state.root_exec_time / 1e9, 'ms')
            print('New execution time:', final_state.exec_time / 1e9, 'ms')
            print('speedup:', speedup_metric)
            break

        state = next_state
        obs = torch.cat(next_obs).to(device)

    print('\n\n\n')
