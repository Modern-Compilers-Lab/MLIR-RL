# IQL for MLIR-RL

Example for `config.json` :
```
{
    "max_num_stores_loads": 7,
    "max_num_loops": 7,
    "max_num_load_store_dim": 7,
    "num_tile_sizes": 7,
    "vect_size_limit": 2048,
    "order": [["I"],["TP"],["T"],["V","NT"]],
    "interchange_mode": "pointers",
    "exploration": ["entropy"],
    "init_epsilon": 0.1,
    "new_architecture": false,
    "normalize_bounds": "max",
    "normalize_adv": "standard",
    "sparse_reward": true,
    "split_ops": true,  
    "reuse_experience": "none",
    "activation": "relu",
    "benchmarks_folder_path": "data/matmul/code/",
    "bench_count": 8,
    "replay_count": 10,
    "nb_iterations": 1200,
    "ppo_epochs": 4,
    "ppo_batch_size": 32,
    "value_epochs": 4,
    "value_batch_size": 32,
    "value_coef": 0.5,
    "value_clip": true,
    "entropy_coef": 0.01,
    "lr": 3e-4,
    "truncate": 10,
    "json_file": "data/matmul/train_operations.json",
    "eval_json_file": "data/matmul/eval_operations.json",
    "tags": ["matmul"],
    "debug": false,
    "main_exec_data_file": "cache/execution.json",
    "results_dir": "offline_iql_adv_norm_gradclip_cosine_scheduler",
    "run_name": "offline_iql_adv_norm_gradclip_cosine_scheduler",
    "collect_offline_data": false,
    "offline_data_save_dir": "offline_dataset",
    "offline_data_file": "offline_dataset_online_ppo.npz",
    
    "gamma": 0.99,
    "tau": 0.9,
    "inverse_temperature":3.0,
    "alpha": 0.005,
    "batch_size": 256,
    "learning_rate": {
        "value": 3e-4,
        "q": 3e-4,
        "policy": 1e-4
    },
    "max_steps": 1000000,
    "target_update_freq": 1
}
```