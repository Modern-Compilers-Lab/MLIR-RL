import torch
import time
import json
import os
import sys

# Setup base config to avoid issues with missing keys
def setup_config(overrides):
    with open("config/train1.json", "r") as f:
        cfg = json.load(f)
    cfg.update({
        "implementation": "rl_autoschedular_v4_5",
        "json_file": "/tmp/tiny_train.json",
        "eval_json_file": "/tmp/tiny_eval.json",
        "bench_count": 1,
        "nb_iterations": 5,
        "ppo_epochs": 1,
        "truncate": 2,
        "results_dir": "results/test_v3_sanity",
    })
    cfg.update(overrides)
    with open("/tmp/ablation_config.json", "w") as f:
        json.dump(cfg, f)
    os.environ["CONFIG_FILE_PATH"] = "/tmp/ablation_config.json"

def run_ablation(name, pooling, num_layers, history_token):
    setup_config({
        "transformer_pooling": pooling,
        "transformer_num_layers": num_layers,
        "transformer_use_action_history_token": history_token,
        "transformer_d_model": 256,
        "transformer_nhead": 8
    })
    
    from rl_autoschedular_v4_5.model import HierarchyModel
    from rl_autoschedular_v4_5.observation import Observation
    
    # Cumulative sizes
    sizes = [p.size() for p in Observation.parts]
    total_size = sum(sizes)
    
    model = HierarchyModel()
    model.eval()
    
    # [batch, total_size]
    obs = torch.randn(1, total_size)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model.get_embedding(obs)
            
    # Benchmark
    start = time.time()
    for _ in range(30):
        with torch.no_grad():
            emb = model.get_embedding(obs)
    end = time.time()
    avg_ms = (end - start) * 1000 / 30
    
    param_count = sum(p.numel() for p in model.parameters())
    emb_size = emb.shape[-1]
    
    print(f"{name}|{avg_ms:.2f}|{emb_size}|{param_count}")

if __name__ == "__main__":
    if len(sys.argv) == 5:
        name = sys.argv[1]
        pooling = sys.argv[2]
        layers = int(sys.argv[3])
        history = sys.argv[4].lower() == "true"
        run_ablation(name, pooling, layers, history)
