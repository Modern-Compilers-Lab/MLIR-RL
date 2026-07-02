from rl_autoschedular_v4_9.state import BenchmarkFeatures, extract_bench_features_from_code, extract_bench_features_from_file
from rl_autoschedular_v4_9.transforms import transform_img2col
from utils.config import Config
from utils.implementation import get_autoschedular_impl, get_split_file_path
from pathlib import Path
import json
from tqdm import tqdm
import os


def _resolve_bench_file(cfg: Config, is_training: bool) -> str:
    """Resolve the train/eval JSON file path.

    Priority:
    1. cfg.json_file / cfg.eval_json_file (explicit in config — backward compat)
    2. Auto-derived from implementation + results_dir (preferred)
    """
    if is_training and cfg.json_file:
        return cfg.json_file
    if not is_training and cfg.eval_json_file:
        return cfg.eval_json_file

    implementation = get_autoschedular_impl()
    return str(get_split_file_path(cfg.results_dir, implementation, is_training))


class Benchmarks:
    """A class that holds benchmarks data"""

    data: list[BenchmarkFeatures]

    def __init__(self, is_training: bool = True):
        """Load benchmarks

        Args:
            is_training (bool): Whether to load train or evaluation set
        """
        cfg = Config()
        bench_json_file = _resolve_bench_file(cfg, is_training)

        with open(bench_json_file) as file:
            benchmarks_json: dict[str, int] = json.load(file)

        # Build benchmark features
        self.data = []
        for bench_name, root_exec_time in tqdm(benchmarks_json.items(), desc="Extracting benchmark features", unit="bench"):
            bench_file = os.path.join(cfg.benchmarks_folder_path, bench_name + ".mlir")
            try:
                benchmark_data = extract_bench_features_from_file(bench_name, bench_file, root_exec_time)
            except Exception as e:
                print(f"Warning: Failed to extract features for {bench_name}: {e}")
                continue

            modified = False
            bench_code = benchmark_data.code
            for op_tag in benchmark_data.operation_tags:
                if 'conv_2d' not in benchmark_data.operations[op_tag].operation_name:
                    continue
                try:
                    bench_code = transform_img2col(bench_code, op_tag)
                    modified = True
                except Exception as e:
                    # If transform fails (e.g., shape incompatible), silently skip transformation
                    pass
            if modified:
                try:
                    benchmark_data = extract_bench_features_from_code(bench_name, bench_code, root_exec_time)
                except Exception as e:
                    print(f"Warning: Could not extract features from transformed code for {bench_name}: {e}")
                    pass
            if not benchmark_data.operation_tags:
                print(f"Warning: No operations found for {bench_name}, skipping")
                continue
            self.data.append(benchmark_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
