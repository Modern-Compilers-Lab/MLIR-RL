from rl_autoschedular.state import BenchmarkFeatures, extract_bench_features_from_file
from rl_autoschedular import config as cfg
import json
from tqdm import tqdm
import os


class Benchmarks:
    """A class that holds benchmarks data"""

    data: list[BenchmarkFeatures]

    def __init__(self, is_training: bool = True):
        """Load benchmarks

        Args:
            is_training (bool): Whether to load train or evaluation set
        """
        # Load benchmark names and execution times from json file
        bench_json_file = cfg.json_file

        # If we are in evaluation mode, use the evaluation json file if provided
        if cfg.eval_json_file and not is_training:
            bench_json_file = cfg.eval_json_file

        with open(bench_json_file) as file:
            benchmarks_json: dict[str, int] = json.load(file)

        # Build benchmark features
        self.data = []
        for bench_name, root_exec_time in tqdm(benchmarks_json.items(), desc="Extracting benchmark features", unit="bench"):
            bench_file = os.path.join(cfg.benchmarks_folder_path, bench_name + ".mlir")
            benchmark_data = extract_bench_features_from_file(bench_name, bench_file, root_exec_time)

            if cfg.split_ops and is_training and len(benchmark_data.operation_tags) > 1 and 'lqcd' in benchmark_data.bench_name:
                # Split LQCD benchmarks into multiple single operations
                # TODO: Improve with operatine-wise timing
                # TODO: Convert LQCD to tensor-based to elliminate all of this
                for tag in benchmark_data.operation_tags:
                    # Create a new benchmark data with only the current operation
                    new_bench_data = benchmark_data.copy()
                    new_bench_data.bench_name = f"{benchmark_data.bench_name}_{tag}"
                    new_bench_data.operation_tags = [tag]
                    new_bench_data.operations = {tag: new_bench_data.operations[tag]}
                    self.data.append(new_bench_data)
            else:
                self.data.append(benchmark_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
