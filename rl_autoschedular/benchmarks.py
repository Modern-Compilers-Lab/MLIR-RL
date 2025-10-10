from rl_autoschedular.state import BenchmarkFeatures, extract_bench_features_from_code, extract_bench_features_from_file
from rl_autoschedular.transforms import transform_img2col
from utils.config import Config
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
        cfg = Config()
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
            modified = False
            bench_code = benchmark_data.code
            for op_tag in benchmark_data.operation_tags:
                if 'conv_2d' not in benchmark_data.operations[op_tag].operation_name:
                    continue
                bench_code = transform_img2col(bench_code, op_tag)
                modified = True
            if modified:
                benchmark_data = extract_bench_features_from_code(bench_name, bench_code, root_exec_time)
            self.data.append(benchmark_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
