"""Benchmark loading and management module.

This module provides functionality for loading benchmark data
and extracting features from benchmark code.
It handles loading MLIR benchmark files, extracting operation features,
and optionally applying img2col transformations
for convolutional operations.
"""

from rl_autoschedular_paper.state import BenchmarkFeatures, extract_bench_features_from_code, extract_bench_features_from_file
from rl_autoschedular_paper.transforms import transform_img2col
from rl_autoschedular_paper.utils.config import Config
from mlir._mlir_libs._mlir.ir import Context, Module  # type: ignore
import json
from tqdm import tqdm
import os

from rl_autoschedular_paper.utils.log import print_alert


class Benchmarks:
    """A class that holds benchmarks data

    Attributes:
        data: The list containing features of loaded benchmarks
    """

    data: list[BenchmarkFeatures]

    def __init__(self, is_training: bool = True):
        """Load benchmarks

        Args:
            is_training: Whether to load train or evaluation set
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
            try:
                benchmark_data = extract_bench_features_from_file(bench_name, bench_file, root_exec_time)
            except RuntimeError as e:
                print_alert(f"Skipping {bench_name}: MLIR crashed during feature extraction ({e})")
                continue
            # NOTE: For now img2col is applied to single operator codes only
            if os.getenv("DISABLE_IMG2COL", "0") != "1" and bench_name.startswith("conv_2d_"):
                modified = False
                with Context():
                    bench_module = Module.parse(benchmark_data.code)
                for op_tag in benchmark_data.operation_tags:
                    if 'conv_2d' not in benchmark_data.operations[op_tag].operation_name:
                        continue
                    try:
                        transform_img2col(bench_module, op_tag)
                    except Exception:
                        print_alert(f"Failed to apply img2col on {bench_name}[{op_tag}]")
                    else:
                        modified = True
                if modified:
                    benchmark_data = extract_bench_features_from_code(bench_name, str(bench_module), root_exec_time)
            self.data.append(benchmark_data)

    def __len__(self) -> int:
        """Get the number of benchmarks loaded.

        Returns:
            The total number of benchmarks.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> BenchmarkFeatures:
        """Get a benchmark by index.

        Args:
            idx: The index of the benchmark to retrieve.

        Returns:
            The benchmark features at the specified index.
        """
        return self.data[idx]
