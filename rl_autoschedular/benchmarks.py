from typing import Union, overload
from rl_autoschedular.state import BenchmarkFeatures, extract_bench_features_from_code, extract_bench_features_from_file
from rl_autoschedular.transforms import transform_img2col
from utils.config import Config
import json
from tqdm import tqdm
import os


class Benchmarks:
    """A class that holds benchmarks data"""

    __data: list[BenchmarkFeatures]
    __data_dict: dict[str, BenchmarkFeatures]

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
        self.__data = []
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
            self.__data.append(benchmark_data)

        self.__data_dict = {bench.bench_name: bench for bench in self.__data}

    def __len__(self):
        return len(self.__data)

    @overload
    def __getitem__(self, idx: int) -> BenchmarkFeatures:
        ...

    @overload
    def __getitem__(self, name: str) -> BenchmarkFeatures:
        ...

    def __getitem__(self, arg: Union[int, str]) -> BenchmarkFeatures:
        if isinstance(arg, int):
            return self.__data[arg]
        else:
            if arg not in self.__data_dict:
                raise KeyError(f"Benchmark {arg} not found")
            return self.__data_dict[arg]
