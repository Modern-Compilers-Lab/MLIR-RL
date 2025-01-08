import os
from utils.singleton import Singleton
import json
from typing import Literal


class Config(metaclass=Singleton):
    """Class to store and load global configuration"""
    max_num_stores_loads: int
    """The maximum number of loads in the nested loops"""
    max_num_loops: int
    """The max number of nested loops"""
    max_num_load_store_dim: int
    """The max number of dimensions in load/store buffers"""
    num_tile_sizes: int
    """The number of tile sizes"""
    num_transformations: int
    """The number of transformations"""
    vect_size_limit: int
    """Vectorization size limit to prevent large sizes vectorization"""
    use_bindings: bool
    """Flag to enable using python bindings for execution, if False, the execution will be done using the command line. Default is False."""
    use_vectorizer: bool
    """Flag to enable using the vectorizer C++ program for vectorization, if False, vectorization is done using transform dialect directly. Default is False."""
    data_format: Literal["json", "mlir"]
    """The format of the data, can be either "json" or "mlir". "json" mode reads json files containing benchmark features, "mlir" mode reads mlir code files directly and extract features from it using AST dumper. Default is "json"."""
    optimization_mode: Literal["last", "all"]
    """The optimization mode to use, "last" will optimize only the last operation, "all" will optimize all operations in the code. Default is "last"."""
    benchmarks_folder_path: str
    """Path to the benchmarks folder. Can be empty if optimization mode is set to "last"."""
    len_trajectory: int
    """Length of the trajectory"""
    ppo_batch_size: int
    """Batch size for PPO"""
    nb_iterations: int
    """Number of iterations"""
    ppo_epochs: int
    """Number of epochs for PPO"""
    entropy_coef: float
    """Entropy coefficient"""
    lr: float
    """Learning rate"""
    truncate: int
    """Maximum number of steps in the schedule"""
    json_file: str
    """Path to the JSON file containing the benchmarks code or features."""
    logging: bool
    """Flag to enable logging to neptune"""

    loaded: bool
    """Flag to check if the config was already loaded from JSON file or not"""

    def __init__(self):
        """Initialize the default values"""
        self.max_num_stores_loads = 7
        self.max_num_loops = 7
        self.max_num_load_store_dim = 7
        self.num_tile_sizes = 7
        self.num_transformations = 5
        self.vect_size_limit = 512
        self.use_bindings = False
        self.use_vectorizer = False
        self.data_format = "json"
        self.optimization_mode = "last"
        self.benchmarks_folder_path = ""
        self.len_trajectory = 64
        self.ppo_batch_size = 64
        self.nb_iterations = 10000
        self.ppo_epochs = 4
        self.entropy_coef = 0.01
        self.lr = 0.001
        self.truncate = 5
        self.json_file = ""
        self.logging = True
        self.loaded = False

    def load_from_json(self):
        """Load the configuration from the JSON file."""
        # Open the JSON file
        with open(os.getenv("CONFIG_FILE_PATH"), "r") as f:
            config = json.load(f)
        # Set the configuration values
        self.max_num_stores_loads = config["max_num_stores_loads"]
        self.max_num_loops = config["max_num_loops"]
        self.max_num_load_store_dim = config["max_num_load_store_dim"]
        self.num_tile_sizes = config["num_tile_sizes"]
        self.num_transformations = config["num_transformations"]
        self.vect_size_limit = config["vect_size_limit"]
        self.use_bindings = config["use_bindings"]
        self.use_vectorizer = config["use_vectorizer"]
        self.data_format = config["data_format"]
        self.optimization_mode = config["optimization_mode"]
        self.benchmarks_folder_path = config["benchmarks_folder_path"]
        self.len_trajectory = config["len_trajectory"]
        self.ppo_batch_size = config["ppo_batch_size"]
        self.nb_iterations = config["nb_iterations"]
        self.ppo_epochs = config["ppo_epochs"]
        self.entropy_coef = config["entropy_coef"]
        self.lr = config["lr"]
        self.truncate = config["truncate"]
        self.json_file = config["json_file"]
        self.logging = config["logging"]
        # Check the configuration values
        assert self.data_format in ["json", "mlir"], "Invalid data format. Should be 'json' or 'mlir'."
        assert self.optimization_mode in ["last", "all"], "Invalid optimization mode. Should be 'last' or 'all'."
        assert len(self.benchmarks_folder_path) > 0 or self.data_format == "json", "Benchmark folder path should be set if data_format is 'mlir'."
        assert self.data_format != "json" or not self.use_bindings, "The specific case of using python bindings with JSON data format is not implemented yet."
        # Set loaded flag
        self.loaded = True

    def to_dict(self):
        """Convert the configuration to a dictionary."""
        return {
            "max_num_stores_loads": self.max_num_stores_loads,
            "max_num_loops": self.max_num_loops,
            "max_num_load_store_dim": self.max_num_load_store_dim,
            "num_tile_sizes": self.num_tile_sizes,
            "num_transformations": self.num_transformations,
            "vect_size_limit": self.vect_size_limit,
            "use_bindings": self.use_bindings,
            "use_vectorizer": self.use_vectorizer,
            "data_format": self.data_format,
            "optimization_mode": self.optimization_mode,
            "benchmarks_folder_path": self.benchmarks_folder_path,
            "len_trajectory": self.len_trajectory,
            "ppo_batch_size": self.ppo_batch_size,
            "nb_iterations": self.nb_iterations,
            "ppo_epochs": self.ppo_epochs,
            "entropy_coef": self.entropy_coef,
            "lr": self.lr,
            "truncate": self.truncate,
            "json_file": self.json_file,
            "logging": self.logging
        }

    def __str__(self):
        """Convert the configuration to a string."""
        return str(self.to_dict())
