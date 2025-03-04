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
    init_action_mask: list[bool]
    """The initial action mask"""
    interchange_mode: Literal['enumerate', 'pointers', 'continuous']
    """The method used for interchange action"""
    interchange_distribution: Literal['binomial', 'normal']
    """The distribution used for continuous interchange action"""
    use_exploration: bool
    """Flag to enable using the mean of the interchange distribution instead of the sampled value (only in case of continuous interchange)"""
    use_bindings: bool
    """Flag to enable using python bindings for execution, if False, the execution will be done using the command line. Default is False."""
    use_vectorizer: bool
    """Flag to enable using the vectorizer C++ program for vectorization, if False, vectorization is done using transform dialect directly. Default is False."""
    update_op_features: bool
    """Flag to enable updating the operation features between steps"""
    reverse_history: bool
    """Flag to indicate if the history should be reversed or not"""
    new_architecture: bool
    """Flag to indicate if the new architecture should be used or not"""
    normalize_bounds: bool
    """Flag to indicate if the upper bounds in the input should be normalized or not"""
    mul_log_p: bool
    """Flag to indicate if the log probability should be multiplied by 0 or selected"""
    force_vector: bool
    """Flag to force vectorization"""
    punish_vector: bool
    """Flag to punish lack of vectorization"""
    activation: Literal["relu", "tanh"]
    """The activation function to use in the network"""
    data_format: Literal["json", "mlir"]
    """The format of the data, can be either "json" or "mlir". "json" mode reads json files containing benchmark features, "mlir" mode reads mlir code files directly and extract features from it using AST dumper. Default is "json"."""
    optimization_mode: Literal["last", "all"]
    """The optimization mode to use, "last" will optimize only the last operation, "all" will optimize all operations in the code. Default is "last"."""
    benchmarks_folder_path: str
    """Path to the benchmarks folder. Can be empty if optimization mode is set to "last"."""
    bench_count: int
    """Number of batches in a trajectory"""
    nb_iterations: int
    """Number of iterations"""
    value_epochs: int
    """Number of epochs for value training"""
    ppo_epochs: int
    """Number of epochs for PPO"""
    ppo_batch_size: int
    """Batch size for PPO"""
    value_coef: float
    """Value coefficient"""
    entropy_coef: float
    """Entropy coefficient"""
    lr: float
    """Learning rate"""
    truncate: int
    """Maximum number of steps in the schedule"""
    json_file: str
    """Path to the JSON file containing the benchmarks code or features."""
    tags: list[str]
    """List of tags to add to the neptune experiment"""
    logging: bool
    """Flag to enable logging to neptune"""
    exec_data_file: str
    """Path to the file containing the execution data"""

    loaded: bool
    """Flag to check if the config was already loaded from JSON file or not"""

    def __init__(self):
        """Initialize the default values"""
        self.max_num_stores_loads = 7
        self.max_num_loops = 7
        self.max_num_load_store_dim = 7
        self.num_tile_sizes = 7
        self.num_transformations = 6
        self.vect_size_limit = 512
        self.init_action_mask = [False, True, False, False, False, False]
        self.use_exploration = False
        self.interchange_mode = "enumerate"
        self.interchange_distribution = "binomial"
        self.use_bindings = False
        self.use_vectorizer = False
        self.update_op_features = False
        self.reverse_history = True
        self.new_architecture = False
        self.normalize_bounds = True
        self.mul_log_p = False
        self.force_vector = True
        self.punish_vector = False
        self.activation = "relu"
        self.data_format = "json"
        self.optimization_mode = "last"
        self.benchmarks_folder_path = ""
        self.bench_count = 20
        self.nb_iterations = 10000
        self.value_epochs = 4
        self.ppo_epochs = 4
        self.ppo_batch_size = 4
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.lr = 0.001
        self.truncate = 5
        self.json_file = ""
        self.tags = []
        self.logging = True
        self.exec_data_file = ""
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
        self.init_action_mask = config["init_action_mask"]
        self.interchange_mode = config["interchange_mode"]
        self.interchange_distribution = config["interchange_distribution"]
        self.use_exploration = config["use_exploration"]
        self.use_bindings = config["use_bindings"]
        self.use_vectorizer = config["use_vectorizer"]
        self.update_op_features = config["update_op_features"]
        self.reverse_history = config["reverse_history"]
        self.new_architecture = config["new_architecture"]
        self.normalize_bounds = config["normalize_bounds"]
        self.mul_log_p = config["mul_log_p"]
        self.force_vector = config["force_vector"]
        self.punish_vector = config["punish_vector"]
        self.activation = config["activation"]
        self.data_format = config["data_format"]
        self.optimization_mode = config["optimization_mode"]
        self.benchmarks_folder_path = config["benchmarks_folder_path"]
        self.bench_count = config["bench_count"]
        self.nb_iterations = config["nb_iterations"]
        self.value_epochs = config["value_epochs"]
        self.ppo_epochs = config["ppo_epochs"]
        self.ppo_batch_size = config["ppo_batch_size"]
        self.value_coef = config["value_coef"]
        self.entropy_coef = config["entropy_coef"]
        self.lr = config["lr"]
        self.truncate = config["truncate"]
        self.json_file = config["json_file"]
        self.tags = config["tags"]
        self.logging = config["logging"]
        self.exec_data_file = config["exec_data_file"]
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
            "init_action_mask": self.init_action_mask,
            "interchange_mode": self.interchange_mode,
            "interchange_distribution": self.interchange_distribution,
            "use_exploration": self.use_exploration,
            "use_bindings": self.use_bindings,
            "use_vectorizer": self.use_vectorizer,
            "update_op_features": self.update_op_features,
            "reverse_history": self.reverse_history,
            "new_architecture": self.new_architecture,
            "normalize_bounds": self.normalize_bounds,
            "mul_log_p": self.mul_log_p,
            "force_vector": self.force_vector,
            "punish_vector": self.punish_vector,
            "activation": self.activation,
            "data_format": self.data_format,
            "optimization_mode": self.optimization_mode,
            "benchmarks_folder_path": self.benchmarks_folder_path,
            "bench_count": self.bench_count,
            "nb_iterations": self.nb_iterations,
            "value_epochs": self.value_epochs,
            "ppo_epochs": self.ppo_epochs,
            "ppo_batch_size": self.ppo_batch_size,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
            "lr": self.lr,
            "truncate": self.truncate,
            "json_file": self.json_file,
            "tags": self.tags,
            "logging": self.logging,
            "exec_data_file": self.exec_data_file
        }

    def __str__(self):
        """Convert the configuration to a string."""
        return str(self.to_dict())
