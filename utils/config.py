import os
from utils.singleton import Singleton
import json
from typing import Literal, Optional


class RASConfig(metaclass=Singleton):
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
    """Path to the benchmarks folder. Can be empty if data format is set to "json"."""
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
    tags: list[str]
    """List of tags to add to the neptune experiment"""
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
        self.tags = []
        self.logging = True
        self.loaded = False

    def load_from_json(self, path: Optional[str] = None):
        """Load the configuration from the JSON file.

        Args:
            path (Optional[str]): The path to the JSON file.
        """
        # Open the JSON file
        if path is None:
            path = os.getenv("RAS_CONFIG_FILE_PATH")
        with open(path, "r") as f:
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
        self.tags = config["tags"]
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
            "tags": self.tags,
            "logging": self.logging
        }

    def __str__(self):
        """Convert the configuration to a string."""
        return str(self.to_dict())


class AASConfig(metaclass=Singleton):
    """Class to store and load global configuration"""
    # Training and evaluation configuration ========================
    nb_iterations: int
    """Number of iterations for training the AlphaAutoScheduler agent"""
    data_queue_max_length: int
    """Maximum number of data points to store in the training queue."""
    epochs: int
    """Number of epochs for training the neural network."""
    batch_size: int
    """Batch size for training the neural network."""
    nb_train_eps: int
    """Number of training episodes per iteration."""
    nb_eval_eps: int
    """Number of evaluation episodes per iteration."""
    normalize_features: bool
    """Flag to normalize the features"""
    learning_rate: float
    """Learning rate"""
    l2_reg: float
    """L2 regularization coefficient"""
    quantile_alpha: float
    """Quantile alpha value for the quantile loss"""
    pitting: bool
    """Flag to enable pitting"""
    enable_hierarchical_space: bool
    """Flag to enable hierarchical space for actions"""
    # MCTS configuration ============================================
    mcts_max_nb_children: int
    """The maximum number of tile combinations to consider in MCTS. If -1, all combinations are considered."""
    mcts_nb_iterations: int
    """The number of MCTS iterations"""
    mcts_c_puct: float
    """The PUCT constant for MCTS"""
    mcts_hierarchical: bool
    """Flag to enable hierarchical MCTS space"""
    # MLIR code features configuration ==============================
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
    transformations: list[Literal["I", "TP", "V"]]
    """The list of transformations to consider"""
    vect_size_limit: int
    """Vectorization size limit to prevent large sizes vectorization"""
    parallelize_reduction: bool
    """Flag to enable parallelizing reduction operations"""
    # Data and execution configuration ==============================
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
    json_file: str
    """Path to the JSON file containing the benchmarks code or features."""
    eval_json_file: str
    """Path to the JSON file containing the evaluation benchmarks code or features."""
    split_ops: bool
    """Flag to enable splitting benchmarks that have more than one operation into multiple single operation benchmarks."""
    exec_db_path: str
    """Path to the execution database file."""
    use_cache: bool
    """Flag to enable using the execution database cache. If False, all codes are executed to get their execution times."""
    # Neptune configuration =========================================
    tags: list[str]
    """List of tags to add to the neptune experiment"""
    logging: bool
    """Flag to enable logging to neptune"""
    debug: bool
    """Flag to enable debug mode"""

    loaded: bool
    """Flag to check if the config was already loaded from JSON file or not"""

    def __init__(self):
        """Initialize the default values"""
        self.nb_iterations = 1000
        self.data_queue_max_length = 1024
        self.epochs = 4
        self.batch_size = 64
        self.nb_train_eps = 40
        self.nb_eval_eps = 10
        self.normalize_features = False
        self.learning_rate = 0.001
        self.l2_reg = 0.0001
        self.pitting = True
        self.mcts_max_nb_children = -1
        self.mcts_nb_iterations = 1000
        self.mcts_c_puct = 1.0
        self.mcts_hierarchical = False
        self.max_num_stores_loads = 7
        self.max_num_loops = 7
        self.max_num_load_store_dim = 7
        self.num_tile_sizes = 7
        self.num_transformations = 3
        self.transformations = ["I", "TP", "V"]
        self.vect_size_limit = 512
        self.parallelize_reduction = False
        self.use_bindings = False
        self.use_vectorizer = False
        self.data_format = "json"
        self.optimization_mode = "last"
        self.benchmarks_folder_path = ""
        self.exec_db_path = ""
        self.use_cache = True
        self.json_file = ""
        self.eval_json_file = ""
        self.split_ops = True
        self.tags = []
        self.logging = True
        self.loaded = False
        self.debug = False

    def load_from_json(self, path: Optional[str] = None):
        """Load the configuration from the JSON file.

        Args:
            path (Optional[str]): The path to the JSON file.
        """
        # Open the JSON file
        if path is None:
            path = os.getenv("AAS_CONFIG_FILE_PATH")
        with open(path, "r") as f:
            config = json.load(f)
        # Set the configuration values
        self.nb_iterations = config["nb_iterations"]
        self.data_queue_max_length = config["data_queue_max_length"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.nb_train_eps = config["nb_train_eps"]
        self.nb_eval_eps = config["nb_eval_eps"]
        self.normalize_features = config["normalize_features"]
        self.learning_rate = config["learning_rate"]
        self.l2_reg = config["l2_reg"]
        self.pitting = config["pitting"]
        self.mcts_max_nb_children = config["mcts_max_nb_children"]
        self.mcts_nb_iterations = config["mcts_nb_iterations"]
        self.mcts_c_puct = config["mcts_c_puct"]
        self.mcts_hierarchical = config["mcts_hierarchical"]
        self.max_num_stores_loads = config["max_num_stores_loads"]
        self.max_num_loops = config["max_num_loops"]
        self.max_num_load_store_dim = config["max_num_load_store_dim"]
        self.num_tile_sizes = config["num_tile_sizes"]
        self.num_transformations = config["num_transformations"]
        self.transformations = config["transformations"]
        self.vect_size_limit = config["vect_size_limit"]
        self.parallelize_reduction = config["parallelize_reduction"]
        self.use_bindings = config["use_bindings"]
        self.use_vectorizer = config["use_vectorizer"]
        self.data_format = config["data_format"]
        self.optimization_mode = config["optimization_mode"]
        self.benchmarks_folder_path = config["benchmarks_folder_path"]
        self.json_file = config["json_file"]
        self.eval_json_file = config["eval_json_file"]
        self.split_ops = config["split_ops"]
        self.exec_db_path = config["exec_db_path"]
        self.use_cache = config["use_cache"]
        self.tags = config["tags"]
        self.logging = config["logging"]
        self.debug = config["debug"]
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
            "nb_iterations": self.nb_iterations,
            "data_queue_max_length": self.data_queue_max_length,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "nb_train_eps": self.nb_train_eps,
            "nb_eval_eps": self.nb_eval_eps,
            "normalize_features": self.normalize_features,
            "learning_rate": self.learning_rate,
            "l2_reg": self.l2_reg,
            "pitting": self.pitting,
            "mcts_max_nb_children": self.mcts_max_nb_children,
            "mcts_nb_iterations": self.mcts_nb_iterations,
            "mcts_c_puct": self.mcts_c_puct,
            "mcts_hierarchical": self.mcts_hierarchical,
            "max_num_stores_loads": self.max_num_stores_loads,
            "max_num_loops": self.max_num_loops,
            "max_num_load_store_dim": self.max_num_load_store_dim,
            "num_tile_sizes": self.num_tile_sizes,
            "num_transformations": self.num_transformations,
            "transformations": self.transformations,
            "vect_size_limit": self.vect_size_limit,
            "parallelize_reduction": self.parallelize_reduction,
            "use_bindings": self.use_bindings,
            "use_vectorizer": self.use_vectorizer,
            "data_format": self.data_format,
            "optimization_mode": self.optimization_mode,
            "benchmarks_folder_path": self.benchmarks_folder_path,
            "json_file": self.json_file,
            "eval_json_file": self.eval_json_file,
            "split_ops": self.split_ops,
            "exec_db_path": self.exec_db_path,
            "use_cache": self.use_cache,
            "tags": self.tags,
            "logging": self.logging,
            "debug": self.debug
        }

    def __str__(self):
        """Convert the configuration to a string."""
        return str(self.to_dict())
