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
    exploration: list[Literal['entropy', 'epsilon', 'curiosity']]
    """The exploration method"""
    init_epsilon: float
    """The initial epsilon value for epsilon greedy exploration"""
    reverse_history: bool
    """Flag to indicate if the history should be reversed or not"""
    new_architecture: bool
    """Flag to indicate if the new architecture should be used or not"""
    normalize_bounds: bool
    """Flag to indicate if the upper bounds in the input should be normalized or not"""
    normalize_adv: bool
    """Flag to indicate if the advantages should be normalized or not"""
    force_vector: bool
    """Flag to force vectorization"""
    sparse_reward: bool
    """Flag to enable sparse reward"""
    split_ops: bool
    """Flag to enable splitting operations into separate benchmarks"""
    activation: Literal["relu", "tanh"]
    """The activation function to use in the network"""
    benchmarks_folder_path: str
    """Path to the benchmarks folder. Can be empty if optimization mode is set to "last"."""
    bench_count: int
    """Number of batches in a trajectory"""
    nb_iterations: int
    """Number of iterations"""
    ppo_epochs: int
    """Number of epochs for PPO"""
    ppo_batch_size: int
    """Batch size for PPO"""
    value_epochs: int
    """Number of epochs for value update"""
    value_batch_size: int
    """Batch size for value update"""
    value_coef: float
    """Value coefficient"""
    value_alpha: float
    """Value alpha"""
    entropy_coef: float
    """Entropy coefficient"""
    reward_scale: float
    """Reward scale"""
    intrinsic_reward_integration: float
    """Intrinsic reward integration"""
    forward_weight: float
    """Forward weight"""
    curiosity_coef: float
    """Curiosity weight"""
    lr: float
    """Learning rate"""
    truncate: int
    """Maximum number of steps in the schedule"""
    json_file: str
    """Path to the JSON file containing the benchmarks execution times."""
    eval_json_file: str
    """Path to the JSON file containing the benchmarks execution times for evaluation."""
    tags: list[str]
    """List of tags to add to the neptune experiment"""
    debug: bool
    """Flag to enable debug mode"""
    exec_data_file: str
    """Path to the file containing the execution data"""
    results_dir: str
    """Path to the results directory"""

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
        self.init_action_mask = [False, True, False, False, False, False]
        self.exploration = ["entropy"]
        self.init_epsilon = 0.1
        self.interchange_mode = "enumerate"
        self.interchange_distribution = "binomial"
        self.reverse_history = True
        self.new_architecture = False
        self.normalize_bounds = True
        self.normalize_adv = True
        self.force_vector = True
        self.sparse_reward = True
        self.split_ops = False
        self.activation = "relu"
        self.benchmarks_folder_path = ""
        self.bench_count = 20
        self.nb_iterations = 10000
        self.ppo_epochs = 4
        self.ppo_batch_size = 4
        self.value_epochs = 32
        self.value_batch_size = 32
        self.value_coef = 0.5
        self.value_alpha = 0.0
        self.entropy_coef = 0.01
        self.reward_scale = 0.01
        self.intrinsic_reward_integration = 0.01
        self.forward_weight = 0.2
        self.curiosity_coef = 1
        self.lr = 0.001
        self.truncate = 5
        self.json_file = ""
        self.eval_json_file = ""
        self.tags = []
        self.debug = False
        self.exec_data_file = ""
        self.results_dir = "results"
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
        self.exploration = config["exploration"]
        self.init_epsilon = config["init_epsilon"]
        self.reverse_history = config["reverse_history"]
        self.new_architecture = config["new_architecture"]
        self.normalize_bounds = config["normalize_bounds"]
        self.normalize_adv = config["normalize_adv"]
        self.force_vector = config["force_vector"]
        self.sparse_reward = config["sparse_reward"]
        self.split_ops = config["split_ops"]
        self.activation = config["activation"]
        self.benchmarks_folder_path = config["benchmarks_folder_path"]
        self.bench_count = config["bench_count"]
        self.nb_iterations = config["nb_iterations"]
        self.ppo_epochs = config["ppo_epochs"]
        self.ppo_batch_size = config["ppo_batch_size"]
        self.value_epochs = config["value_epochs"]
        self.value_batch_size = config["value_batch_size"]
        self.value_coef = config["value_coef"]
        self.value_alpha = config["value_alpha"]
        self.entropy_coef = config["entropy_coef"]
        self.reward_scale = config["reward_scale"]
        self.intrinsic_reward_integration = config["intrinsic_reward_integration"]
        self.forward_weight = config["forward_weight"]
        self.curiosity_coef = config["curiosity_coef"]
        self.lr = config["lr"]
        self.truncate = config["truncate"]
        self.json_file = config["json_file"]
        self.eval_json_file = config["eval_json_file"]
        self.tags = config["tags"]
        self.debug = config["debug"]
        self.exec_data_file = config["exec_data_file"]
        self.results_dir = config["results_dir"]
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
            "exploration": self.exploration,
            "init_epsilon": self.init_epsilon,
            "reverse_history": self.reverse_history,
            "new_architecture": self.new_architecture,
            "normalize_bounds": self.normalize_bounds,
            "normalize_adv": self.normalize_adv,
            "force_vector": self.force_vector,
            "sparse_reward": self.sparse_reward,
            "split_ops": self.split_ops,
            "activation": self.activation,
            "benchmarks_folder_path": self.benchmarks_folder_path,
            "bench_count": self.bench_count,
            "nb_iterations": self.nb_iterations,
            "value_alpha": self.value_alpha,
            "ppo_epochs": self.ppo_epochs,
            "ppo_batch_size": self.ppo_batch_size,
            "value_epochs": self.value_epochs,
            "value_batch_size": self.value_batch_size,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
            "reward_scale": self.reward_scale,
            "intrinsic_reward_integration": self.intrinsic_reward_integration,
            "forward_weight": self.forward_weight,
            "curiosity_coef": self.curiosity_coef,
            "lr": self.lr,
            "truncate": self.truncate,
            "json_file": self.json_file,
            "eval_json_file": self.eval_json_file,
            "tags": self.tags,
            "debug": self.debug,
            "exec_data_file": self.exec_data_file,
            "results_dir": self.results_dir
        }

    def __str__(self):
        """Convert the configuration to a string."""
        return str(self.to_dict())
