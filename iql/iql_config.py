from dotenv import load_dotenv
load_dotenv()


import os
from utils.singleton import Singleton
import json
from typing import Literal




class Config(metaclass=Singleton):
    """Class to store and load global configuration"""

    ############## IQL specific parameters ##############
    gamma : float
    """Discount factor"""
    tau : float
    """expectile regression parameter"""
    inverse_temperature : float
    """Inverse temperature for advantage-weighted regression"""
    alpha : float
    """target smoothing coefficient"""
    batch_size : int
    """Batch size for training"""
    learning_rate : dict[str, float]
    """Learning rate for the optimizer"""
    max_steps : int
    """Maximum number of training steps"""
    target_update_freq : int
    """Frequency of target network updates"""
    sparse_reward : bool
    """Flag to enable sparse reward"""

    offline_data_directory : str
    """The offline data directory"""
    offline_data_file : str
    """The offline data file"""


    ############## Environment specific parameters ##############
    max_num_stores_loads: int
    """The maximum number of loads in the nested loops"""
    max_num_loops: int
    """The max number of nested loops"""
    max_num_load_store_dim: int
    """The max number of dimensions in load/store buffers"""
    num_tile_sizes: int
    """The number of tile sizes"""
    vect_size_limit: int
    """Vectorization size limit to prevent large sizes vectorization"""
    order: list[list[str]]
    """The order of actions that needs to bo followed"""
    interchange_mode: Literal['enumerate', 'pointers', 'continuous']
    """The method used for interchange action"""
    exploration: list[Literal['entropy', 'epsilon']]
    """The exploration method"""
    init_epsilon: float
    """The initial epsilon value for epsilon greedy exploration"""

    normalize_bounds: Literal['none', 'max', 'log']
    """Flag to indicate if the upper bounds in the input should be normalized or not"""
    
    
    split_ops: bool
    """Flag to enable splitting operations into separate benchmarks"""
    
    activation: Literal["relu", "tanh"]
    """The activation function to use in the network"""
    
    benchmarks_folder_path: str
    """Path to the benchmarks folder. Can be empty if optimization mode is set to "last"."""
    
    bench_count: int
    """Number of batches in a trajectory"""

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
        # IQL specific parameters
        self.gamma = 0.99
        self.tau = 0.7
        self.inverse_temperature = 3.0
        self.alpha = 0.005
        self.batch_size = 256
        self.learning_rate = {
            "value": 3e-4,
            "q": 3e-4,
            "policy": 3e-4
        }
        self.max_steps = 1000000
        self.target_update_freq = 1
        self.sparse_reward = True

        self.offline_data_directory = "./data"
        self.offline_data_file = "offline_data.npz"

        # Environment specific parameters
        self.max_num_stores_loads = 2
        self.max_num_loops = 4
        self.max_num_load_store_dim = 2
        self.num_tile_sizes = 2
        self.vect_size_limit = 16
        self.order = []
        self.interchange_mode = 'continuous'
        self.exploration = ['entropy', 'epsilon']
        self.init_epsilon = 1.0
        self.normalize_bounds = 'log'
        self.split_ops = False
        self.activation = "relu"
        self.benchmarks_folder_path = "./benchmarks"
        self.bench_count = 1
        self.truncate = 20
        self.json_file = "./config/exec_times.json"
        self.eval_json_file = "./config/exec_times.json"
        self.tags = []
        self.debug = False
        self.exec_data_file = "./data/exec_data.npz"
        self.results_dir = "./results"
        self.loaded = False

    def load_from_json(self):
        """Load the configuration from the JSON file."""
        # Open the JSON file
        with open(os.getenv("OFFLINE_RL_CONFIG_FILE_PATH"), "r") as f:
            config = json.load(f)
        # Set the configuration values
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        """Convert the configuration to a dictionary."""
        return self.__dict__

    def __str__(self):
        """Convert the configuration to a string."""
        return str(self.to_dict())