from typing import Literal, Any, Optional
from typeguard import check_type, CollectionCheckStrategy
from .singleton import Singleton
import json
import os


class Config(metaclass=Singleton):
    """Class to store and load global configuration"""

    intermediate_transforms: bool = True
    """A flag indicating if transformations should be applied at each step."""
    max_num_stores_loads: int = 7
    """The maximum number of loads in the nested loops"""
    max_num_loops: int = 12
    """The max number of nested loops"""
    max_num_load_store_dim: int = 12
    """The max number of dimensions in load/store buffers"""
    num_tile_sizes: int = 7
    """The number of tile sizes"""
    vect_size_limit: int = 512
    """Vectorization size limit to prevent large sizes vectorization"""
    order: list[list[str]] = []
    """The order of actions that needs to bo followed"""
    interchange_mode: Literal['enumerate', 'pointers', 'continuous']
    """The method used for interchange action"""
    exploration: list[Literal['entropy', 'epsilon']] = ['entropy']
    """The exploration method"""
    init_epsilon: float = 0.5
    """The initial epsilon value for epsilon greedy exploration"""
    normalize_bounds: Literal['none', 'max', 'log'] = 'max'
    """Flag to indicate if the upper bounds in the input should be normalized or not"""
    normalize_adv: Literal['none', 'standard', 'max-abs'] = 'standard'
    """The advantage normalization method"""
    reuse_experience: Literal['none', 'random', 'topk'] = 'none'
    """Strategy for experience replay"""
    benchmarks_folder_path: str
    """Path to the benchmarks folder. Can be empty if optimization mode is set to "last"."""
    bench_count: int = 64
    """Number of batches in a trajectory"""
    replay_count: int = 10
    """Number of trajectories to keep in the replay buffer"""
    nb_iterations: int
    """Number of iterations"""
    ppo_epochs: int
    """Number of epochs for PPO"""
    ppo_batch_size: Optional[int]
    """Batch size for PPO"""
    value_epochs: int = 0
    """Number of epochs for value update"""
    value_batch_size: Optional[int] = 32
    """Batch size for value update"""
    value_coef: float = 0.5
    """Value coefficient"""
    value_clip: bool = False
    """Clip value loss or not"""
    entropy_coef: float = 0.01
    """Entropy coefficient"""
    lr: float = 1e-3
    """Learning rate"""
    truncate: int = 5
    """Maximum number of steps in the schedule"""
    json_file: str
    """Path to the JSON file containing the benchmarks execution times."""
    eval_json_file: str
    """Path to the JSON file containing the benchmarks execution times for evaluation."""
    tags: list[str] = []
    """List of tags to add to the neptune experiment"""
    debug: bool = False
    """Flag to enable debug mode"""
    main_exec_data_file: str = ''
    """Path to the file containing the execution data"""
    results_dir: str
    """Path to the results directory"""

    def __init__(self):
        """Load the configuration from the JSON file
        or get existing instance if any.
        """
        # Open the JSON file
        with open(os.getenv("CONFIG_FILE_PATH"), "r") as f:
            config_data: dict[str, Any] = json.load(f)

        for element, element_t in self.__annotations__.items():
            if element in config_data:
                element_raw_v = config_data[element]
            elif element in self.__class__.__dict__:
                element_raw_v = self.__class__.__dict__[element]
            else:
                raise ValueError(f"{element} is required but missing from the config file")
            element_v = check_type(element_raw_v, element_t, collection_check_strategy=CollectionCheckStrategy.ALL_ITEMS)
            setattr(self, element, element_v)

    def to_dict(self):
        """Convert the configuration to a dictionary."""
        return {k: self.__dict__[k] for k in self.__annotations__}

    def print_config(self):
        """Print the configs in a readable format"""
        print("Configuration:")
        for k in self.__annotations__:
            print(f"- {k}: {self.__dict__[k]}")
