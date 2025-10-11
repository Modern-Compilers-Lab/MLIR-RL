from utils.config import Config
from utils.file_logger import  TensorBoardLogger
import torch

device = torch.device("cuda")

# Load global configuration
config = Config()
if not config.loaded:
    config.load_from_json()

# Pass run_name explicitly (from config or CLI)
file_logger = TensorBoardLogger(
    log_dir=config.results_dir,
    run_name=config.run_name,
    tags=['iql'] + config.tags
)

offline_data_collector = None

if config.collect_offline_data:
    from utils.data_collector import OfflineDataset
    offline_data_collector = OfflineDataset(
        save_dir=config.offline_data_save_dir,
        fname=config.offline_data_file
    )