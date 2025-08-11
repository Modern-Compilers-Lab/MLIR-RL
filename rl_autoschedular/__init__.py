from utils.config import Config
import torch

device = torch.device("cpu")

# Load global configuration
config = Config()
if not config.loaded:
    config.load_from_json()
