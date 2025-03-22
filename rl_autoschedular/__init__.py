from utils.config import Config
from utils.file_logger import FileLogger


# Load global configuration
config = Config()
if not config.loaded:
    config.load_from_json()

file_logger = FileLogger(config.results_dir, ['ppo'] + config.tags)
