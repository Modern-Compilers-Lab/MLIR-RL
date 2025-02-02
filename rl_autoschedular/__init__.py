from utils.config import Config


# Load global configuration
config = Config()
if not config.loaded:
    config.load_from_json()
