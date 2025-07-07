from utils.config import RASConfig


# Load global configuration
config = RASConfig()
if not config.loaded:
    config.load_from_json()
