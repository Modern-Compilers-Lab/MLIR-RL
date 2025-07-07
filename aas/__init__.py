from utils.config import AASConfig


# Load global configuration
config = AASConfig()
if not config.loaded:
    config.load_from_json()
