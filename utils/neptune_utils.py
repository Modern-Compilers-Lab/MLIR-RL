import os
import neptune
from typing import Literal


def init_neptune(tags: list, mode: Literal["async", "sync", "offline", "read-only", "debug"] = "async"):
    # Get tags
    tags = list(map(str, tags))
    # Initialize neptune run
    run = neptune.init_run(
        project=os.getenv('NEPTUNE_PROJECT'),
        api_token=os.getenv('NEPTUNE_TOKEN'),
        tags=tags,
        mode=mode
    )
    # Upload source code
    run["src"].upload_files([
        './rl_autoschedular',
        './utils',
        './train.py'
    ])
    # Upload config
    run["config"].upload(os.getenv('CONFIG_FILE_PATH'))
    return run
