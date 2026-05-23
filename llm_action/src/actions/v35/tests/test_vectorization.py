from llm_action.src.actions.v35.implementation.vectorization import Vectorization

from llm_action.src.actions.test import test_action

# Instance: add_240_224_15_150 (dims 240x224x15x150)
# 240 % 4 == 0, 224 % 4 == 0, 15 % 1 == 0, 150 % 2 == 0
params_per_family = {
    "add": {
        "tile_sizes": [4, 4, 1, 2]
    }
}

if __name__ == "__main__":
    test_action(Vectorization, params_per_family, benchmark="dataset_add")
