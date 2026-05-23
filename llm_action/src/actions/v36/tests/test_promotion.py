from llm_action.src.actions.v36.implementation.tiling import Tiling
from llm_action.src.actions.v36.implementation.promotion import Promotion

from llm_action.src.actions.test import test_action
from llm_action.src.data.benchmarks import group_by_family, load_benchmark_set

import random


def test_promotion():
    """Test promotion on a pre-tiled instance (promotion requires tiling first)."""
    benchmark = "dataset_conv2d_img2col"
    split = "train"
    seed = 0

    instances = load_benchmark_set(benchmark, split=split)
    groups = group_by_family(instances)
    rng = random.Random(seed)

    family = "img2col_conv_2d_nchw_fchw"
    bucket = groups.get(family)
    if not bucket:
        print(f"[skip] family '{family}': no instances in '{benchmark}/{split}'")
        return

    chosen = rng.choice(bucket)
    print(f"--- Testing Promotion on family '{family}' (instance='{chosen.name}') ---\n")
    code = chosen.code

    # Step 1: tile first (promotion requires tiling)
    tile_params = {"tile_sizes": [0, 0, 0, 16]}
    if Tiling.precondition(code, tile_params):
        tiled_code = Tiling.implement(code, tile_params)
        if not Tiling.postcondition(code, tiled_code, tile_params):
            print("Tiling failed, cannot test promotion.")
            return
    else:
        print("Tiling precondition not met, cannot test promotion.")
        return

    print(f"Tiled code (pre-promotion):\n{tiled_code[:500]}...\n")

    # Step 2: promote
    promote_params = {"operands_to_promote": [0, 1, 2]}
    print(f"Using Parameters: {promote_params}\n")

    if Promotion.precondition(tiled_code, promote_params):
        promoted_code = Promotion.implement(tiled_code, promote_params)
        print(f"Promoted Code:\n{promoted_code[:800]}...\n")

        if Promotion.postcondition(tiled_code, promoted_code, promote_params):
            print("Postcondition satisfied: Promotion applied successfully.")
        else:
            print("Postcondition failed: Promotion not applied as expected.")
    else:
        print("Precondition not met; Promotion not applied.")
    print("=" * 80)


if __name__ == "__main__":
    test_promotion()
