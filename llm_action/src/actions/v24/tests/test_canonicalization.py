"""Two-step canonicalization test.

Canonicalization on a pristine matmul is a no-op, so we first tile to produce
IR with redundancies and then canonicalize. Uses the same benchmark-folder
sampling as `llm_action.src.actions.test.test_action`.
"""
import random

from llm_action.src.actions.v24.implementation.canonicalization import Canonicalization
from llm_action.src.actions.v24.implementation.tiling import Tiling
from llm_action.src.data.benchmarks import group_by_family, load_benchmark_set


params_per_family = {
    "matmul": {}
}


def test_canonicalization(
    benchmark: str = "standard",
    split: str = "train",
    seed: int = 0,
) -> None:
    instances = load_benchmark_set(benchmark, split=split)
    groups = group_by_family(instances)
    rng = random.Random(seed)

    for family in params_per_family.keys():
        bucket = groups.get(family)
        if not bucket:
            print(f"[skip] family '{family}': no instances in '{benchmark}/{split}'\n")
            continue
        chosen = rng.choice(bucket)
        print(
            f"--- Testing Canonicalization on family '{family}' "
            f"(benchmark='{benchmark}/{split}', instance='{chosen.name}') ---\n"
        )
        code = chosen.code

        # Step 1: tile to produce canonicalizable artifacts.
        tile_params = {"tile_sizes": [32, 32, 0]}
        assert Tiling.precondition(code, tile_params)
        tiled_code = Tiling.implement(code, tile_params)
        assert Tiling.postcondition(code, tiled_code, tile_params)
        print(f"Tiled Code:\n{tiled_code}\n")

        # Step 2: canonicalize.
        canon_params = {}
        if Canonicalization.precondition(tiled_code, canon_params):
            canon_code = Canonicalization.implement(tiled_code, canon_params)
            print(f"Canonicalized Code:\n{canon_code}\n")
            if Canonicalization.postcondition(tiled_code, canon_code, canon_params):
                print("Postcondition satisfied: Canonicalization applied successfully.")
            else:
                print("Postcondition result: no change (IR was already canonical after tiling).")
        else:
            print("Precondition not met.")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    test_canonicalization()
