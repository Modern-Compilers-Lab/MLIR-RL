from llm_action.src.models import KernelType
from llm_action.src.actions.v12.implementation.tiling import Tiling
from llm_action.src.actions.v12.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.test import test_action

# LoopUnrolling requires pre-tiled code (scf.for loops must exist).
# We first tile, then test unrolling on the tiled output.
from llm_action.src.utils.persistence import load_kernel_code

def test_loop_unrolling():
    kernel_type = KernelType.MATMUL
    code = load_kernel_code(kernel_type)
    print(f"--- Testing LoopUnrolling Action on {kernel_type.value} Kernel ---\n")

    # Step 1: Tile first to create scf.for loops
    tile_params = {"tile_sizes": [32, 32, 0]}
    assert Tiling.precondition(code, tile_params)
    tiled_code = Tiling.implement(code, tile_params)
    assert Tiling.postcondition(code, tiled_code, tile_params)
    print(f"Tiled code (prerequisite):\n{tiled_code}\n")

    # Step 2: Test unrolling on the tiled code
    unroll_params = {"unroll_factor": 4}
    assert LoopUnrolling.precondition(tiled_code, unroll_params), "Precondition failed"
    unrolled_code = LoopUnrolling.implement(tiled_code, unroll_params)
    post = LoopUnrolling.postcondition(tiled_code, unrolled_code, unroll_params)
    print(f"Postcondition: {post}")
    print(f"Unrolled code:\n{unrolled_code}\n")
    assert post, "Postcondition failed"
    print("LoopUnrolling test PASSED")

if __name__ == "__main__":
    test_loop_unrolling()
