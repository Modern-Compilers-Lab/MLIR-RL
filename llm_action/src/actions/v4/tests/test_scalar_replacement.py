from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code

from llm_action.src.actions.v4.implementation.tiling import Tiling
from llm_action.src.actions.v4.implementation.scalar_replacement import ScalarReplacement

# First tile to create loops, then apply LICM/CSE
tiling_params_per_kernel = {
    KernelType.MATMUL: {"tile_sizes": [32, 64, 16]},
    KernelType.CONV2D: {"tile_sizes": [16, 32, 0, 0, 0, 0, 0]},
    KernelType.GENERIC: {"tile_sizes": [4, 4, 0, 0, 0]},
}

if __name__ == "__main__":

    ACTION = ScalarReplacement

    for kernel_type in [KernelType.MATMUL, KernelType.CONV2D, KernelType.GENERIC]:
        print(f"--- Testing {ACTION.__name__} Action on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)

        # First tile to create loops (pre-step)
        tile_params = tiling_params_per_kernel[kernel_type]
        assert Tiling.precondition(code, tile_params), \
            f"Tiling precondition failed for {kernel_type.value}"
        code = Tiling.implement(code, tile_params)
        print("Pre-step: Tiled code to create loops for LICM/CSE")

        parameters = {}
        print(f"Using Parameters: {parameters}")

        assert ACTION.precondition(code, parameters), f"Precondition failed for {kernel_type.value}"
        print("Precondition: PASS")

        transformed_code = ACTION.implement(code, parameters)
        # ScalarReplacement postcondition allows identity transforms
        # (LICM/CSE may or may not change the code depending on the input)
        assert ACTION.postcondition(code, transformed_code, parameters), \
            f"Postcondition failed for {kernel_type.value}"
        print("Postcondition: PASS")

        # Should preserve func.func
        assert "func.func" in transformed_code, f"No func.func found for {kernel_type.value}"
        print("Structure check (func.func present): PASS")

        print(f"\n{ACTION.__name__} on {kernel_type.value}: ALL CHECKS PASSED")
        print("=" * 80 + "\n")

    # Test precondition rejects invalid inputs
    # Code without loops should be rejected
    raw_code = load_kernel_code(KernelType.MATMUL)
    assert not ACTION.precondition(raw_code, {}), \
        "Should reject code without loops"
    assert not ACTION.precondition("no tag here", {}), \
        "Should reject code without tag"
    print("Precondition rejection tests: ALL PASSED")
    print("\n=== ALL SCALAR_REPLACEMENT TESTS PASSED ===")
