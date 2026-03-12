from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code

from llm_action.src.actions.v4.implementation.tiling import Tiling
from llm_action.src.actions.v4.implementation.canonicalization import Canonicalization

# Canonicalization is more effective on IR that has been transformed
tiling_params_per_kernel = {
    KernelType.MATMUL: {"tile_sizes": [32, 64, 16]},
    KernelType.CONV2D: {"tile_sizes": [16, 32, 0, 0, 0, 0, 0]},
    KernelType.GENERIC: {"tile_sizes": [4, 4, 0, 0, 0]},
}

if __name__ == "__main__":

    ACTION = Canonicalization

    for kernel_type in [KernelType.MATMUL, KernelType.CONV2D, KernelType.GENERIC]:
        print(f"--- Testing {ACTION.__name__} Action on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)

        # First tile to make canonicalization meaningful
        tile_params = tiling_params_per_kernel[kernel_type]
        assert Tiling.precondition(code, tile_params), \
            f"Tiling precondition failed for {kernel_type.value}"
        code = Tiling.implement(code, tile_params)
        print("Pre-step: Tiled code to create IR for canonicalization")

        parameters = {}
        print(f"Using Parameters: {parameters}")

        assert ACTION.precondition(code, parameters), f"Precondition failed for {kernel_type.value}"
        print("Precondition: PASS")

        transformed_code = ACTION.implement(code, parameters)
        # Canonicalization postcondition allows identity transforms (cleanup action)
        assert ACTION.postcondition(code, transformed_code, parameters), \
            f"Postcondition failed for {kernel_type.value}"
        print("Postcondition: PASS")

        # Should preserve func.func
        assert "func.func" in transformed_code, f"No func.func found for {kernel_type.value}"
        print("Structure check (func.func present): PASS")

        print(f"\n{ACTION.__name__} on {kernel_type.value}: ALL CHECKS PASSED")
        print("=" * 80 + "\n")

    # Test precondition rejects invalid inputs
    assert not ACTION.precondition("no tag here, no func", {}), \
        "Should reject code without tag"
    assert not ACTION.precondition('tag = "operation_0" but no func', {}), \
        "Should reject code without func.func"
    print("Precondition rejection tests: ALL PASSED")
    print("\n=== ALL CANONICALIZATION TESTS PASSED ===")
