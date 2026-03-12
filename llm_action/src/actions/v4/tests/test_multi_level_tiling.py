from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code

from llm_action.src.actions.v4.implementation.multi_level_tiling import MultiLevelTiling

params_per_kernel = {
    KernelType.MATMUL: {
        "outer_tile_sizes": [64, 128, 64],
        "inner_tile_sizes": [16, 32, 8],
    },
    KernelType.CONV2D: {
        "outer_tile_sizes": [32, 64, 0, 0, 0, 0, 0],
        "inner_tile_sizes": [8, 16, 0, 0, 0, 0, 0],
    },
    KernelType.GENERIC: {
        "outer_tile_sizes": [4, 4, 0, 0, 0],
        "inner_tile_sizes": [2, 2, 0, 0, 0],
    },
}

if __name__ == "__main__":

    ACTION = MultiLevelTiling

    for kernel_type in [KernelType.MATMUL, KernelType.CONV2D, KernelType.GENERIC]:
        print(f"--- Testing {ACTION.__name__} Action on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)

        parameters = params_per_kernel[kernel_type]
        print(f"Using Parameters: {parameters}")

        assert ACTION.precondition(code, parameters), f"Precondition failed for {kernel_type.value}"
        print("Precondition: PASS")

        transformed_code = ACTION.implement(code, parameters)
        assert transformed_code.strip() != code.strip(), f"Transform produced no-op for {kernel_type.value}"
        print("Transform applied: PASS")

        assert ACTION.postcondition(code, transformed_code, parameters), \
            f"Postcondition failed for {kernel_type.value}"
        print("Postcondition: PASS")

        # Multi-level tiling should produce nested scf.for loops
        assert "scf.for" in transformed_code, f"No scf.for loops found for {kernel_type.value}"
        print("Structure check (scf.for present): PASS")

        print(f"\n{ACTION.__name__} on {kernel_type.value}: ALL CHECKS PASSED")
        print("=" * 80 + "\n")

    # Test precondition rejects invalid inputs
    code = load_kernel_code(KernelType.MATMUL)
    assert not ACTION.precondition(code, {"outer_tile_sizes": [0, 0, 0], "inner_tile_sizes": [16, 32, 8]}), \
        "Should reject all-zero outer tile sizes"
    assert not ACTION.precondition(code, {"outer_tile_sizes": [64, 128, 64], "inner_tile_sizes": [0, 0, 0]}), \
        "Should reject all-zero inner tile sizes"
    assert not ACTION.precondition(code, {"outer_tile_sizes": [64, 128, 64]}), \
        "Should reject missing inner_tile_sizes"
    assert not ACTION.precondition(code, {}), "Should reject empty params"
    assert not ACTION.precondition("no tag here", {"outer_tile_sizes": [4], "inner_tile_sizes": [2]}), \
        "Should reject missing tag"
    print("Precondition rejection tests: ALL PASSED")
    print("\n=== ALL MULTI_LEVEL_TILING TESTS PASSED ===")
