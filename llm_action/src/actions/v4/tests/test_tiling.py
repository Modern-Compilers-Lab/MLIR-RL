from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code

from llm_action.src.actions.v4.implementation.tiling import Tiling

params_per_kernel = {
    KernelType.MATMUL: {
        "tile_sizes": [32, 64, 16],
    },
    KernelType.CONV2D: {
        "tile_sizes": [16, 32, 0, 0, 0, 0, 0],
    },
    KernelType.GENERIC: {
        "tile_sizes": [4, 4, 0, 0, 0],
    },
}

if __name__ == "__main__":

    ACTION = Tiling
    all_passed = True

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

        # Verify the transformed code has scf.for loops (tiling creates loops)
        assert "scf.for" in transformed_code, f"No scf.for loops found for {kernel_type.value}"
        print("Structure check (scf.for present): PASS")

        # Verify tag is preserved
        assert 'tag = "operation_0"' in transformed_code, f"Tag lost for {kernel_type.value}"
        print("Tag preserved: PASS")

        print(f"\n{ACTION.__name__} on {kernel_type.value}: ALL CHECKS PASSED")
        print("=" * 80 + "\n")

    # Test precondition rejects invalid inputs
    code = load_kernel_code(KernelType.MATMUL)
    assert not ACTION.precondition(code, {"tile_sizes": [0, 0, 0]}), "Should reject all-zero tile sizes"
    assert not ACTION.precondition(code, {"tile_sizes": [-1, 4]}), "Should reject negative tile sizes"
    assert not ACTION.precondition(code, {}), "Should reject missing tile_sizes"
    assert not ACTION.precondition("no tag here", {"tile_sizes": [4]}), "Should reject missing tag"
    print("Precondition rejection tests: ALL PASSED")
    print("\n=== ALL TILING TESTS PASSED ===")
