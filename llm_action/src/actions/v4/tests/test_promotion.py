from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code

from llm_action.src.actions.v4.implementation.promotion import Promotion

params_per_kernel = {
    KernelType.MATMUL: {
        "tile_sizes": [32, 64, 16],
        "operands_to_promote": [0, 1],
    },
    KernelType.CONV2D: {
        "tile_sizes": [16, 32, 0, 0, 0, 0, 0],
        "operands_to_promote": [0, 1],
    },
    KernelType.GENERIC: {
        "tile_sizes": [4, 4, 0, 0, 0],
        "operands_to_promote": [0],
    },
}

if __name__ == "__main__":

    ACTION = Promotion

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

        # Promotion tiles first, so scf.for should be present
        assert "scf.for" in transformed_code, f"No scf.for loops found for {kernel_type.value}"
        print("Structure check (scf.for present): PASS")

        # Promotion should introduce alloca or memref.alloc for promoted operands
        has_alloc = "memref.alloca" in transformed_code or "memref.alloc" in transformed_code
        assert has_alloc, f"No alloca/alloc found for {kernel_type.value}"
        print("Structure check (alloca/alloc present): PASS")

        print(f"\n{ACTION.__name__} on {kernel_type.value}: ALL CHECKS PASSED")
        print("=" * 80 + "\n")

    # Test precondition rejects invalid inputs
    code = load_kernel_code(KernelType.MATMUL)
    assert not ACTION.precondition(code, {"tile_sizes": [0, 0, 0], "operands_to_promote": [0, 1]}), \
        "Should reject all-zero tile sizes"
    assert not ACTION.precondition(code, {"tile_sizes": [32, 64, 16]}), \
        "Should reject missing operands_to_promote"
    assert not ACTION.precondition(code, {}), "Should reject empty params"
    assert not ACTION.precondition("no tag here", {"tile_sizes": [4], "operands_to_promote": [0]}), \
        "Should reject missing tag"
    print("Precondition rejection tests: ALL PASSED")
    print("\n=== ALL PROMOTION TESTS PASSED ===")
