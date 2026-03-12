from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code

from llm_action.src.actions.v4.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v4.implementation.generalization import Generalization

# Interchange requires linalg.generic -- generalize first for named ops
params_per_kernel = {
    KernelType.MATMUL: {
        "permutation": [1, 2, 0],  # 3 dims: M, N, K
    },
    KernelType.CONV2D: {
        "permutation": [1, 0, 2, 3, 4, 5, 6],  # 7 dims for conv2d
    },
    KernelType.GENERIC: {
        "permutation": [1, 0, 2, 3, 4],  # 5 dims
    },
}

if __name__ == "__main__":

    ACTION = LoopInterchange

    for kernel_type in [KernelType.MATMUL, KernelType.CONV2D, KernelType.GENERIC]:
        print(f"--- Testing {ACTION.__name__} Action on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)

        # Generalize named ops first (matmul, conv2d) to get linalg.generic
        if kernel_type != KernelType.GENERIC:
            assert Generalization.precondition(code, {}), \
                f"Generalization precondition failed for {kernel_type.value}"
            code = Generalization.implement(code, {})
            print("Pre-step: Generalized code to linalg.generic")

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

        # After interchange, should still have linalg.generic
        assert "linalg.generic" in transformed_code, f"No linalg.generic found for {kernel_type.value}"
        print("Structure check (linalg.generic present): PASS")

        print(f"\n{ACTION.__name__} on {kernel_type.value}: ALL CHECKS PASSED")
        print("=" * 80 + "\n")

    # Test precondition rejects invalid inputs
    code = load_kernel_code(KernelType.MATMUL)
    # Named op (not generic) should be rejected
    assert not ACTION.precondition(code, {"permutation": [1, 2, 0]}), \
        "Should reject non-generic code"
    # Identity permutation should be rejected
    generic_code = Generalization.implement(code, {})
    assert not ACTION.precondition(generic_code, {"permutation": [0, 1, 2]}), \
        "Should reject identity permutation"
    assert not ACTION.precondition(generic_code, {"permutation": [1, 1, 0]}), \
        "Should reject invalid permutation (duplicates)"
    assert not ACTION.precondition(generic_code, {}), "Should reject empty params"
    assert not ACTION.precondition("no tag here", {"permutation": [1, 0]}), \
        "Should reject missing tag"
    print("Precondition rejection tests: ALL PASSED")
    print("\n=== ALL LOOP_INTERCHANGE TESTS PASSED ===")
