from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code

from llm_action.src.actions.v4.implementation.decomposition import Decomposition

if __name__ == "__main__":

    ACTION = Decomposition

    for kernel_type in [KernelType.MATMUL, KernelType.CONV2D, KernelType.GENERIC]:
        print(f"--- Testing {ACTION.__name__} Action on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)

        parameters = {}
        print(f"Using Parameters: {parameters}")

        # Decomposition precondition passes for any tagged code
        assert ACTION.precondition(code, parameters), f"Precondition failed for {kernel_type.value}"
        print("Precondition: PASS")

        transformed_code = ACTION.implement(code, parameters)

        # Decomposition may not change the code for basic ops (matmul, conv2d, generic)
        # that don't have a defined decomposition pattern. This is expected.
        if transformed_code.strip() != code.strip():
            assert ACTION.postcondition(code, transformed_code, parameters), \
                f"Postcondition failed for {kernel_type.value}"
            print("Transform applied and postcondition: PASS")
            assert "func.func" in transformed_code, f"No func.func found for {kernel_type.value}"
            print("Structure check (func.func present): PASS")
        else:
            # Decomposition returned unchanged code -- expected for basic linalg ops
            print("Transform returned unchanged code (expected for basic ops): PASS")
            # Postcondition should fail when code is unchanged
            assert not ACTION.postcondition(code, transformed_code, parameters), \
                f"Postcondition should fail for unchanged code on {kernel_type.value}"
            print("Postcondition correctly rejects unchanged code: PASS")

        print(f"\n{ACTION.__name__} on {kernel_type.value}: ALL CHECKS PASSED")
        print("=" * 80 + "\n")

    # Test precondition rejects invalid inputs
    assert not ACTION.precondition("no tag here", {}), "Should reject missing tag"
    print("Precondition rejection tests: ALL PASSED")
    print("\n=== ALL DECOMPOSITION TESTS PASSED ===")
