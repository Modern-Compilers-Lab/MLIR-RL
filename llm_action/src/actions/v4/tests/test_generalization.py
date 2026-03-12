from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code

from llm_action.src.actions.v4.implementation.generalization import Generalization

if __name__ == "__main__":

    ACTION = Generalization

    # Generalization works on named ops (matmul, conv2d) but NOT on generic
    for kernel_type in [KernelType.MATMUL, KernelType.CONV2D]:
        print(f"--- Testing {ACTION.__name__} Action on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)

        parameters = {}
        print(f"Using Parameters: {parameters}")

        assert ACTION.precondition(code, parameters), f"Precondition failed for {kernel_type.value}"
        print("Precondition: PASS")

        transformed_code = ACTION.implement(code, parameters)
        assert transformed_code.strip() != code.strip(), f"Transform produced no-op for {kernel_type.value}"
        print("Transform applied: PASS")

        assert ACTION.postcondition(code, transformed_code, parameters), \
            f"Postcondition failed for {kernel_type.value}"
        print("Postcondition: PASS")

        # After generalization, should contain linalg.generic
        assert "linalg.generic" in transformed_code, f"No linalg.generic found for {kernel_type.value}"
        print("Structure check (linalg.generic present): PASS")

        print(f"\n{ACTION.__name__} on {kernel_type.value}: ALL CHECKS PASSED")
        print("=" * 80 + "\n")

    # Generic kernel should be rejected by precondition (already generic)
    print("--- Testing {ACTION.__name__} Action on generic Kernel (expected rejection) ---\n")
    generic_code = load_kernel_code(KernelType.GENERIC)
    assert not ACTION.precondition(generic_code, {}), \
        "Should reject generic kernel (no named linalg op)"
    print("Precondition correctly rejects generic kernel: PASS")

    # Test precondition rejects invalid inputs
    assert not ACTION.precondition("no tag here", {}), "Should reject missing tag"
    print("Precondition rejection tests: ALL PASSED")
    print("\n=== ALL GENERALIZATION TESTS PASSED ===")
