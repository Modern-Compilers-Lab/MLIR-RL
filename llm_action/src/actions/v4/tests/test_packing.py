from llm_action.src.models import KernelType, InputType
from llm_action.src.utils.persistence import load_kernel_code

from llm_action.src.actions.v4.implementation.packing import Packing

# Packing requires tensor semantics
params_per_kernel = {
    KernelType.MATMUL: {
        "packed_sizes": [32, 64, 16],
    },
    KernelType.CONV2D: {
        "packed_sizes": [16, 32, 0, 0, 0, 0, 0],
    },
    KernelType.GENERIC: {
        "packed_sizes": [4, 4, 0, 0, 0],
    },
}

if __name__ == "__main__":

    ACTION = Packing

    for kernel_type in [KernelType.MATMUL, KernelType.CONV2D, KernelType.GENERIC]:
        print(f"--- Testing {ACTION.__name__} Action on {kernel_type.value} Kernel ---\n")
        # Packing requires tensor semantics
        code = load_kernel_code(kernel_type, input_type=InputType.TENSOR)

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

        # Packing should introduce pack operations
        has_pack = "linalg.pack" in transformed_code or "tensor.pack" in transformed_code
        assert has_pack, f"No pack ops found for {kernel_type.value}"
        print("Structure check (pack ops present): PASS")

        print(f"\n{ACTION.__name__} on {kernel_type.value}: ALL CHECKS PASSED")
        print("=" * 80 + "\n")

    # Test precondition rejects invalid inputs
    # Memref code should be rejected
    memref_code = load_kernel_code(KernelType.MATMUL, input_type=InputType.MEMREF)
    assert not ACTION.precondition(memref_code, {"packed_sizes": [32, 64, 16]}), \
        "Should reject memref code"
    print("Precondition rejects memref code: PASS")

    tensor_code = load_kernel_code(KernelType.MATMUL, input_type=InputType.TENSOR)
    assert not ACTION.precondition(tensor_code, {"packed_sizes": [0, 0, 0]}), \
        "Should reject all-zero packed sizes"
    assert not ACTION.precondition(tensor_code, {}), "Should reject empty params"
    assert not ACTION.precondition("no tag here", {"packed_sizes": [4]}), \
        "Should reject missing tag"
    print("Precondition rejection tests: ALL PASSED")
    print("\n=== ALL PACKING TESTS PASSED ===")
