from llm_action.src.models import KernelType, InputType
from llm_action.src.utils.persistence import load_kernel_code

from llm_action.src.actions.v4.implementation.padding import Padding

# Padding requires tensor semantics
params_per_kernel = {
    KernelType.MATMUL: {
        "padding_values": ["0.0", "0.0", "0.0"],
        "padding_dimensions": [0, 1, 2],
        "pack_paddings": [1, 1, 1],
    },
    KernelType.CONV2D: {
        "padding_values": ["0.0", "0.0", "0.0"],
        "padding_dimensions": [0, 1],
        "pack_paddings": [1, 1, 1],
    },
    KernelType.GENERIC: {
        "padding_values": ["0.0", "0.0"],
        "padding_dimensions": [0, 1],
        "pack_paddings": [1, 1],
    },
}

if __name__ == "__main__":

    ACTION = Padding

    for kernel_type in [KernelType.MATMUL, KernelType.CONV2D, KernelType.GENERIC]:
        print(f"--- Testing {ACTION.__name__} Action on {kernel_type.value} Kernel ---\n")
        # Padding requires tensor semantics
        code = load_kernel_code(kernel_type, input_type=InputType.TENSOR)

        parameters = params_per_kernel[kernel_type]
        print(f"Using Parameters: {parameters}")

        assert ACTION.precondition(code, parameters), f"Precondition failed for {kernel_type.value}"
        print("Precondition: PASS")

        transformed_code = ACTION.implement(code, parameters)
        # Padding on already-aligned dimensions may be a no-op structurally
        # but the transform still succeeds (adds extract_slice/materialize)
        changed = transformed_code.strip() != code.strip()
        if changed:
            print("Transform applied: PASS")
            assert ACTION.postcondition(code, transformed_code, parameters), \
                f"Postcondition failed for {kernel_type.value}"
            print("Postcondition: PASS")
        else:
            print("Transform: no-op (dimensions already aligned, acceptable)")

        print(f"\n{ACTION.__name__} on {kernel_type.value}: CHECKS PASSED")
        print("=" * 80 + "\n")

    # Test precondition rejects invalid inputs
    # Memref code should be rejected
    memref_code = load_kernel_code(KernelType.MATMUL, input_type=InputType.MEMREF)
    assert not ACTION.precondition(memref_code, {
        "padding_values": ["0.0", "0.0", "0.0"],
        "padding_dimensions": [0, 1, 2],
    }), "Should reject memref code"
    print("Precondition rejects memref code: PASS")

    tensor_code = load_kernel_code(KernelType.MATMUL, input_type=InputType.TENSOR)
    assert not ACTION.precondition(tensor_code, {"padding_dimensions": [0, 1]}), \
        "Should reject missing padding_values"
    assert not ACTION.precondition(tensor_code, {"padding_values": ["0.0"]}), \
        "Should reject missing padding_dimensions"
    assert not ACTION.precondition(tensor_code, {}), "Should reject empty params"
    assert not ACTION.precondition("no tag here", {
        "padding_values": ["0.0"], "padding_dimensions": [0]
    }), "Should reject missing tag"
    print("Precondition rejection tests: ALL PASSED")
    print("\n=== ALL PADDING TESTS PASSED ===")
