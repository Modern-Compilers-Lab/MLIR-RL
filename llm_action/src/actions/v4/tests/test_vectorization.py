from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code

from llm_action.src.actions.v4.implementation.vectorization import Vectorization
from llm_action.src.actions.v4.implementation.generalization import Generalization

params_per_kernel = {
    KernelType.MATMUL: {
        "vector_sizes": [4, 4, 16],  # total = 256 <= 256 (rank-3 limit)
    },
    KernelType.GENERIC: {
        "vector_sizes": [2, 2, 4, 2, 4],  # total = 128, all dims tiled
    },
}

if __name__ == "__main__":

    ACTION = Vectorization

    # Matmul and generic should vectorize successfully
    for kernel_type in [KernelType.MATMUL, KernelType.GENERIC]:
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

        # Vectorization should produce vector operations
        assert "vector" in transformed_code.lower(), f"No vector ops found for {kernel_type.value}"
        print("Structure check (vector ops present): PASS")

        print(f"\n{ACTION.__name__} on {kernel_type.value}: ALL CHECKS PASSED")
        print("=" * 80 + "\n")

    # Conv2d sliding-window patterns may not vectorize - test graceful handling
    print("--- Testing Vectorization graceful handling on conv2d ---\n")
    conv_code = load_kernel_code(KernelType.CONV2D)
    conv_params = {"vector_sizes": [1, 4, 1, 1, 4, 1, 1]}
    if Generalization.precondition(conv_code, {}):
        conv_code = Generalization.implement(conv_code, {})
    result = ACTION.implement(conv_code, conv_params)
    if result.strip() != conv_code.strip():
        print("Conv2d vectorized successfully (unexpected but OK)")
    else:
        print("Conv2d vectorization returned original code (expected for sliding-window)")
    print("Conv2d graceful handling: PASS")
    print("=" * 80 + "\n")

    # Test precondition rejects invalid inputs
    code = load_kernel_code(KernelType.MATMUL)
    assert not ACTION.precondition(code, {"vector_sizes": [0, 0, 0]}), \
        "Should reject zero vector sizes"
    assert not ACTION.precondition(code, {"vector_sizes": [-1, 4]}), \
        "Should reject negative vector sizes"
    assert not ACTION.precondition(code, {}), "Should reject empty params"
    assert not ACTION.precondition("no tag here", {"vector_sizes": [4]}), \
        "Should reject missing tag"
    # Reject excessively large vectors
    assert not ACTION.precondition(code, {"vector_sizes": [1024, 2]}), \
        "Should reject vector total > 1024"
    print("Precondition rejection tests: ALL PASSED")
    print("\n=== ALL VECTORIZATION TESTS PASSED ===")
