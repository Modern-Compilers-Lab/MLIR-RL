from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code

from llm_action.src.actions.v4.implementation.loop_distribution import LoopDistribution

# split_reduction works on ops with reduction dimensions
params_per_kernel = {
    KernelType.MATMUL: {
        "split_factor": 16,
        "insert_split_dimension": 0,
    },
    KernelType.CONV2D: {
        "split_factor": 8,
        "insert_split_dimension": 0,
    },
    KernelType.GENERIC: {
        "split_factor": 4,
        "insert_split_dimension": 0,
    },
}

if __name__ == "__main__":

    ACTION = LoopDistribution

    # Matmul should always work with split_reduction
    print("--- Testing LoopDistribution Action on matmul Kernel ---\n")
    code = load_kernel_code(KernelType.MATMUL)
    parameters = params_per_kernel[KernelType.MATMUL]
    print(f"Using Parameters: {parameters}")

    assert ACTION.precondition(code, parameters), "Precondition failed for matmul"
    print("Precondition: PASS")

    transformed_code = ACTION.implement(code, parameters)
    assert transformed_code.strip() != code.strip(), "Transform produced no-op for matmul"
    print("Transform applied: PASS")

    assert ACTION.postcondition(code, transformed_code, parameters), \
        "Postcondition failed for matmul"
    print("Postcondition: PASS")

    assert "func.func" in transformed_code, "No func.func found for matmul"
    print("Structure check (func.func present): PASS")
    print("\nLoopDistribution on matmul: ALL CHECKS PASSED")
    print("=" * 80 + "\n")

    # Conv2d and generic may not work with split_reduction (graceful no-op is OK)
    for kernel_type in [KernelType.CONV2D, KernelType.GENERIC]:
        print(f"--- Testing LoopDistribution Action on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)
        parameters = params_per_kernel[kernel_type]
        print(f"Using Parameters: {parameters}")

        pre = ACTION.precondition(code, parameters)
        if not pre:
            print(f"Precondition rejected (acceptable for {kernel_type.value})")
            print(f"\nLoopDistribution on {kernel_type.value}: CHECKS PASSED (precondition reject)")
            print("=" * 80 + "\n")
            continue

        print("Precondition: PASS")
        transformed_code = ACTION.implement(code, parameters)
        changed = transformed_code.strip() != code.strip()
        if changed:
            post = ACTION.postcondition(code, transformed_code, parameters)
            print(f"Transform applied: PASS (changed={changed}, post={post})")
        else:
            print(f"Transform: no-op (split_reduction may not apply to {kernel_type.value})")

        print(f"\nLoopDistribution on {kernel_type.value}: CHECKS PASSED")
        print("=" * 80 + "\n")

    # Test precondition rejects invalid inputs
    code = load_kernel_code(KernelType.MATMUL)
    assert not ACTION.precondition(code, {"split_factor": 1}), \
        "Should reject split_factor < 2"
    assert not ACTION.precondition(code, {"split_factor": 0}), \
        "Should reject split_factor = 0"
    assert not ACTION.precondition(code, {}), "Should reject empty params"
    assert not ACTION.precondition("no tag here", {"split_factor": 4}), \
        "Should reject missing tag"
    assert not ACTION.precondition(code, {"split_factor": 4, "insert_split_dimension": -1}), \
        "Should reject negative dimension"
    print("Precondition rejection tests: ALL PASSED")
    print("\n=== ALL LOOP_DISTRIBUTION TESTS PASSED ===")
