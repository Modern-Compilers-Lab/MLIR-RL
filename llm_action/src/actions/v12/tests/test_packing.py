from llm_action.src.models import KernelType
from llm_action.src.actions.v12.implementation.packing import Packing
from llm_action.src.utils.persistence import load_kernel_code

def test_packing():
    kernel_type = KernelType.MATMUL
    code = load_kernel_code(kernel_type)
    print(f"--- Testing Packing Action on {kernel_type.value} Kernel ---\n")
    print(f"Original Code:\n{code}\n")

    params = {"packed_sizes": [32, 32, 32]}
    print(f"Using Parameters: {params}\n")

    assert Packing.precondition(code, params), "Precondition failed"
    transformed_code = Packing.implement(code, params)
    print(f"Transformed Code:\n{transformed_code}\n")

    post = Packing.postcondition(code, transformed_code, params)
    print(f"Postcondition: {post}")

    assert post, "Postcondition failed"
    assert "linalg.pack" in transformed_code or "linalg.generic" in transformed_code, \
        "Expected linalg.pack or linalg.generic in output"
    assert 'tag = "operation_0"' in transformed_code, "Tag not preserved"

    # Note: Packed code requires additional lowering (linalg.pack -> lower-level ops)
    # before it can be executed through the default pipeline. Execution is tested
    # via the MCP execute_mlir_code tool which has the full lowering pipeline.
    print("Packing test PASSED (transform verified, execution requires extended pipeline)")

if __name__ == "__main__":
    test_packing()
