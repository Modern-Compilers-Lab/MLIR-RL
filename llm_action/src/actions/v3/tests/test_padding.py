from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code
from llm_action.src.execution.mlir_execution import execute_mlir

from llm_action.src.actions.v3.implementation.padding import Padding

params_per_kernel = {
    KernelType.MATMUL: {
        "padding_values": [0.0, 0.0, 0.0],
        "padding_dimensions": [0, 1, 2],
        "copy_back_op": "none",
    },
    KernelType.CONV2D: {
        "padding_values": [0.0, 0.0, 0.0],
        "padding_dimensions": [0, 1],
        "copy_back_op": "none",
    },
    KernelType.GENERIC: {
        "padding_values": [0.0, 0.0],
        "padding_dimensions": [0, 1, 2, 3, 4],
        "copy_back_op": "none",
    },
}

if __name__ == "__main__":

    ACTION = Padding

    for kernel_type in [KernelType.MATMUL, KernelType.CONV2D, KernelType.GENERIC]:
        print(f"--- Testing {ACTION.__name__} Action on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)
        print(f"Original Code:\n{code}\n")

        original_time_ns, success = execute_mlir(code)
        print(f"Original Execution; Success = {success}, Time = {original_time_ns} ns")

        parameters = params_per_kernel[kernel_type]
        print(f"Using Parameters: {parameters}\n")

        if ACTION.precondition(code, parameters):
            transformed_code = ACTION.implement(code, parameters)
            print(f"Transformed Code:\n{transformed_code}\n")

            if transformed_code.strip() != code.strip():
                transformed_time_ns, success = execute_mlir(transformed_code)
                print(f"Transformed Execution; Success = {success}, Time = {transformed_time_ns} ns")
                print(f"Speedup = {(original_time_ns / transformed_time_ns):.4f}")
            else:
                print("Transform returned original code (no-op).")

            if ACTION.postcondition(code, transformed_code, parameters):
                print(f"Postcondition satisfied: {ACTION.__name__} applied successfully.")
            else:
                print(f"Postcondition failed: {ACTION.__name__} not applied as expected.")
        else:
            print(f"Precondition not met; {ACTION.__name__} not applied.")
        print("=" * 80 + "\n")
