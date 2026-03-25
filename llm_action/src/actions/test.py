from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code
from llm_action.src.execution.mlir_execution import execute_mlir

from llm_action.src.actions.base import ActionBase

def test_action(action: ActionBase, params_per_kernel: dict[KernelType, dict]):
    for kernel_type in params_per_kernel.keys():
        print(f"--- Testing {action.__name__} Action on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)
        print(f"Original Code:\n{code}\n")
        
        original_time_ns, success = execute_mlir(code)
        print(f"Original Execution; Success = {success}, Time = {original_time_ns} ns")

        parameters = params_per_kernel[kernel_type]
        print(f"Using Parameters: {parameters}\n")

        if action.precondition(code, parameters):
            transformed_code = action.implement(code, parameters)
            print(f"Transformed Code:\n{transformed_code}\n")
            
            transformed_time_ns, success = execute_mlir(transformed_code)
            print(f"Transformed Execution; Success = {success}, Time = {transformed_time_ns} ns")
            print(f"Speedup = {(original_time_ns / transformed_time_ns):.4f}")
        
            if action.postcondition(code, transformed_code, parameters):
                print(f"Postcondition satisfied: {action.__name__} applied successfully.")
            else:
                print(f"Postcondition failed: {action.__name__} not applied as expected.")
        else:
            print(f"Precondition not met; {action.__name__} not applied.")
        print("=" * 80 + "\n")
