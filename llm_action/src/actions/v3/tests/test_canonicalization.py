from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code
from llm_action.src.execution.mlir_execution import execute_mlir

from llm_action.src.actions.v3.implementation.canonicalization import Canonicalization
from llm_action.src.actions.v3.implementation.tiling import Tiling

params_per_kernel = {}

if __name__ == "__main__":

    ACTION = Canonicalization

    for kernel_type in [KernelType.MATMUL, KernelType.CONV2D, KernelType.GENERIC]:
        print(f"--- Testing {ACTION.__name__} Action on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)

        # First tile to create opportunities for canonicalization
        tile_params = {
            KernelType.MATMUL: {"tile_sizes": [64, 0, 0]},
            KernelType.CONV2D: {"tile_sizes": [16, 0, 0, 0, 0, 0, 0]},
            KernelType.GENERIC: {"tile_sizes": [4, 0, 0, 0, 0]},
        }
        if Tiling.precondition(code, tile_params[kernel_type]):
            code = Tiling.implement(code, tile_params[kernel_type])

        print(f"Input Code (after tiling):\n{code}\n")

        original_time_ns, success = execute_mlir(code)
        print(f"Original Execution; Success = {success}, Time = {original_time_ns} ns")

        parameters = {}
        print(f"Using Parameters: {parameters}\n")

        if ACTION.precondition(code, parameters):
            transformed_code = ACTION.implement(code, parameters)
            print(f"Transformed Code:\n{transformed_code}\n")

            if transformed_code.strip() != code.strip():
                transformed_time_ns, success = execute_mlir(transformed_code)
                print(f"Transformed Execution; Success = {success}, Time = {transformed_time_ns} ns")
                print(f"Speedup = {(original_time_ns / transformed_time_ns):.4f}")
            else:
                print("Transform returned original code (already canonical).")

            if ACTION.postcondition(code, transformed_code, parameters):
                print(f"Postcondition satisfied: {ACTION.__name__} applied successfully.")
            else:
                print(f"Postcondition failed: {ACTION.__name__} not applied as expected.")
        else:
            print(f"Precondition not met; {ACTION.__name__} not applied.")
        print("=" * 80 + "\n")
