from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code
from llm_action.src.execution.mlir_execution import execute_mlir

from llm_action.src.actions.v3.implementation.loop_fusion import LoopFusion

# Loop fusion requires at least two linalg operations in the IR.
# Since our standard kernels have a single linalg op, the precondition
# will reject them. This test demonstrates the precondition check.
params_per_kernel = {
    KernelType.MATMUL: {
        "tile_sizes": [64, 64, 0],
    },
    KernelType.CONV2D: {
        "tile_sizes": [16, 32, 0, 0, 0, 0, 0],
    },
    KernelType.GENERIC: {
        "tile_sizes": [4, 4, 0, 0, 0],
    },
}

if __name__ == "__main__":

    ACTION = LoopFusion

    for kernel_type in [KernelType.MATMUL, KernelType.CONV2D, KernelType.GENERIC]:
        print(f"--- Testing {ACTION.__name__} Action on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)
        print(f"Original Code:\n{code}\n")

        parameters = params_per_kernel[kernel_type]
        print(f"Using Parameters: {parameters}\n")

        if ACTION.precondition(code, parameters):
            transformed_code = ACTION.implement(code, parameters)
            print(f"Transformed Code:\n{transformed_code}\n")

            if ACTION.postcondition(code, transformed_code, parameters):
                print(f"Postcondition satisfied: {ACTION.__name__} applied successfully.")
            else:
                print(f"Postcondition failed: {ACTION.__name__} not applied as expected.")
        else:
            print(f"Precondition not met; {ACTION.__name__} not applied (requires >= 2 linalg ops).")
        print("=" * 80 + "\n")
