from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code

from llm_action.playground.actions.candidates.Tiling_af31 import TilingAction

params_per_kernel = {
    KernelType.MATMUL: {
        "tile_sizes": [8, 8],
    },
    KernelType.CONV2D: {
        "tile_sizes": [8, 8, 4, 4],
    },
    KernelType.GENERIC: {
        "tile_sizes": [4, 4],
    },
}

if __name__ == "__main__":
    
    ACTION = TilingAction

    for kernel_type in [KernelType.MATMUL, KernelType.CONV2D, KernelType.GENERIC]:
        print(f"--- Testing Tiling Action on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)
        print(f"Original Code:\n{code}\n")

        parameters = params_per_kernel[kernel_type]
        
        print(f"Using Parameters: {parameters}\n")

        if ACTION.precondition(code, parameters):
            transformed_code = ACTION.implement(code, parameters)
            print(f"Transformed Code:\n{transformed_code}\n")
            if ACTION.postcondition(code, transformed_code, parameters):
                print("Postcondition satisfied: Tiling applied successfully.")
            else:
                print("Postcondition failed: Tiling not applied as expected.")
        else:
            print("Precondition not met; tiling not applied.")
        print("=" * 80 + "\n")
