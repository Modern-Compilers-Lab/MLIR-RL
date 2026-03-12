from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code
from llm_action.src.execution.mlir_execution import execute_mlir

from llm_action.playground.actions.candidates.Tiling_af31 import TilingAction
from llm_action.playground.actions.candidates.Tile import Tile
from llm_action.playground.actions.candidates.Tiling4 import Tiling

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
    
    # ACTION = TilingAction
    # ACTION = Tile
    ACTION = Tiling

    for kernel_type in [KernelType.MATMUL, KernelType.CONV2D, KernelType.GENERIC]:
        print(f"--- Testing {ACTION.__name__} on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)
        print(f"Original Code:\n{code}\n")
        
        
        
        print(f"Original Execution; Success = {success}, Time = {original_time_ns} ns")

        parameters = params_per_kernel[kernel_type]
        
        print(f"Using Parameters: {parameters}\n")
        
        if ACTION.precondition(code, parameters):
            transformed_code = ACTION.implement(code, parameters)
            
            print(f"Transformed Code:\n{transformed_code}\n")
            
            transformed_time_ns, success = execute_mlir(transformed_code)
            
            print(f"Transformed Execution; Success = {success}, Time = {transformed_time_ns} ns")
            
            print(f"Speedup = {(original_time_ns / transformed_time_ns):.4f}")
            
            if ACTION.postcondition(code, transformed_code, parameters):
                print(f"Postcondition satisfied: {ACTION.__name__} applied successfully.")
            else:
                print(f"Postcondition failed: {ACTION.__name__} not applied as expected.")
        else:
            print(f"Precondition not met; {ACTION.__name__} not applied.")
        print("=" * 80 + "\n")
