from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code
from llm_action.src.execution.mlir_execution import execute_mlir

from llm_action.src.actions.v3.implementation.vectorization import Vectorization
from llm_action.src.actions.v3.implementation.tiling import Tiling

# Vectorization requires vector_sizes >= iteration space sizes.
# For large kernels, we must tile first to create small tiles that can be vectorized.
tile_params_per_kernel = {
    KernelType.MATMUL: {"tile_sizes": [4, 4, 4]},
    KernelType.CONV2D: {"tile_sizes": [1, 4, 0, 0, 0, 0, 0]},
    KernelType.GENERIC: {"tile_sizes": [8, 8, 16, 8, 4]},
}

vectorize_params_per_kernel = {
    KernelType.MATMUL: {"vector_sizes": [4, 4, 4]},
    KernelType.CONV2D: {"vector_sizes": [1, 4, 1, 1, 1, 1, 1]},
    KernelType.GENERIC: {"vector_sizes": [8, 8, 16, 8, 4]},
}

if __name__ == "__main__":

    ACTION = Vectorization

    for kernel_type in [KernelType.MATMUL, KernelType.CONV2D, KernelType.GENERIC]:
        print(f"--- Testing {ACTION.__name__} Action on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)

        # First tile to create manageable sizes
        tile_params = tile_params_per_kernel[kernel_type]
        if Tiling.precondition(code, tile_params):
            code = Tiling.implement(code, tile_params)
            print(f"After tiling with {tile_params['tile_sizes']}:\n")

        original_time_ns, success = execute_mlir(code)
        print(f"Tiled Execution; Success = {success}, Time = {original_time_ns} ns")

        parameters = vectorize_params_per_kernel[kernel_type]
        print(f"Using Vectorization Parameters: {parameters}\n")

        if ACTION.precondition(code, parameters):
            transformed_code = ACTION.implement(code, parameters)

            if transformed_code.strip() != code.strip():
                print(f"Transformed Code (first 500 chars):\n{transformed_code[:500]}\n...")
                transformed_time_ns, success = execute_mlir(transformed_code)
                print(f"Transformed Execution; Success = {success}, Time = {transformed_time_ns} ns")
                print(f"Speedup = {(original_time_ns / transformed_time_ns):.4f}")
            else:
                print("Transform returned original code (vectorization failed or not applicable).")

            if ACTION.postcondition(code, transformed_code, parameters):
                print(f"Postcondition satisfied: {ACTION.__name__} applied successfully.")
            else:
                print(f"Postcondition failed: {ACTION.__name__} not applied as expected.")
        else:
            print(f"Precondition not met; {ACTION.__name__} not applied.")
        print("=" * 80 + "\n")
