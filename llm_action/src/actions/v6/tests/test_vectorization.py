from llm_action.src.models import KernelType
from llm_action.src.actions.v6.implementation.tiling import Tiling
from llm_action.src.actions.v6.implementation.vectorization import Vectorization

from llm_action.src.utils.persistence import load_kernel_code
from llm_action.src.execution.mlir_execution import execute_mlir

# Vectorization typically works on tiled (small) ops.
# We first tile, then vectorize the tiled result.
# Note: conv2d_nchw_fchw is not supported by the MLIR transform vectorizer,
# so we only test matmul and generic kernels.

tile_and_vectorize_params = {
    KernelType.MATMUL: {
        "tile_sizes": [4, 4, 4],
        "vector_sizes": [4, 4, 4],
    },
    KernelType.GENERIC: {
        "tile_sizes": [2, 2, 4, 2, 4],
        "vector_sizes": [2, 2, 4, 2, 4],
    },
}

def test_vectorization():
    for kernel_type, params in tile_and_vectorize_params.items():
        print(f"--- Testing Vectorization Action on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)
        print(f"Original Code:\n{code}\n")

        original_time_ns, success = execute_mlir(code)
        print(f"Original Execution; Success = {success}, Time = {original_time_ns} ns")

        tile_params = {"tile_sizes": params["tile_sizes"]}
        vec_params = {"vector_sizes": params["vector_sizes"]}

        # Step 1: Tile
        assert Tiling.precondition(code, tile_params), "Tiling precondition failed"
        tiled_code = Tiling.implement(code, tile_params)
        assert Tiling.postcondition(code, tiled_code, tile_params), "Tiling postcondition failed"
        print(f"After tiling with {tile_params['tile_sizes']}: OK")

        # Step 2: Vectorize
        print(f"Using Vectorization Parameters: {vec_params}\n")
        if Vectorization.precondition(tiled_code, vec_params):
            vectorized_code = Vectorization.implement(tiled_code, vec_params)
            print(f"Vectorized Code:\n{vectorized_code}\n")

            vec_time_ns, success = execute_mlir(vectorized_code)
            print(f"Vectorized Execution; Success = {success}, Time = {vec_time_ns} ns")
            print(f"Speedup vs original = {(original_time_ns / vec_time_ns):.4f}")

            if Vectorization.postcondition(tiled_code, vectorized_code, vec_params):
                print(f"Postcondition satisfied: Vectorization applied successfully.")
            else:
                print(f"Postcondition failed: Vectorization not applied as expected.")
        else:
            print(f"Precondition not met; Vectorization not applied.")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    test_vectorization()
