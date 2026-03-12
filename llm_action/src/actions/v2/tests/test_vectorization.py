from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code
from llm_action.src.execution.mlir_execution import execute_mlir

from llm_action.src.actions.v2.implementation.tiling import Tiling
from llm_action.src.actions.v2.implementation.vectorization import Vectorization

# Vectorization is typically applied after tiling to reduce iteration space.
# First tile, then vectorize the tiled result.
tile_then_vectorize_params = {
    KernelType.MATMUL: {
        "tile_params": {"tile_sizes": [8, 4, 4]},
        "vector_params": {"vector_sizes": [8, 4, 4]},
    },
    KernelType.CONV2D: {
        # Conv2d requires img2col decomposition before vectorization.
        # The Vectorization action handles this internally: conv2d is decomposed
        # into img2col + matmul-like linalg.generic with 4 iterators
        # [batch=128, filters=256, spatial=49, reduction=32].
        # tile_sizes and vector_sizes apply to this 4D matmul-like op.
        # Vector sizes: product(8*7*4) = 224 <= 1024, rank(>1) = 3 <= 3.
        "tile_params": None,  # Tiling is handled inside Vectorization for conv2d
        "vector_params": {"vector_sizes": [1, 8, 7, 4], "tile_sizes": [1, 8, 7, 4]},
    },
    KernelType.GENERIC: {
        # For generic 5D tensor, tile outer dims to 1 and vectorize innermost.
        "tile_params": {"tile_sizes": [1, 1, 1, 1]},
        "vector_params": {"vector_sizes": [1, 1, 1, 1, 32]},
    },
}

if __name__ == "__main__":

    for kernel_type in [KernelType.MATMUL, KernelType.CONV2D, KernelType.GENERIC]:
        print(f"--- Testing Vectorization Action on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)
        print(f"Original Code:\n{code}\n")

        original_time_ns, success = execute_mlir(code)
        print(f"Original Execution; Success = {success}, Time = {original_time_ns} ns")

        params = tile_then_vectorize_params[kernel_type]
        tile_params = params["tile_params"]
        vector_params = params["vector_params"]

        # Step 1: Tiling (skipped for conv2d where tiling is part of vectorization)
        code_to_vectorize = code
        if tile_params is not None:
            print(f"Step 1: Tiling with {tile_params}")
            if Tiling.precondition(code, tile_params):
                tiled_code = Tiling.implement(code, tile_params)
                tiling_ok = Tiling.postcondition(code, tiled_code, tile_params)
                print(f"Tiling succeeded: {tiling_ok}\n")
                if not tiling_ok:
                    print("Tiling failed; skipping vectorization.")
                    print("=" * 80 + "\n")
                    continue
                code_to_vectorize = tiled_code
            else:
                print(f"Tiling precondition not met; skipping.\n")
                print("=" * 80 + "\n")
                continue
        else:
            print("Step 1: Tiling skipped (handled inside Vectorization)\n")

        print(f"Step 2: Vectorizing with {vector_params}")
        if Vectorization.precondition(code_to_vectorize, vector_params):
            vectorized_code = Vectorization.implement(code_to_vectorize, vector_params)

            if Vectorization.postcondition(code_to_vectorize, vectorized_code, vector_params):
                print(f"Vectorized Code:\n{vectorized_code}\n")

                transformed_time_ns, success = execute_mlir(vectorized_code)
                print(f"Transformed Execution; Success = {success}, Time = {transformed_time_ns} ns")
                print(f"Speedup = {(original_time_ns / transformed_time_ns):.4f}")
                print(f"Postcondition satisfied: Vectorization applied successfully.")
            else:
                print(f"Postcondition failed: Vectorization transform did not change IR (may be unsupported for this kernel type).")
        else:
            print(f"Vectorization precondition not met (vector sizes too large or invalid).")
        print("=" * 80 + "\n")
