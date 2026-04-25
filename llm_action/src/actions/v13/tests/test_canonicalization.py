from llm_action.src.models import KernelType
from llm_action.src.actions.v13.implementation.canonicalization import Canonicalization
from llm_action.src.actions.v13.implementation.tiling import Tiling
from llm_action.src.utils.persistence import load_kernel_code
from llm_action.src.execution.mlir_execution import execute_mlir


if __name__ == "__main__":
    print("--- Testing Canonicalization Action on matmul Kernel ---\n")
    code = load_kernel_code(KernelType.MATMUL)

    # First tile to create IR with complexity for canonicalization
    tile_params = {"tile_sizes": [32, 32, 32]}
    tiled_code = Tiling.implement(code, tile_params)
    print(f"Pre-tiled code (input to canonicalization):\n{tiled_code}...\n")

    original_time_ns, success = execute_mlir(tiled_code)
    print(f"Pre-tiled Execution; Success = {success}, Time = {original_time_ns} ns")

    canon_params = {}
    print(f"Precondition: {Canonicalization.precondition(tiled_code, canon_params)}")

    if Canonicalization.precondition(tiled_code, canon_params):
        result = Canonicalization.implement(tiled_code, canon_params)
        print(f"\nCanonicalized code:\n{result}...\n")

        transformed_time_ns, success = execute_mlir(result)
        print(f"Canonicalized Execution; Success = {success}, Time = {transformed_time_ns} ns")
        print(f"Speedup = {(original_time_ns / transformed_time_ns):.4f}")

        if Canonicalization.postcondition(tiled_code, result, canon_params):
            print("Postcondition satisfied: Canonicalization applied successfully.")
        else:
            print("Postcondition failed: Canonicalization not applied as expected.")
    else:
        print("Precondition not met; Canonicalization not applied.")
    print("=" * 80)
