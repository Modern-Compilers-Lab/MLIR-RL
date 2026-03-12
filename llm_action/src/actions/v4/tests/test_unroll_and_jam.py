from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code

from llm_action.src.actions.v4.implementation.tiling import Tiling
from llm_action.src.actions.v4.implementation.unroll_and_jam import UnrollAndJam

# First tile to create loop nests, then unroll-and-jam an outer loop
tiling_params_per_kernel = {
    KernelType.MATMUL: {"tile_sizes": [32, 64, 16]},
    KernelType.CONV2D: {"tile_sizes": [16, 32, 0, 0, 0, 0, 0]},
    KernelType.GENERIC: {"tile_sizes": [4, 4, 0, 0, 0]},
}

unroll_and_jam_params = {
    "jam_factor": 2,
    "outer_loop_depth": 2,  # unroll the 2nd parent loop
}

if __name__ == "__main__":

    ACTION = UnrollAndJam

    for kernel_type in [KernelType.MATMUL, KernelType.CONV2D, KernelType.GENERIC]:
        print(f"--- Testing {ACTION.__name__} Action on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)

        # First tile to create loops (pre-step)
        tile_params = tiling_params_per_kernel[kernel_type]
        assert Tiling.precondition(code, tile_params), \
            f"Tiling precondition failed for {kernel_type.value}"
        code = Tiling.implement(code, tile_params)
        print("Pre-step: Tiled code to create scf.for loops")

        parameters = unroll_and_jam_params
        print(f"Using Parameters: {parameters}")

        assert ACTION.precondition(code, parameters), f"Precondition failed for {kernel_type.value}"
        print("Precondition: PASS")

        transformed_code = ACTION.implement(code, parameters)
        assert transformed_code.strip() != code.strip(), f"Transform produced no-op for {kernel_type.value}"
        print("Transform applied: PASS")

        assert ACTION.postcondition(code, transformed_code, parameters), \
            f"Postcondition failed for {kernel_type.value}"
        print("Postcondition: PASS")

        # Should still have scf.for loops
        assert "scf.for" in transformed_code, f"No scf.for loops found for {kernel_type.value}"
        print("Structure check (scf.for present): PASS")

        print(f"\n{ACTION.__name__} on {kernel_type.value}: ALL CHECKS PASSED")
        print("=" * 80 + "\n")

    # Test precondition rejects invalid inputs
    # Use untiled code (no scf.for)
    raw_code = load_kernel_code(KernelType.MATMUL)
    assert not ACTION.precondition(raw_code, {"jam_factor": 2, "outer_loop_depth": 2}), \
        "Should reject code without scf.for loops"
    # Use tiled code but invalid params
    tiled_code = Tiling.implement(raw_code, {"tile_sizes": [32, 64, 16]})
    assert not ACTION.precondition(tiled_code, {"jam_factor": 1}), \
        "Should reject jam_factor < 2"
    assert not ACTION.precondition(tiled_code, {}), "Should reject empty params"
    assert not ACTION.precondition("no tag here", {"jam_factor": 2}), \
        "Should reject missing tag"
    print("Precondition rejection tests: ALL PASSED")
    print("\n=== ALL UNROLL_AND_JAM TESTS PASSED ===")
