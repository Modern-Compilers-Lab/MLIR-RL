"""
Unit tests for all v1 actions.
Tests precondition, implement, and postcondition for each action
across matmul, conv2d, and generic kernels.
"""

from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code
from llm_action.src.execution.mlir_execution import execute_mlir

from llm_action.src.actions.v1.implementation.action_1 import TilingAction
from llm_action.src.actions.v1.implementation.action_2 import LoopInterchangeAction
from llm_action.src.actions.v1.implementation.action_3 import LoopFusionAction
from llm_action.src.actions.v1.implementation.action_4 import VectorizationAction
from llm_action.src.actions.v1.implementation.action_5 import UnrollingAction
from llm_action.src.actions.v1.implementation.action_6 import ParallelizationAction
from llm_action.src.actions.v1.implementation.action_7 import PackingAction


# ===== Test parameters per kernel type per action =====

TILING_PARAMS = {
    KernelType.MATMUL: {"tile_sizes": [64, 64, 64]},
    KernelType.CONV2D: {"tile_sizes": [32, 64, 0, 0, 0, 0, 0]},
    KernelType.GENERIC: {"tile_sizes": [4, 4, 0, 0, 0]},
}

INTERCHANGE_PARAMS = {
    # matmul has 3 iterators (M, N, K): swap M and N
    KernelType.MATMUL: {"iterator_interchange": [1, 0, 2]},
    # conv2d has 7 iterators (N,F,C,OH,OW,KH,KW): swap N and F
    KernelType.CONV2D: {"iterator_interchange": [1, 0, 2, 3, 4, 5, 6]},
    # generic has 5 iterators: swap first two
    KernelType.GENERIC: {"iterator_interchange": [1, 0, 2, 3, 4]},
}

FUSION_PARAMS = {
    KernelType.MATMUL: {"tile_sizes": [64, 64, 64]},
    KernelType.CONV2D: {"tile_sizes": [32, 64, 0, 0, 0, 0, 0]},
    KernelType.GENERIC: {"tile_sizes": [4, 4, 0, 0, 0]},
}

VECTORIZATION_TILE_PARAMS = {
    # First tile to small sizes, then vectorize the tiled op
    KernelType.MATMUL: {"tile_sizes": [4, 4, 4]},
    KernelType.CONV2D: {"tile_sizes": [1, 1, 1, 7, 7, 1, 1]},
    KernelType.GENERIC: {"tile_sizes": [1, 1, 1, 8, 4]},
}

VECTORIZATION_PARAMS = {
    # Vector sizes must match the tile sizes (>= iteration space of tiled op)
    KernelType.MATMUL: {"vector_sizes": [4, 4, 4]},
    KernelType.CONV2D: {"vector_sizes": [1, 1, 1, 7, 7, 1, 1]},
    KernelType.GENERIC: {"vector_sizes": [1, 1, 1, 8, 4]},
}

UNROLLING_PARAMS = {
    KernelType.MATMUL: {"tile_sizes": [64, 64, 64], "unroll_factor": 4},
    KernelType.CONV2D: {"tile_sizes": [32, 64, 0, 0, 0, 0, 0], "unroll_factor": 2},
    KernelType.GENERIC: {"tile_sizes": [4, 4, 0, 0, 0], "unroll_factor": 2},
}

PARALLELIZATION_PARAMS = {
    KernelType.MATMUL: {"num_threads": [4, 4, 0]},
    KernelType.CONV2D: {"num_threads": [4, 4, 0, 0, 0, 0, 0]},
    KernelType.GENERIC: {"num_threads": [4, 4, 0, 0, 0]},
}

PACKING_PARAMS = {
    KernelType.MATMUL: {"packed_sizes": [32, 32, 32]},
    # conv2d has 7 iterators; pack the first two
    KernelType.CONV2D: {"packed_sizes": [32, 32, 0, 0, 0, 0, 0]},
    # generic has 5 iterators; pack first two
    KernelType.GENERIC: {"packed_sizes": [4, 4, 0, 0, 0]},
}


def run_action_test(action_cls, params_per_kernel, kernel_types=None, test_execution=True):
    """Generic test runner for an action across kernel types."""
    if kernel_types is None:
        kernel_types = [KernelType.MATMUL, KernelType.CONV2D, KernelType.GENERIC]

    for kernel_type in kernel_types:
        print(f"\n--- Testing {action_cls.__name__} on {kernel_type.value} ---")
        code = load_kernel_code(kernel_type)
        params = params_per_kernel[kernel_type]
        print(f"Parameters: {params}")

        # Test precondition
        assert action_cls.precondition(code, params), \
            f"Precondition failed for {action_cls.__name__} on {kernel_type.value}"
        print("  Precondition: PASS")

        # Test preprocess (should return code unchanged for most actions)
        preprocessed = action_cls.preprocess(code, params)
        assert isinstance(preprocessed, str) and len(preprocessed) > 0, \
            f"Preprocess returned invalid result for {action_cls.__name__} on {kernel_type.value}"
        print("  Preprocess: PASS")

        # Test implement
        transformed = action_cls.implement(preprocessed, params)
        assert isinstance(transformed, str) and len(transformed) > 0, \
            f"Implement returned invalid result for {action_cls.__name__} on {kernel_type.value}"
        print("  Implement: PASS (IR generated)")

        # Test postcondition
        assert action_cls.postcondition(code, transformed, params), \
            f"Postcondition failed for {action_cls.__name__} on {kernel_type.value}"
        print("  Postcondition: PASS")

        # Test execution (optional)
        if test_execution:
            try:
                exec_time, success = execute_mlir(transformed)
                print(f"  Execution: {'PASS' if success else 'FAIL'} (time={exec_time} ns)")
                assert success, f"Execution failed for {action_cls.__name__} on {kernel_type.value}"
            except Exception as e:
                print(f"  Execution: FAIL ({e})")
                raise

    print(f"\n=== {action_cls.__name__}: ALL TESTS PASSED ===\n")


def test_precondition_rejects_invalid():
    """Test that preconditions correctly reject invalid inputs."""
    code_with_tag = 'tag = "operation_0"'
    code_without_tag = 'some other code'

    # Tiling
    assert not TilingAction.precondition(code_without_tag, {"tile_sizes": [64]})
    assert not TilingAction.precondition(code_with_tag, {"tile_sizes": [0, 0, 0]})
    assert not TilingAction.precondition(code_with_tag, {"tile_sizes": []})
    assert not TilingAction.precondition(code_with_tag, {})

    # Interchange
    assert not LoopInterchangeAction.precondition(code_with_tag, {"iterator_interchange": [0, 1, 2]})  # identity
    assert not LoopInterchangeAction.precondition(code_with_tag, {"iterator_interchange": [0, 0]})  # not a perm
    assert not LoopInterchangeAction.precondition(code_with_tag, {"iterator_interchange": []})

    # Vectorization
    assert not VectorizationAction.precondition(code_with_tag, {"vector_sizes": [0, 4]})  # zero not allowed
    assert not VectorizationAction.precondition(code_with_tag, {"vector_sizes": []})

    # Unrolling
    assert not UnrollingAction.precondition(code_with_tag, {"tile_sizes": [64], "unroll_factor": 1})
    assert not UnrollingAction.precondition(code_with_tag, {"tile_sizes": [0], "unroll_factor": 4})

    # Parallelization
    assert not ParallelizationAction.precondition(code_with_tag, {"num_threads": [0, 0]})
    assert not ParallelizationAction.precondition(code_with_tag, {"num_threads": []})

    # Packing
    assert not PackingAction.precondition(code_with_tag, {"packed_sizes": [0, 0, 0]})
    assert not PackingAction.precondition(code_with_tag, {"packed_sizes": []})

    print("=== Precondition rejection tests: ALL PASSED ===\n")


if __name__ == "__main__":
    import sys

    # Run precondition rejection tests first (fast, no MLIR needed)
    test_precondition_rejects_invalid()

    # Determine which actions to test
    test_execution = "--no-exec" not in sys.argv

    print("=" * 80)
    print("Action 1: Tiling")
    print("=" * 80)
    run_action_test(TilingAction, TILING_PARAMS, test_execution=test_execution)

    print("=" * 80)
    print("Action 2: Loop Interchange")
    print("=" * 80)
    run_action_test(LoopInterchangeAction, INTERCHANGE_PARAMS, test_execution=test_execution)

    print("=" * 80)
    print("Action 3: Loop Fusion")
    print("=" * 80)
    run_action_test(LoopFusionAction, FUSION_PARAMS, test_execution=test_execution)

    print("=" * 80)
    print("Action 4: Vectorization (tile first, then vectorize)")
    print("=" * 80)
    # Vectorization must be applied after tiling to produce safe vector sizes.
    # Conv2d vectorization requires decomposition not handled here; tested on matmul+generic.
    for kernel_type in [KernelType.MATMUL, KernelType.GENERIC]:
        print(f"\n--- Testing VectorizationAction on {kernel_type.value} ---")
        code = load_kernel_code(kernel_type)
        # First tile
        tile_params = VECTORIZATION_TILE_PARAMS[kernel_type]
        tiled = TilingAction.implement(code, tile_params)
        assert TilingAction.postcondition(code, tiled, tile_params), \
            f"Tiling failed for vectorization test on {kernel_type.value}"
        print(f"  Tiling: PASS (tile_sizes={tile_params['tile_sizes']})")
        # Then vectorize
        vec_params = VECTORIZATION_PARAMS[kernel_type]
        assert VectorizationAction.precondition(tiled, vec_params)
        vectorized = VectorizationAction.implement(tiled, vec_params)
        assert VectorizationAction.postcondition(tiled, vectorized, vec_params), \
            f"Vectorization postcondition failed on {kernel_type.value}"
        print(f"  Vectorization: PASS (vector_sizes={vec_params['vector_sizes']})")
        if test_execution:
            exec_time, success = execute_mlir(vectorized)
            print(f"  Execution: {'PASS' if success else 'FAIL'} (time={exec_time} ns)")
            assert success, f"Vectorized execution failed on {kernel_type.value}"
    print("\n=== VectorizationAction: ALL TESTS PASSED ===\n")

    print("=" * 80)
    print("Action 5: Unrolling")
    print("=" * 80)
    run_action_test(UnrollingAction, UNROLLING_PARAMS, test_execution=test_execution)

    print("=" * 80)
    print("Action 6: Parallelization")
    print("=" * 80)
    run_action_test(ParallelizationAction, PARALLELIZATION_PARAMS, test_execution=test_execution)

    print("=" * 80)
    print("Action 7: Packing")
    print("=" * 80)
    run_action_test(PackingAction, PACKING_PARAMS, test_execution=test_execution)

    print("\n" + "=" * 80)
    print("ALL V1 ACTION TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
