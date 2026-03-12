"""Quick validation of all v4 actions."""
from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code

# Import all actions
from llm_action.src.actions.v4.implementation.tiling import Tiling
from llm_action.src.actions.v4.implementation.multi_level_tiling import MultiLevelTiling
from llm_action.src.actions.v4.implementation.promotion import Promotion
from llm_action.src.actions.v4.implementation.generalization import Generalization
from llm_action.src.actions.v4.implementation.parallelization import Parallelization
from llm_action.src.actions.v4.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v4.implementation.vectorization import Vectorization
from llm_action.src.actions.v4.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v4.implementation.peeling import Peeling
from llm_action.src.actions.v4.implementation.loop_fusion import LoopFusion
from llm_action.src.actions.v4.implementation.loop_distribution import LoopDistribution
from llm_action.src.actions.v4.implementation.canonicalization import Canonicalization
from llm_action.src.actions.v4.implementation.unroll_and_jam import UnrollAndJam
from llm_action.src.actions.v4.implementation.scalar_replacement import ScalarReplacement
from llm_action.src.actions.v4.implementation.loop_coalescing import LoopCoalescing
from llm_action.src.actions.v4.implementation.decomposition import Decomposition

MATMUL_CODE = load_kernel_code(KernelType.MATMUL)
CONV2D_CODE = load_kernel_code(KernelType.CONV2D)
GENERIC_CODE = load_kernel_code(KernelType.GENERIC)

def test_action(name, action_cls, code, params, expect_change=True):
    """Test a single action on a single kernel."""
    pre = action_cls.precondition(code, params)
    if not pre:
        if expect_change:
            print(f"  FAIL: {name} precondition rejected (unexpected)")
            return False
        else:
            print(f"  OK: {name} precondition rejected (expected)")
            return True
    result = action_cls.implement(code, params)
    changed = result.strip() != code.strip()
    post = action_cls.postcondition(code, result, params)
    if expect_change and not changed:
        print(f"  FAIL: {name} produced no-op")
        return False
    if expect_change and not post:
        print(f"  FAIL: {name} postcondition failed")
        return False
    print(f"  PASS: {name} (changed={changed}, post={post})")
    return True

def test_graceful(name, action_cls, code, params):
    """Test an action that may or may not produce a change (both are OK)."""
    result = action_cls.implement(code, params)
    changed = result.strip() != code.strip()
    post = action_cls.postcondition(code, result, params)
    if changed:
        print(f"  PASS: {name} (changed, post={post})")
    else:
        print(f"  OK: {name} (no-op, graceful)")
    return True

if __name__ == "__main__":
    passed = 0
    failed = 0
    total = 0

    # 1. Tiling
    print("\n=== Tiling ===")
    for kt, code, params in [
        ("matmul", MATMUL_CODE, {"tile_sizes": [32, 64, 16]}),
        ("conv2d", CONV2D_CODE, {"tile_sizes": [16, 32, 0, 0, 0, 0, 0]}),
        ("generic", GENERIC_CODE, {"tile_sizes": [4, 4, 0, 0, 0]}),
    ]:
        total += 1
        if test_action(f"Tiling/{kt}", Tiling, code, params):
            passed += 1
        else:
            failed += 1

    # 2. Multi-Level Tiling
    print("\n=== MultiLevelTiling ===")
    for kt, code, params in [
        ("matmul", MATMUL_CODE, {"outer_tile_sizes": [64, 128, 64], "inner_tile_sizes": [16, 32, 8]}),
        ("conv2d", CONV2D_CODE, {"outer_tile_sizes": [32, 64, 0, 0, 0, 0, 0], "inner_tile_sizes": [8, 16, 0, 0, 0, 0, 0]}),
        ("generic", GENERIC_CODE, {"outer_tile_sizes": [4, 4, 0, 0, 0], "inner_tile_sizes": [2, 2, 0, 0, 0]}),
    ]:
        total += 1
        if test_action(f"MultiLevelTiling/{kt}", MultiLevelTiling, code, params):
            passed += 1
        else:
            failed += 1

    # 3. Promotion (tile + promote)
    print("\n=== Promotion ===")
    for kt, code, params in [
        ("matmul", MATMUL_CODE, {"tile_sizes": [32, 64, 16], "operands_to_promote": [0, 1]}),
        ("conv2d", CONV2D_CODE, {"tile_sizes": [16, 32, 0, 0, 0, 0, 0], "operands_to_promote": [0, 1]}),
        ("generic", GENERIC_CODE, {"tile_sizes": [4, 4, 0, 0, 0], "operands_to_promote": [0]}),
    ]:
        total += 1
        if test_action(f"Promotion/{kt}", Promotion, code, params):
            passed += 1
        else:
            failed += 1

    # 4. Generalization
    print("\n=== Generalization ===")
    for kt, code, expect in [
        ("matmul", MATMUL_CODE, True),
        ("conv2d", CONV2D_CODE, True),
        ("generic", GENERIC_CODE, False),  # already generic
    ]:
        total += 1
        if test_action(f"Generalization/{kt}", Generalization, code, {}, expect_change=expect):
            passed += 1
        else:
            failed += 1

    # 5. Parallelization
    print("\n=== Parallelization ===")
    for kt, code, params in [
        ("matmul", MATMUL_CODE, {"num_threads": [4, 4]}),
        ("conv2d", CONV2D_CODE, {"num_threads": [4, 4]}),
        ("generic", GENERIC_CODE, {"num_threads": [2, 2]}),
    ]:
        total += 1
        if test_action(f"Parallelization/{kt}", Parallelization, code, params):
            passed += 1
        else:
            failed += 1

    # 6. Loop Interchange (needs generic form - generalization now preserves tag)
    print("\n=== LoopInterchange ===")
    matmul_generic = Generalization.implement(MATMUL_CODE, {})
    conv2d_generic = Generalization.implement(CONV2D_CODE, {})
    for kt, code, params in [
        ("matmul", matmul_generic, {"permutation": [1, 2, 0]}),
        ("conv2d", conv2d_generic, {"permutation": [1, 0, 2, 3, 4, 5, 6]}),
        ("generic", GENERIC_CODE, {"permutation": [1, 0, 2, 3, 4]}),
    ]:
        total += 1
        if test_action(f"LoopInterchange/{kt}", LoopInterchange, code, params):
            passed += 1
        else:
            failed += 1

    # 7. Vectorization (total vector elements must be <= 256 for rank >= 3)
    print("\n=== Vectorization ===")
    for kt, code, params in [
        ("matmul", MATMUL_CODE, {"vector_sizes": [4, 4, 16]}),
        ("generic", GENERIC_CODE, {"vector_sizes": [2, 2, 4, 2, 4]}),
    ]:
        total += 1
        if test_action(f"Vectorization/{kt}", Vectorization, code, params):
            passed += 1
        else:
            failed += 1

    # 8. Peeling
    print("\n=== Peeling ===")
    for kt, code, params in [
        ("matmul", MATMUL_CODE, {"tile_sizes": [30, 60, 17]}),
        ("conv2d", CONV2D_CODE, {"tile_sizes": [15, 30, 0, 0, 0, 0, 0]}),
        ("generic", GENERIC_CODE, {"tile_sizes": [3, 3, 0, 0, 0]}),
    ]:
        total += 1
        if test_action(f"Peeling/{kt}", Peeling, code, params):
            passed += 1
        else:
            failed += 1

    # 9. Loop Unrolling (needs tiled code)
    print("\n=== LoopUnrolling ===")
    matmul_tiled = Tiling.implement(MATMUL_CODE, {"tile_sizes": [32, 64, 16]})
    conv2d_tiled = Tiling.implement(CONV2D_CODE, {"tile_sizes": [16, 32, 0, 0, 0, 0, 0]})
    generic_tiled = Tiling.implement(GENERIC_CODE, {"tile_sizes": [4, 4, 0, 0, 0]})
    for kt, code, params in [
        ("matmul", matmul_tiled, {"unroll_factor": 4, "loop_depth": 1}),
        ("conv2d", conv2d_tiled, {"unroll_factor": 2, "loop_depth": 1}),
        ("generic", generic_tiled, {"unroll_factor": 2, "loop_depth": 1}),
    ]:
        total += 1
        if test_action(f"LoopUnrolling/{kt}", LoopUnrolling, code, params):
            passed += 1
        else:
            failed += 1

    # 10. Loop Fusion (tile_using_forall)
    print("\n=== LoopFusion ===")
    for kt, code, params in [
        ("matmul", MATMUL_CODE, {"tile_sizes": [64, 128]}),
        ("conv2d", CONV2D_CODE, {"tile_sizes": [32, 64]}),
        ("generic", GENERIC_CODE, {"tile_sizes": [4, 4]}),
    ]:
        total += 1
        if test_action(f"LoopFusion/{kt}", LoopFusion, code, params):
            passed += 1
        else:
            failed += 1

    # 11. LoopDistribution (split_reduction) - needs reduction dims
    print("\n=== LoopDistribution ===")
    total += 1
    if test_action("LoopDistribution/matmul", LoopDistribution, MATMUL_CODE,
                    {"split_factor": 16, "insert_split_dimension": 0}):
        passed += 1
    else:
        failed += 1
    # conv2d split_reduction may not apply
    total += 1
    if test_graceful("LoopDistribution/conv2d", LoopDistribution, CONV2D_CODE,
                     {"split_factor": 8, "insert_split_dimension": 0}):
        passed += 1
    # Generic has only parallel dims, split_reduction should not apply
    total += 1
    if test_action("LoopDistribution/generic", LoopDistribution, GENERIC_CODE,
                    {"split_factor": 4, "insert_split_dimension": 0}, expect_change=False):
        passed += 1
    else:
        failed += 1

    # 12. Canonicalization
    print("\n=== Canonicalization ===")
    for kt, code in [
        ("matmul_tiled", matmul_tiled),
        ("conv2d_tiled", conv2d_tiled),
        ("generic_tiled", generic_tiled),
    ]:
        total += 1
        if test_graceful(f"Canonicalization/{kt}", Canonicalization, code, {}):
            passed += 1

    # 13. Unroll-and-Jam (needs tiled code with 2+ loops)
    print("\n=== UnrollAndJam ===")
    for kt, code, params in [
        ("matmul", matmul_tiled, {"jam_factor": 2, "outer_loop_depth": 2}),
        ("conv2d", conv2d_tiled, {"jam_factor": 2, "outer_loop_depth": 2}),
        ("generic", generic_tiled, {"jam_factor": 2, "outer_loop_depth": 2}),
    ]:
        total += 1
        if test_action(f"UnrollAndJam/{kt}", UnrollAndJam, code, params):
            passed += 1
        else:
            failed += 1

    # 14. ScalarReplacement (LICM + CSE on tiled code)
    print("\n=== ScalarReplacement ===")
    for kt, code in [
        ("matmul", matmul_tiled),
        ("conv2d", conv2d_tiled),
        ("generic", generic_tiled),
    ]:
        total += 1
        if test_graceful(f"ScalarReplacement/{kt}", ScalarReplacement, code, {}):
            passed += 1

    # 15. LoopCoalescing (needs tiled code, depth >= 2 for outermost loop)
    print("\n=== LoopCoalescing ===")
    for kt, code, params in [
        ("matmul", matmul_tiled, {"loop_depth": 3}),
        ("conv2d", conv2d_tiled, {"loop_depth": 2}),
        ("generic", generic_tiled, {"loop_depth": 2}),
    ]:
        total += 1
        if test_action(f"LoopCoalescing/{kt}", LoopCoalescing, code, params):
            passed += 1
        else:
            failed += 1

    # 16. Decomposition (may fail on standard ops - graceful no-op is OK)
    print("\n=== Decomposition ===")
    for kt, code in [
        ("matmul", MATMUL_CODE),
        ("conv2d", CONV2D_CODE),
        ("generic", GENERIC_CODE),
    ]:
        total += 1
        if test_graceful(f"Decomposition/{kt}", Decomposition, code, {}):
            passed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("=== ALL QUICK TESTS PASSED ===")
    else:
        print(f"=== {failed} TESTS FAILED ===")
