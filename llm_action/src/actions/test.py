import random

from llm_action.src.actions.base import ActionBase
from llm_action.src.data.benchmarks import group_by_family, load_benchmark_set
from llm_action.src.execution.mlir_execution import execute_mlir


def test_action(
    action_cls: type[ActionBase],
    params_per_family: dict[str, dict],
    benchmark: str = "standard",
    split: str = "train",
    seed: int = 0,
) -> None:
    """Run the action against one random instance per family declared in `params_per_family`.

    Args:
        action_cls: action class (subclass of `ActionBase`). All ActionBase
            methods are `@classmethod`, so the class is the natural unit.
        params_per_family: per-family parameter dict keyed by the family strings
            from `_FAMILY_PATTERNS` (e.g. `"matmul"`, `"conv_2d_nchw_fchw"`,
            `"pooling_nchw"`, `"add"`, `"relu"`, `"generic"`).
        benchmark: name of the benchmark set under `data/benchmarks/`.
        split: `"train"`, `"eval"`, or `"all"`.
        seed: RNG seed for the per-family random pick (deterministic).
    """
    instances = load_benchmark_set(benchmark, split=split)
    groups = group_by_family(instances)
    rng = random.Random(seed)

    for family, parameters in params_per_family.items():
        bucket = groups.get(family)
        if not bucket:
            print(f"[skip] family '{family}': no instances in '{benchmark}/{split}'\n")
            continue

        chosen = rng.choice(bucket)
        print(
            f"--- Testing {action_cls.__name__} on family '{family}' "
            f"(benchmark='{benchmark}/{split}', instance='{chosen.name}') ---\n"
        )
        code = chosen.code
        print(f"Original Code:\n{code}\n")

        original_time_ns, success = execute_mlir(code)
        print(f"Original Execution; Success = {success}, Time = {original_time_ns} ns")

        print(f"Using Parameters: {parameters}\n")

        if action_cls.precondition(code, parameters):
            transformed_code = action_cls.implement(code, parameters)
            print(f"Transformed Code:\n{transformed_code}\n")

            transformed_time_ns, success = execute_mlir(transformed_code)
            print(f"Transformed Execution; Success = {success}, Time = {transformed_time_ns} ns")
            if transformed_time_ns:
                print(f"Speedup = {(original_time_ns / transformed_time_ns):.4f}")

            if action_cls.postcondition(code, transformed_code, parameters):
                print(f"Postcondition satisfied: {action_cls.__name__} applied successfully.")
            else:
                print(f"Postcondition failed: {action_cls.__name__} not applied as expected.")
        else:
            print(f"Precondition not met; {action_cls.__name__} not applied.")
        print("=" * 80 + "\n")
