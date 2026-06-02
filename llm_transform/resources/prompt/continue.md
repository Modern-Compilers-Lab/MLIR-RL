Target not met for all in-scope instances. Do not conclude the session.

Step back and execute a deeper microarchitectural analysis:
1. Enumerate the exact bottlenecks (e.g., register pressure, cache thrashing, instruction-level parallelism stalls) keeping the current best configs from hitting >2x.
2. Propose at least 3 fundamentally distinct transformation/lowering strategies or radical pass-ordering shifts you have not yet tried.
3. Test them systematically using `run_schedule` and verify the generated assembly via `lower_schedule`.

Continue iterating until the target is hit or all structural alternatives are mathematically exhausted.