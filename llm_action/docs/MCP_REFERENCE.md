# MCP Tools

## transform_mlir_code(code: str, transformation_code: str) -> str
Applies MLIR transformations to the given code using custom transformation scripts.
    
This tool takes base MLIR code and applies user-defined transformations to it,
allowing for optimization passes, dialect conversions, or other code modifications.

Use this when you need to:
- Apply specific MLIR transformation passes to code
- Test different optimization strategies
- Convert between MLIR dialects
- Modify MLIR operations programmatically

Args:
    code: The base MLIR code to transform
    transformation_code: The transformation script/pass to apply

Returns:
    The transformed MLIR code as a string

## execute_mlir_code(code: str, bufferization_lowering_v_transform_code: str | None = None, pass_pipeline: list[str] | None = None) -> tuple[float, bool]
Submits a SLURM job to execute the given MLIR code on a dedicated compute node and returns the median execution time. This mirrors
execute_torch_matmul_by_shape to ensure fair benchmarking: both MLIR and
PyTorch run on identical hardware with the same resource reservations and thread-affinity settings.

Args:
    code (str): The MLIR code to execute.
    bufferization_lowering_v_transform_code (Optional[str]): Optional
        transformation code to apply for bufferization and lowering before execution.
    pass_pipeline (Optional[list[str]]): Optional list of MLIR passes to apply during execution. Do not include a wrapper, that's handled internally, basically provide the list of passes you want to run in the form of ["pass1", "pass2", ...].

Defaults (Current Functional Behavior):
    bufferization_lowering_v_transform_code:
```mlir
module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
        %all_loops = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
        transform.apply_licm to %all_loops : !transform.any_op

        transform.structured.eliminate_empty_tensors %arg0 : !transform.any_op
        %empty = transform.structured.match ops{["tensor.empty"]} in %arg0 : (!transform.any_op) -> !transform.op<"tensor.empty">
        transform.bufferization.empty_tensor_to_alloc_tensor %empty : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">

        %f0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f0 {
            transform.apply_patterns.vector.transfer_permutation_patterns
            transform.apply_patterns.vector.reduction_to_contract
        } : !transform.any_op
        transform.apply_patterns to %f0 {
            transform.apply_patterns.canonicalization
            transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers
        } : !transform.any_op

        %arg1 = transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %arg0 {bufferize_function_boundaries = true} : (!transform.any_op) -> !transform.any_op

        %f1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f1 {
            transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
            transform.apply_patterns.vector.transfer_permutation_patterns
            transform.apply_patterns.vector.lower_outerproduct
            transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerparallel"
            transform.apply_patterns.vector.transfer_to_scf max_transfer_rank = 1 full_unroll = true
            transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1
            transform.apply_patterns.vector.lower_shape_cast
            transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d"
            transform.apply_patterns.canonicalization
        } : !transform.any_op
        transform.yield
    }
}
```

    pass_pipeline:
```python
[
    "canonicalize",
    "cse",
    "buffer-deallocation-pipeline",
    "convert-bufferization-to-memref",
    "convert-linalg-to-loops",
    "scf-forall-to-parallel",
    "convert-scf-to-openmp",
    "fold-memref-alias-ops",
    "expand-strided-metadata",
    "finalize-memref-to-llvm",
    "convert-scf-to-cf",
    "lower-affine",

    "convert-openmp-to-llvm",
    "convert-vector-to-llvm",
    "convert-math-to-llvm",
    "convert-math-to-libm",
    "finalize-memref-to-llvm",
    "convert-func-to-llvm",
    "convert-index-to-llvm",
    "arith-unsigned-when-equivalent",
    "convert-arith-to-llvm",
    "convert-cf-to-llvm",

    "reconcile-unrealized-casts",
    "canonicalize",
    "cse"
]
```

Returns:
    tuple[float, bool]: (median execution time in milliseconds, assertion result)

## execute_torch_matmul_by_shape(M: int, K: int, N: int) -> float
Submits a SLURM job to execute a matrix multiplication (M×K) @ (K×N)
using PyTorch JIT on a compute node and returns the median execution time.

Use this to obtain a PyTorch baseline execution time for a given matrix
multiplication shape, which can then be compared against MLIR execution times
via the measure_speedup tool.

Args:
    M: Number of rows of the first matrix.
    K: Shared inner dimension (columns of first matrix / rows of second matrix).
    N: Number of columns of the second matrix.

Returns:
    float: the median execution time in milliseconds.

## measure_speedup(mlir_base_execution_time: float, mlir_optimized_execution_time: float, torch_execution_time: Optional[float] = None) -> dict[str, float]
Measures the speedup achieved by MLIR transformations.

This tool compares the execution time of base code against transformed code
to calculate the performance improvement factor. It compares against
a PyTorch baseline to compute the speedup relative to PyTorch.

Use this when you need to:
- Quantify optimization effectiveness
- Compare performance before and after transformations
- Calculate speedup ratios
- Evaluate transformation impact relative to PyTorch

Args:
    mlir_base_execution_time: Execution time of original (unoptimized) MLIR code in milliseconds
    mlir_optimized_execution_time: Execution time of transformed (optimized) MLIR code in milliseconds
    torch_execution_time: execution time of PyTorch baseline in milliseconds

Returns:
    dict with:
        speedup: mlir_base_execution_time / mlir_optimized_execution_time
        speedup_to_torch: torch_execution_time / mlir_optimized_execution_time  

## delegate_documentation_lookup(task: str) -> str
Delegates a documentation lookup task to the Documentation Lookup Agent.

This tool forwards a specific documentation retrieval task to the Documentation Lookup Agent, which specializes in finding authoritative references for MLIR Transform dialect operations from the official up-to-date documentation online.

Args:
    task: The documentation lookup task or question to be answered
    
Returns:
    The response from the Documentation Lookup Agent containing the requested documentation information