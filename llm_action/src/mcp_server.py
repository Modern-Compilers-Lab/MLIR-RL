from fastmcp import FastMCP
from llm_action.src.utils.transformation import run_transform_code, transform_bufferize_and_lower_v, execute_bufferized_code
from llm_action.src.documentation import load_documentation

mcp = FastMCP("mlir-optimization-tools")

@mcp.tool()
def transform_code(code: str, transformation_code: str) -> str:
    """
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
    """
    return run_transform_code(code, transformation_code)

@mcp.tool()
def execute_code(code: str, bufferization_lowering_v_transform_code: str | None = None, pass_pipeline: list[str] | None = None) -> dict:
    """
    Executes a given MLIR code with a timeout.

    Args:
        code (str): The MLIR code to execute
        bufferization_lowering_v_transform_code (Optional[str]): Optional transformation code to apply for bufferization and lowering before execution.
        pass_pipeline (Optional[list[str]]): Optional list of MLIR passes to apply during execution.
            
    Defaults (Current Functional Behavior):
        bufferization_lowering_v_transform_code:
        ```module attributes {transform.with_named_sequence} {
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
                    transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerparallel"
                    transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "linalg-copy"
                    transform.apply_patterns.vector.transfer_to_scf max_transfer_rank = 3 full_unroll = true
                    transform.apply_patterns.vector.lower_transfer max_transfer_rank = 3
                    transform.apply_patterns.vector.lower_shape_cast
                    transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d"
                    transform.apply_patterns.canonicalization
                } : !transform.any_op
                transform.yield
            }
        }```
        
        pass_pipeline:
        [canonicalize, buffer-deallocation-pipeline, convert-bufferization-to-memref, convert-linalg-to-loops, scf-forall-to-parallel, convert-scf-to-openmp, expand-strided-metadata, finalize-memref-to-llvm, convert-scf-to-cf, lower-affine, convert-openmp-to-llvm, convert-vector-to-llvm, convert-math-to-llvm, convert-math-to-libm, finalize-memref-to-llvm, convert-func-to-llvm, convert-index-to-llvm, convert-arith-to-llvm, convert-cf-to-llvm, reconcile-unrealized-casts, canonicalize, cse]

    Returns:
        tuple[int, bool]: (execution time in milliseconds, assertion result)
    """
    bufferized = transform_bufferize_and_lower_v(code, transform_code=bufferization_lowering_v_transform_code)
    exec_time_ns, success = execute_bufferized_code(bufferized, pass_pipeline=pass_pipeline)
    return {"execution_time_ms": exec_time_ns / 1_000_000, "success": success}

@mcp.tool()
def measure_speedup(base_execution_time: float, execution_time: float) -> float:
    """
    Measures the speedup achieved by MLIR transformations.
    
    This tool compares the execution time of base code against transformed code
    to calculate the performance improvement factor.
    
    Use this when you need to:
    - Quantify optimization effectiveness
    - Compare performance before and after transformations
    - Calculate speedup ratios
    - Evaluate transformation impact
    
    Args:
        base_execution_time: Execution time of original code in nanoseconds
        execution_time: Execution time of transformed code in nanoseconds
    
    Returns:
        The speedup ratio as a float (base_time / transformed_time)
    """
    return base_execution_time / execution_time

@mcp.tool()
def list_transformations() -> str:
    """Lists all available MLIR Transform dialect transformation categories and names."""
    doc = load_documentation()
    lines = []
    for category, transforms in doc.items():
        lines.append(f"## {category}")
        for name in transforms:
            lines.append(f"  - {name}")
    return "\n".join(lines)

@mcp.tool()
def lookup_transformation(category_name: str, transformation_name: str) -> str:
    """
    Looks up MLIR Transform dialect documentation for a specific transformation.
    
    This tool retrieves detailed documentation for a given transformation from the MLIR Transform dialect reference, including operation names, required operands/results, attributes, and example snippets.
    
    Args:
        category_name: The name of the transformation category
        transformation_name: The name of the specific transformation
    
    Returns:
        Detailed documentation string for the specified transformation
    """
    doc = load_documentation()
    return doc[category_name][transformation_name]

if __name__ == "__main__":
    mcp.run()
