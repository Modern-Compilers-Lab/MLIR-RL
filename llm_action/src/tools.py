from llm_action.src.utils.transformation import transform_bufferize_and_lower_v, execute_bufferized_code, run_transform_code

from agno.tools import tool

@tool(
    name="measure_speedup",
    description="""
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
    """,
    show_result=True,
    stop_after_tool_call=False
)
def measure_speedup(base_execution_time: float, execution_time: float) -> float:
    return base_execution_time / execution_time

@tool(
    name="transform_code",
    description="""
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
    """,
    show_result=True,
    stop_after_tool_call=False
)
def transform_code(code: str, transformation_code: str) -> str:
    return run_transform_code(code, transformation_code)

@tool(
    name="execute_code",
    description="""
    Executes MLIR code and measures its performance with assertion validation.
    
    This tool compiles and runs MLIR code through a bufferization and lowering pipeline,
    then executes it to measure real execution time and verify correctness through assertions.
    
    Use this when you need to:
    - Benchmark MLIR code performance
    - Verify that transformations maintain correctness
    - Measure execution time in nanoseconds
    - Validate code functionality through assertions
    
    The code goes through:
    1. Bufferization (converts tensor operations to memref)
    2. Lowering (converts high-level dialects to lower-level representations)
    3. Execution with timing and assertion checking
    
    Args:
        code: The MLIR code to execute
    
    Returns:
        tuple[int, bool]: (execution time in nanoseconds, assertion success/failure)
        - First element: Real execution time measured in nanoseconds
        - Second element: True if all assertions passed, False otherwise
    """,
    show_result=True,
    stop_after_tool_call=False
)
def execute_code(code: str) -> tuple[int, bool]:
    """Evaluates the given MLIR code with a timeout.

    Args:
        state (OperationState): The operation state to evaluate.
        tmp_exec_data_file (str): The path to the temporary execution data file.

    Returns:
        tuple[int, bool]: (execution time in nanoseconds, assertion result)
    """

    bufferized_code = transform_bufferize_and_lower_v(code)
    real_exec_time, success = execute_bufferized_code(bufferized_code)
    return real_exec_time, success

