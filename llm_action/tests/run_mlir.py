from llm_action.src.utils.transformation import transform_bufferize_and_lower_v, execute_bufferized_code, run_transform_code

from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code

code = load_kernel_code(KernelType.MATMUL)
print("=== Base Code ===")
print(code)

bufferized_code = transform_bufferize_and_lower_v(code)
print("=== Bufferized Code ===")
print(bufferized_code)

real_exec_time, success = execute_bufferized_code(bufferized_code)
print(f"=== Execution Result ===")
print(f"Execution Time (ms): {real_exec_time/1000000}")
print(f"Assertion Success: {success}")
