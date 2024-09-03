from utils.transforms import *
from utils.observation_utils import *
from data_generation.data_generation_from_model import transform_wrapper
import numpy as np

operation = """linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"], tag = "operation_0"} ins(%4 : tensor<16xf32>) outs(%19 : tensor<32x16x52x52xf32>) {\n^bb0(%in: f32, %out: f32):\n  linalg.yield %in : f32\n} -> tensor<32x16x52x52xf32>"""


code = transform_wrapper(operation)


old_exec_time = np.median([evaluate_code(code) for _ in range(30)])
print('Base execution time:', old_exec_time / 1e9, 'ms')

new_code = code
new_code = transform_dialect_TP(new_code, 'operation_0', [4, 4, 0, 0])
new_code = transform_dialect_tile(new_code, 'operation_0', [2, 16, 4, 4])
new_code = transform_dialect_vectorise(new_code, 'operation_0')


# print(new_code)

new_exec_time = np.median([evaluate_code(new_code) for _ in range(30)])
# exec_time = evaluate_code_2(new_code)
print('New execution time:', new_exec_time / 1e9, 'ms')

speedup = old_exec_time / new_exec_time
print('Speedup:', speedup)


# print(new_code)