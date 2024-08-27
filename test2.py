from fusion_utils.transforms import (
    evaluate_code_2,
    transform_dialect_TP,
    transform_dialect_tile,
    transform_dialect_vectorise,
    transform_dialect_vectorise_with_backend,
    transform_dialect_vectorise_whole_thing,
    transform_dialect_vectorise_img2col,
    transform_dialect_img2col,
    transform_dialect_fuse,
    get_raw_ast_info, get_ast,
    transform_dialect_vectorise_,
    apply_conv2d_decomposition,
    
)

from utils.transform_utils import evaluate_code_with_timeout
from utils.observation_utils import function_wrapper, lower_linalg_to_loops

tmp_file = 'examples/temp_nn_mlir.mlir'


with open('generated_mlir/resnet18_with_tags.mlir', 'r') as file:
    code = file.read()



s = [
    (7, [8, 2, 2]),
    (22, [8, 2, 0]),
    (25, [0, 2, 0]),
    (27, [2, 2, 0]),
    (31, [32, 2, 0]),
    (42, [32, 2, 0]),
    (44, [2, 2, 0]),
    (56, [32, 7, 0]),
]


# for i, param in s:
for i in range(77):
    if i in [7, 22, 25, 27, 31, 42, 44, 56]:
        code = transform_dialect_img2col(code, f"operation_{i}", tmp_file)
        code = transform_dialect_TP(code, f"operation_{i}", [2, 0, 0], tmp_file)
    # break
    else:
        code = transform_dialect_tile(code, f"operation_{i}", [1, 2], tmp_file)
    
    print(i, '?' in code)


with open('generated_mlir/resnet18_test.mlir', 'w') as file:
    file.write(code)

code = transform_dialect_vectorise_whole_thing(code, tmp_file)

# print(code)
print(len(code))


exit()


res = []
for _ in range(30):
  exec = evaluate_code_with_timeout(code, 6000, tmp_file)
  print(exec*1e-9)
  res.append(exec)

print(sum(res)/len(res))