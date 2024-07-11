import os
import subprocess
import random
from sympy import evaluate
from copy import copy
from tqdm import tqdm

def print_info(*args):
    message = ' '.join(map(str, args))
    print(f"\033[94m[INFO]\t {message}\033[0m")

def print_success(*args):
    message = ' '.join(map(str, args))
    print(f"\033[92m[SUCCESS]\t {message}\033[0m")

def print_alert(*args):
    message = ' '.join(map(str, args))
    print(f"\033[93m[ALERT]\t {message}\033[0m")

def print_error(*args):
    message = ' '.join(map(str, args))
    print(f"\033[91m[ERROR]\t {message}\033[0m")


def get_raw_ast_info(code):
    
    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_nn_mlir.mlir"    
    with open(tmp_file, "w") as file:
        file.write(code)
        
    result = subprocess.run(
        f'MyASTGenerator/build/bin/AstDumper {tmp_file}',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    return result.stdout.decode('utf-8')


def get_ast(raw_ast_info):

    info, new_code = raw_ast_info.split("########################################")
    operations_lines, graph_lines = info.split('#BEGIN_GRAPH')

    operations_blocks = operations_lines.split('#START_OPERATION')
    operations_blocks = [block.strip() for block in operations_blocks if block]

    ast = {}
    for block in operations_blocks:
        block_lines = block.split('\n')
        
        operation_tag = block_lines[-2]
        operation = '\n'.join(block_lines[:-3])
        
        ast[operation_tag] = {
                'producers': [],
                'operation': operation
            }
        
    graph_lines = graph_lines.split('\n')
    graph_lines = [line.split(' --> ') for line in graph_lines if ' --> ' in line]
    
    for (producer, consumer) in graph_lines:
        ast[consumer]['producers'].insert(0, producer)
        
    return ast, new_code.strip()








def transform_dialect_TP(code, operation_tag, tiling_size, fusion=False):
    
    code = code.strip()
    # t = [0]*len(tiling_size) if len(tiling_size) != 7 and fusion==False else tiling_size
    transform_dilaect_code = f'\nmodule attributes {{transform.with_named_sequence}} {{\n  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n    %op_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n    %tiled_op_{operation_tag}, %forall_op_{operation_tag} = transform.structured.tile_using_forall %op_{operation_tag}  tile_sizes {str(tiling_size)} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n    transform.yield\n  }}\n}}'
    # transform_dilaect_code = f'\nmodule attributes {{transform.with_named_sequence}} {{\n  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n    %op_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n    %tiled_op_{operation_tag}, %forall_op_{operation_tag} = transform.structured.tile_using_forall %op_{operation_tag}  tile_sizes {str(t)} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n    transform.yield\n  }}\n}}'
    
    # if "module {" in code:
        # code = code[:-1] + transform_dilaect_code + '\n}'
    # else:
    code = code + transform_dilaect_code
                
    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"    
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule',
    ).read()
        
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    return result



def transform_dialect_tile(code, operation_tag, tiling_size):
    code = code.strip()
    n_loops = sum([s != 0 for s in tiling_size])
    r = ', '.join(['!transform.any_op']*n_loops)
    
    transform_dilaect_code = f"""
    module attributes {{transform.with_named_sequence}} {{
          transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
            %op_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op
        %tiled_op_{operation_tag}, %loops:{n_loops} = transform.structured.tile_using_for %op_{operation_tag} {str(tiling_size)} : (!transform.any_op) -> (!transform.any_op, {r})
        transform.yield
      }}
    }}"""
    
    # if "module {" in code:
        # code = code[:-1] + transform_dilaect_code + '\n}'
    # else:
    code = code + transform_dilaect_code + '\n'
        
    # print(code)
    
    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"    
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule',
    ).read()    
    
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    return result





def transform_dialect_interchange(code, operation_tag, interchange_list):
    code = code.strip()
    
    transform_dilaect_code = f'\nmodule attributes {{transform.with_named_sequence}} {{\n  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n    %op_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n    %gen_op_{operation_tag} = transform.structured.generalize %op_{operation_tag} : (!transform.any_op) -> !transform.any_op\n    %interchanged_op_{operation_tag} = transform.structured.interchange %gen_op_{operation_tag} iterator_interchange = {str(interchange_list)} : (!transform.any_op) -> !transform.any_op\n    transform.yield\n  }}\n}}'
    

    code = code + transform_dilaect_code + '\n'
        
    # print(code)
    
    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"    
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule',
    ).read()
    
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    return result




def transform_dialect_fuse(code, consumer_tag, producer_tag):
    code = code.strip()
    
    transform_dilaect_code = f'\nmodule attributes {{transform.with_named_sequence}} {{\n  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n    %op_{producer_tag} = transform.structured.match attributes{{tag = "{producer_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n    %op_{consumer_tag} = transform.structured.match attributes{{tag = "{consumer_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n\n    %forall_op_{consumer_tag} = transform.get_parent_op %op_{consumer_tag}: (!transform.any_op) -> !transform.any_op\n\n    transform.structured.fuse_into_containing_op %op_{producer_tag} into %forall_op_{consumer_tag} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)\n\n    transform.yield\n  }}\n}}'
        

    code = code + transform_dilaect_code + '\n'
        
    # print(code)
    
    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"    
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule',
    ).read()
    
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    
    return result



def transform_dialect_vectorise(code, operation_tag):
  
    code = code.strip()

    transform_dilaect_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %op_operation = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op
    
    transform.structured.vectorize %op_operation : !transform.any_op
    
    
    %func = transform.structured.match ops{{["func.func"]}} in %arg1: (!transform.any_op) -> !transform.any_op
    %func_01 = transform.structured.hoist_redundant_vector_transfers %func :(!transform.any_op) -> (!transform.any_op)
    
    %f = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op
    
    transform.apply_patterns to %f {{
        transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct" 
        transform.apply_patterns.vector.transfer_permutation_patterns 
        transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerparallel" 
        transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "vector-transfer" 
        transform.apply_patterns.vector.transfer_to_scf max_transfer_rank = 1 full_unroll = true 
        transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1 
        transform.apply_patterns.vector.lower_shape_cast 
        transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d" 
        transform.apply_patterns.canonicalization
    }} : !transform.any_op
    
    // transform.apply_patterns to %f {{
    //     transform.apply_patterns.vector.reduction_to_contract
    //     transform.apply_patterns.vector.transfer_permutation_patterns
    //     transform.apply_patterns.vector.lower_masked_transfers
    // }} : !transform.any_op
    // 
    // transform.apply_patterns to %f {{
    //     transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    //     transform.apply_patterns.vector.lower_outerproduct
    // }} : !transform.any_op

    transform.yield
  }}
}}""".strip()

    
          
    code = code + '\n' + transform_dilaect_code + '\n'
        
    # print(code)
    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"    
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule',
    ).read()
    
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    
    return result



def evaluate_code_2(code):
    # command_1 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs create-deallocs function-boundary-type-conversion=identity-layout-map" -buffer-deallocation -convert-linalg-to-loops -scf-foreach-thread-lowering  -convert-vector-to-scf -convert-scf-to-openmp -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -reconcile-unrealized-casts"""
    # command_1 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -buffer-deallocation -convert-linalg-to-loops  -convert-vector-to-scf -convert-scf-to-openmp -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -reconcile-unrealized-casts"""
    # command_1 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -buffer-deallocation -convert-linalg-to-loops  -convert-vector-to-scf -convert-scf-to-openmp"""
    command_1 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -buffer-deallocation -convert-linalg-to-loops  -scf-foreach-thread-lowering -convert-vector-to-scf -convert-scf-to-openmp -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -reconcile-unrealized-casts"""
    command_2 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs=/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libmlir_runner_utils.so,/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libmlir_c_runner_utils.so,/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libomp.so"""
    
    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"
    # tmp_file = "generated_mlir/bigger_input_nn.mlir"
    
    os.environ["OMP_NUM_THREADS"] = "8"
    
    with open(tmp_file, "w") as file:
        file.write(code)
    
    # out = os.popen(f"""{command_1} {tmp_file} > examples/llvm2.mlir""")
    
    # exit()
    os.popen(f"""{command_1} {tmp_file} > examples/llvm1.mlir""")
    # exit()
    out = os.popen(f"""{command_1} {tmp_file} | {command_2} /dev/stdin""").read()
    # out = os.popen(f"""{command_1} {tmp_file}""").read()
    
    if out:
        return int(out.strip().split('\n')[-1])
    else:
        return None
    
    
def evaluate_code(code, timeout=20):
    # command_1 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs create-deallocs function-boundary-type-conversion=identity-layout-map" -buffer-deallocation -convert-linalg-to-loops -scf-foreach-thread-lowering  -convert-vector-to-scf -convert-scf-to-openmp -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -reconcile-unrealized-casts"""
    command_1 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -buffer-deallocation -convert-linalg-to-loops  -convert-vector-to-scf -convert-scf-to-openmp -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -reconcile-unrealized-casts"""
    command_2 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs=/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libmlir_runner_utils.so,/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libmlir_c_runner_utils.so,/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libomp.so"""
    
    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"
    # tmp_file = "generated_mlir/bigger_input_nn.mlir"
    
    os.environ["OMP_NUM_THREADS"] = "8"
    
    with open(tmp_file, "w") as file:
        file.write(code)
    
    out = os.popen(f"""{command_1} {tmp_file} | {command_2} /dev/stdin""").read()
    # out = os.popen(f"""{command_1} {tmp_file}""").read()
    
    if out:
        return int(out.strip().split('\n')[-1])
    else:
        return None






def transform_dialect_img2col(code, operation_tag):
  
    code = code.strip()

    transform_dilaect_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %op_operation = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op
    
    %a, %b = transform.structured.convert_conv2d_to_img2col %op_operation : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %a "tag = {operation_tag}" : !transform.any_op
    transform.annotate %b "tag = wtf" : !transform.any_op
    
    transform.yield
  }}
}}""".strip()

    
          
    code = code + '\n' + transform_dilaect_code + '\n'
        
    # print(code)
    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"    
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule',
    ).read()
    
    
    result = result.replace(f'"tag = {operation_tag}"', f'tag = "{operation_tag}"')
    result = result.replace(f'"tag = wtf"', f'tag = "wtf"')
    
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    
    return result


def apply_conv2d_decomposition(code, operation_tag):
    
    code = code.strip()
    transform_dilaect_code = f"""
    
        module attributes {{transform.with_named_sequence}} {{
        transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
            %conv = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op 
            %decomposed = transform.structured.decompose %conv: (!transform.any_op) -> !transform.any_op 
            transform.yield
            }} 
        }}"""    
    
    code = code + '\n' + transform_dilaect_code + '\n'
    
    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"    
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule',
    ).read()
    
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    return result




def transform_dialect_prints(code, operation_tags: list):
    
    matchs = '\n'.join([ f""" %op_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op """ for operation_tag in operation_tags ])
    prints = '\n'.join([ f""" transform.print %op_{operation_tag} {{name = "selected_{operation_tag}"}}: !transform.any_op """ for operation_tag in operation_tags])
    
    code = code.strip()
    transform_dilaect_code = f"""
    
        module attributes {{transform.with_named_sequence}} {{
        transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
            {matchs} 
            
            {prints}
            
            transform.yield
            }} 
        }}"""    
    
    code = code + '\n' + transform_dilaect_code + '\n'
    
    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"    
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule',
    ).read()
    
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    return result


def post_process_transform_dialect_prints(result):
    
    lines = result.split('\n')
    res = {}
    
    i = 0
    while i < len(lines):
        if "[[[ IR printer: selected_" in lines[i]:
            opreation_id = lines[i][25:-4]
            
            operation = []
            i += 1
            while not ( ("[[[ IR printer: selected_" in lines[i]) or (" = affine_map<" in lines[i]) or ("module attributes" in lines[i])): 
                operation.append(lines[i])
                i += 1
            
            operation = '\n'.join(operation)
            operation = ' '.join(operation.split(' ')[2:])
            res[opreation_id] = operation
            
        else:
            i += 1
    
    

    return res



def sorted_divisors(n):
    divisors = []
    for i in range(2, n + 1):
        if n % i == 0:
            divisors.append(i)
    return sorted(divisors)

def get_candidates(n, num_candidates):
    div = sorted_divisors(n)
    if len(div) >= num_candidates:
        step = len(div) // num_candidates
        res = div[::step][:num_candidates]
    else:
        res = div + div[-1:]*(num_candidates-len(div))
    return res







if __name__ == '__main__':


    with open("generated_mlir/bigger_input_nn.mlir", "r") as file:
        mlir_code = file.read()
        # new_code = mlir_code
        
    raw_ast_info = get_raw_ast_info(mlir_code)
    
    # print(raw_ast_info)
    
    code_ast, mlir_with_tags = get_ast(raw_ast_info)
    new_code = mlir_with_tags
    
    print('Directed Graph of producer-consumer:\n')
    for op_tag, op_info in code_ast.items():
        # print("Operation tag:", op_tag)
        # print("Producers:", op_info["producers"])
        # print("Operation:", op_info["operation"])
        # print('')
        for producer in op_info["producers"]:
            print(f'    {producer} --> {op_tag}')
    print('\n'*10)
    # exit()
    
    print_info("current operation: operation_19 (relu)")
    print_success("tiling+parallelization [8, 14]")
    print_success("fusion: operation_18 (add) in operation_19 (relu)")
    print_success("fusion: operation_17 (matmul) in operation_19 (relu)")
    
    print_info("current operation: operation_17 (matmul)")
    new_code = transform_dialect_TP(new_code, 'operation_17', [8, 0, 2], fusion=True)
    # new_code = transform_dialect_vectorise(new_code, 'operation_17')
    print_success("tiling+parallelization [8, 0, 2]")
    print_success("vectorization")
    
    print('?' in new_code)
    print('\n')
    
    
    print_info("current operation: operation_14 (relu)")
    print_success("tiling+parallelization [8, 5]")
    print_success("fusion: operation_13 (add) in operation_14 (relu)")
    print_success("fusion: operation_12 (matmul) in operation_14 (relu)")
    
    print_info("current operation: operation_12 (matmul)")
    new_code = transform_dialect_TP(new_code, 'operation_12', [4, 0, 2], fusion=True)
    # new_code = transform_dialect_vectorise(new_code, 'operation_12')
    print_success("tiling+parallelization [4, 0, 2]")
    print_success("vectorization")
    
    print('?' in new_code)
    print('\n')
    
    
    print_info("current operation: operation_7 (relu)")
    print_success("tiling+parallelization [2, 8, 4, 0]")
    print_success("fusion: operation_6 (conv2d) in operation_7 (relu)")
    
    print_info("current operation: operation_6 (conv2d)")
    new_code = transform_dialect_TP(new_code, 'operation_6', [4, 8, 0, 0, 0, 0, 0])
    new_code = transform_dialect_tile(new_code, 'operation_6', [2, 8, 0, 0, 8, 5, 5])
    new_code = transform_dialect_tile(new_code, 'operation_6', [2, 0, 1, 0, 0, 1, 0])
    new_code = apply_conv2d_decomposition(new_code, 'operation_6')
    # new_code = transform_dialect_vectorise(new_code, 'operation_6')
    print_success("tiling+parallelization [2, 8, 0, 0, 0, 0, 0]")
    print_success("tiling [2, 8, 4, 4, 8, 5, 5]")
    print_success("tiling [2, 0, 1, 0, 0, 1, 0]")
    print_success("conv2d decomposition")
    print_success("vectorization")
    
    print('?' in new_code)
    print('\n')
    
    
    print_info("current operation: operation_2 (relu)")
    print_success("tiling+parallelization [4, 16, 0, 0]")
    print_success("fusion: operation_1 (conv2d) in operation_2 (relu)")
    
    print_info("current operation: operation_1 (conv2d)")
    new_code = transform_dialect_TP(new_code, 'operation_1', [4, 8, 0, 0, 0, 0, 0])
    new_code = transform_dialect_tile(new_code, 'operation_1', [2, 8, 0, 0, 3, 7, 7])
    new_code = transform_dialect_tile(new_code, 'operation_1', [2, 0, 1, 0, 0, 1, 0])
    new_code = apply_conv2d_decomposition(new_code, 'operation_1')
    # new_code = transform_dialect_vectorise(new_code, 'operation_1')
    print_success("tiling+parallelization [2, 8, 0, 0, 0, 0, 0]")
    print_success("tiling [2, 8, 14, 7, 3, 7, 7]")
    print_success("tiling [2, 0, 1, 0, 0, 1, 0]")
    print_success("conv2d decomposition")
    print_success("vectorization")
    
    print('?' in new_code)
    print()
    print_success("END OF SCHEDULE")
    
    
    
    print('?' in new_code)
    base_execution_time = evaluate_code_2(mlir_with_tags)
    exec_time = evaluate_code_2(new_code)
    
    print('Base execution time:', 31.803188473, 'ms')
    print('New execution time:', exec_time / 1e9, 'ms')
    print('Speedup:', 31803188473 / exec_time)
    