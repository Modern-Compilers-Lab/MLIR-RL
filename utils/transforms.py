import os
import subprocess
import random
from sympy import evaluate
from copy import copy
from tqdm import tqdm
import multiprocessing




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


def get_raw_ast_info(code, tmp_file):
    
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








def transform_dialect_TP(code, operation_tag, tiling_size, tmp_file):
    
    code = code.strip()
    transform_dilaect_code = f'\nmodule attributes {{transform.with_named_sequence}} {{\n  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n    %op_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n    %tiled_op_{operation_tag}, %forall_op_{operation_tag} = transform.structured.tile_using_forall %op_{operation_tag}  tile_sizes {str(tiling_size)} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n    transform.yield\n  }}\n}}'
    
    code = code + transform_dilaect_code
                
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule',
    ).read()
        
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    return result



def transform_dialect_tile(code, operation_tag, tiling_size, tmp_file):
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
    
    code = code + transform_dilaect_code + '\n'

    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule',
    ).read()    
    
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    return result





def transform_dialect_interchange(code, operation_tag, interchange_list, tmp_file):
    code = code.strip()
    
    transform_dilaect_code = f"""
    module attributes {{transform.with_named_sequence}} {{
          transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
            %op_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op
        %gen_op_{operation_tag} = transform.structured.generalize %op_{operation_tag} : (!transform.any_op) -> !transform.any_op
        %interchanged_op = transform.structured.interchange %gen_op_{operation_tag} iterator_interchange = {str(interchange_list)} : (!transform.any_op) -> !transform.any_op
        transform.annotate %interchanged_op "XXXXX" : !transform.any_op
        transform.yield
      }}\n}}"""

    code = code + transform_dilaect_code + '\n'

    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule',
    ).read()
    
    if not f'tag = "{operation_tag}"' in result:
        result = result.replace("XXXXX", f'tag = "{operation_tag}"')
    else:
        result = result.replace("XXXXX, ", "")
    
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    return result




def transform_dialect_fuse(code, consumer_tag, producer_tag, tmp_file):
    code = code.strip()
    
    transform_dilaect_code = f'\nmodule attributes {{transform.with_named_sequence}} {{\n  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n    %op_{producer_tag} = transform.structured.match attributes{{tag = "{producer_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n    %op_{consumer_tag} = transform.structured.match attributes{{tag = "{consumer_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n\n    %forall_op_{consumer_tag} = transform.get_parent_op %op_{consumer_tag}: (!transform.any_op) -> !transform.any_op\n\n    transform.structured.fuse_into_containing_op %op_{producer_tag} into %forall_op_{consumer_tag} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)\n\n    transform.yield\n  }}\n}}'
        

    code = code + transform_dilaect_code + '\n'
  
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -test-transform-dialect-erase-schedule',
    ).read()
    
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    
    return result



def transform_dialect_vectorise_(code, operation_tag, tmp_file):
  
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
 
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule',
    ).read()
    
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    
    return result



def transform_dialect_vectorise_whole_thing(code, tmp_file):
  
    code = code.strip()

    transform_dilaect_code = f"""
module attributes {{transform.with_named_sequence}} {{
transform.named_sequence @__transform_main(%variant_op: !transform.any_op {{transform.readonly}}) 
{{
  
   %func = transform.structured.match ops{{["func.func"]}} in %variant_op
   : (!transform.any_op) -> !transform.any_op
  %func_0 = transform.structured.vectorize_children_and_apply_patterns %func {{vectorize_padding}}
    : (!transform.any_op) -> (!transform.any_op)

       // Step 4. Vector backend
  // ======================================================
  %f = transform.structured.match ops{{["func.func"]}} in %variant_op
    : (!transform.any_op) -> !transform.any_op

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
  
  

  transform.yield
}}
}}
""".strip()

    
          
    code = code + '\n' + transform_dilaect_code + '\n'

    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule',
    ).read()
    
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    
    return result
 


def transform_dialect_vectorise_img2col(code, operation_tag, tmp_file):
  
    code = code.strip()

    transform_dilaect_code = f"""
module attributes {{transform.with_named_sequence}} {{
transform.named_sequence @__transform_main(%variant_op: !transform.any_op {{transform.readonly}}) 
{{
    
  // %conv_gen_2 = transform.structured.match attributes{{tag = "{operation_tag}"}} in %variant_op : (!transform.any_op) -> !transform.any_op
  // %forall_op = transform.get_parent_op %conv_gen_2: (!transform.any_op) -> !transform.any_op
  
  %forall_op = transform.structured.match ops{{["scf.forall"]}}  in %variant_op : (!transform.any_op) -> !transform.any_op
  
  
  
  %producer = transform.structured.match attributes{{tag = "img2col_producer"}} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.structured.fuse_into_containing_op %producer into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  %fb = transform.structured.match ops{{["func.func"]}} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %fb {{
    transform.apply_patterns.canonicalization
  }} : !transform.any_op
  transform.apply_cse to %fb : !transform.any_op
  
  
  %original_fill = transform.structured.match ops{{["linalg.fill"]}} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.structured.fuse_into_containing_op %original_fill into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  
  %fb1 = transform.structured.match ops{{["func.func"]}} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %fb1 {{
    transform.apply_patterns.canonicalization
  }} : !transform.any_op
  transform.apply_cse to %fb1 : !transform.any_op
  
  
  
   %func = transform.structured.match ops{{["func.func"]}} in %variant_op
   : (!transform.any_op) -> !transform.any_op
  %func_0 = transform.structured.vectorize_children_and_apply_patterns %func {{vectorize_padding}}
    : (!transform.any_op) -> (!transform.any_op)

       // Step 4. Vector backend
  // ======================================================
  %f = transform.structured.match ops{{["func.func"]}} in %variant_op
    : (!transform.any_op) -> !transform.any_op

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
  
  

  transform.yield
}}
}}
""".strip()

    
          
    code = code + '\n' + transform_dilaect_code + '\n'
  
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule',
    ).read()
    
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    
    return result
 
def transform_dialect_vectorise(code, operation_tag, tmp_file):
  
    code = code.strip()

    transform_dilaect_code = f"""
module attributes {{transform.with_named_sequence}} {{
transform.named_sequence @__transform_main(%variant_op: !transform.any_op {{transform.readonly}}) 
{{
    
  // %conv_gen_2 = transform.structured.match attributes{{tag = "{operation_tag}"}} in %variant_op : (!transform.any_op) -> !transform.any_op
  // %forall_op = transform.get_parent_op %conv_gen_2: (!transform.any_op) -> !transform.any_op
  
  %forall_op = transform.structured.match ops{{["scf.forall"]}}  in %variant_op : (!transform.any_op) -> !transform.any_op
  
  
  %original_fill = transform.structured.match ops{{["linalg.fill"]}} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.structured.fuse_into_containing_op %original_fill into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  
  %func = transform.structured.match ops{{["func.func"]}} in %variant_op: (!transform.any_op) -> !transform.any_op
  %func_0 = transform.structured.vectorize_children_and_apply_patterns %func {{vectorize_padding}}: (!transform.any_op) -> (!transform.any_op)

  transform.yield
}}
}}
""".strip()

    
          
    code = code + '\n' + transform_dilaect_code + '\n'
        
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule',
    ).read()
    
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    
    return result


def transform_dialect_vectorise_with_backend(code, operation_tag, tmp_file):
  
    code = code.strip()

    transform_dilaect_code = f"""
module attributes {{transform.with_named_sequence}} {{
transform.named_sequence @__transform_main(%variant_op: !transform.any_op {{transform.readonly}}) 
{{
    
  // %conv_gen_2 = transform.structured.match attributes{{tag = "{operation_tag}"}} in %variant_op : (!transform.any_op) -> !transform.any_op
  // %forall_op = transform.get_parent_op %conv_gen_2: (!transform.any_op) -> !transform.any_op
  
  %forall_op = transform.structured.match ops{{["scf.forall"]}}  in %variant_op : (!transform.any_op) -> !transform.any_op
  
  
  %original_fill = transform.structured.match ops{{["linalg.fill"]}} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.structured.fuse_into_containing_op %original_fill into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  
  %func = transform.structured.match ops{{["func.func"]}} in %variant_op: (!transform.any_op) -> !transform.any_op
  %func_0 = transform.structured.vectorize_children_and_apply_patterns %func {{vectorize_padding}}: (!transform.any_op) -> (!transform.any_op)

  // Step 4. Vector backend
  // ======================================================
  %f = transform.structured.match ops{{["func.func"]}} in %variant_op
    : (!transform.any_op) -> !transform.any_op

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

  transform.yield
}}
}}
""".strip()

    
          
    code = code + '\n' + transform_dilaect_code + '\n'
        
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule',
    ).read()
    
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    
    return result




def evaluate_code_2(code, tmp_file):
    command_1 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -buffer-deallocation -convert-linalg-to-loops  -scf-foreach-thread-lowering -convert-vector-to-scf -convert-scf-to-openmp -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -reconcile-unrealized-casts"""
    command_2 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs=/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libmlir_runner_utils.so,/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libmlir_c_runner_utils.so,/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libomp.so"""
        
    os.environ["OMP_NUM_THREADS"] = "8"
    
    with open(tmp_file, "w") as file:
        file.write(code)
    

    os.popen(f"""{command_1} {tmp_file} > examples/llvm1.mlir""")
    out = os.popen(f"""{command_1} {tmp_file} | {command_2} /dev/stdin""").read()
    
    if out:
        return int(out.strip().split('\n')[-1])
    else:
        return None
    
    




def transform_dialect_img2col(code, operation_tag, tmp_file):
  
    code = code.strip()
    
    transform_dilaect_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %op_operation = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op
    
    %a, %b = transform.structured.convert_conv2d_to_img2col %op_operation : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    transform.annotate %a "AAAAA" : !transform.any_op
    
    %matmul_op = transform.get_producer_of_operand %b[0]: (!transform.any_op) -> !transform.any_op
    transform.annotate %matmul_op "XXXXX" : !transform.any_op
    
    transform.yield
  }}
}}""".strip()

    
          
    code = code + transform_dilaect_code
                
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -test-transform-dialect-erase-schedule',
    ).read()
    
    
    if not f'tag = "{operation_tag}"' in result:
        result = result.replace("XXXXX", f'tag = "{operation_tag}"')
    else:
        result = result.replace("XXXXX, ", "")
        
    if not f'tag = "img2col_producer"' in result:
        result = result.replace("AAAAA", f'tag = "img2col_producer"')
    else:
        result = result.replace("AAAAA, ", "")
    
    
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    
    return result


def apply_conv2d_decomposition(code, operation_tag, tmp_file):
    
    code = code.strip()
    transform_dilaect_code = f"""
    
        module attributes {{transform.with_named_sequence}} {{
        transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
            %conv = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op 
            %decomposed = transform.structured.decompose %conv: (!transform.any_op) -> !transform.any_op
            
            transform.annotate %decomposed "XXXXX" : !transform.any_op
            
            
            transform.yield
            }} 
        }}"""    
    
    code = code + '\n' + transform_dilaect_code + '\n'
    
        
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule',
    ).read()
    
    if not f'tag = "{operation_tag}"' in result:
        result = result.replace("XXXXX", f'tag = "{operation_tag}"')
    else:
        result = result.replace("XXXXX, ", "")
    result = result.replace("module {\n", "")
    result = result.replace("\n}\n", "")
    result = result.replace("module attributes {transform.with_named_sequence} {\n  }", "")
    
    return result




def transform_dialect_prints(code, operation_tags: list, tmp_file):
    
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







def apply_transformation(state, code, transformation, parameters):
    
    tmp_file = state.tmp_file
    
    code = code.strip()
    code = code.replace("module {\n", "")
    
    if transformation == 'tiling':
        new_code = transform_dialect_tile(code, state.operation_tag, parameters, tmp_file)
    elif transformation == 'parallelization':
        new_code = transform_dialect_TP(code, state.operation_tag, parameters, tmp_file)
    elif transformation == 'interchange':
        new_code = transform_dialect_interchange(code, state.operation_tag, parameters, tmp_file)
    elif transformation == 'img2col':
        new_code = transform_dialect_img2col(code, state.operation_tag, tmp_file)
    elif transformation == 'vectorization':
        if state.operation_type == 'conv_2d+img2col':
            new_code = transform_dialect_vectorise_img2col(code, state.operation_tag, tmp_file)
        else:
            new_code = transform_dialect_vectorise(code, state.operation_tag, tmp_file)
    else:
        raise ValueError

    return new_code

def apply_transformation_wrapper(state, code, transformation, parameters, return_list):
    res = apply_transformation(state, code, transformation, parameters)
    return_list.append(res)
    
def apply_transformation_with_timeout(state, code, transformation, parameters, timeout):
    
    manager = multiprocessing.Manager()
    return_list = manager.list()
    process = multiprocessing.Process(target=apply_transformation_wrapper, args=(state, code, transformation, parameters, return_list))
    process.start()
    process.join(timeout)

    if process.is_alive():
        # The function is still running, terminate the process
        process.terminate()
        process.join()

        return None
    else:
        # The function completed within the timeout
        return return_list[0]







def evaluate_code(code, tmp_file):
    command_1 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -buffer-deallocation -convert-linalg-to-loops  -scf-foreach-thread-lowering -convert-vector-to-scf -convert-scf-to-openmp -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -reconcile-unrealized-casts"""
    command_2 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs=/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libmlir_runner_utils.so,/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libmlir_c_runner_utils.so,/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libomp.so"""
        
    os.environ["OMP_NUM_THREADS"] = "8"
    
    with open(tmp_file, "w") as file:
        file.write(code)
        
    out = os.popen(f"""{command_1} {tmp_file} | {command_2} /dev/stdin""").read()
        
    if out:
        return int(out.strip().split('\n')[-1])
    else:
        return None

def evaluate_code_wrapper(code, return_list, tmp_file):
    res = evaluate_code(code, tmp_file)
    return_list.append(res)

def evaluate_code_with_timeout(code, timeout, tmp_file):
    manager = multiprocessing.Manager()
    return_list = manager.list()
    process = multiprocessing.Process(target=evaluate_code_wrapper, args=(code, return_list, tmp_file))
    process.start()
    process.join(timeout)

    if process.is_alive():
        # The function is still running, terminate the process
        process.terminate()
        process.join()

        return None
    else:
        # The function completed within the timeout
        return return_list[0]