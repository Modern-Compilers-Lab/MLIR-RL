# pylint: skip-file


from utils.consts import EXECUTABLE_PATH
import subprocess
import sys
import ctypes
import os

sys.path.append('./llvm-project/build/tools/mlir/python_packages/mlir_core')
from mlir.passmanager import PassManager
from mlir.execution_engine import ExecutionEngine
from mlir.ir import Context, Module


import ctypes
import multiprocessing
import time

context = Context()




# def apply_transformation(code, transformation, parameters):

#     code = code.replace('"', '\\"')

#     transformation_id = {
#         "interchange":0,
#         "tiling":1,
#         "parallelization":2,
#         "vectorization":3
#     }

#     if isinstance(transformation, str):
#         transformation = transformation_id[transformation]

#     parameters = ','.join(map(str,parameters))

#     result = subprocess.run(
#         f'echo "{code}" | {EXECUTABLE_PATH} /dev/stdin {transformation} {parameters}',
#         shell=True,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE
#     )

#     return result.stdout.decode('utf-8')


def apply_transformation(code, transformation, parameters):

    code = code.replace('"', '\\"')

    transformation_id = {
        "interchange":0,
        "tiling":1,
        "parallelization":2,
        "vectorization":3
    }

    if isinstance(transformation, str):
        transformation = transformation_id[transformation]

    parameters = ','.join(map(str,parameters))

    result = subprocess.run(
        f'echo "{code}" | {EXECUTABLE_PATH} /dev/stdin {transformation} {parameters}',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    return result.stdout.decode('utf-8')


def apply_conv_2d_nhwc_hwcf_decomposition(code):
    code = code.strip()
    code = code[:-1] + "\ntransform.sequence failures(propagate) { ^bb1(%variant_op: !transform.any_op): %conv = transform.structured.match ops{[\"linalg.conv_2d_nhwc_hwcf\"]} in %variant_op : (!transform.any_op) -> !transform.any_op  %decomposed = transform.structured.decompose %conv: (!transform.any_op) -> !transform.any_op} \n }"    
    
    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"    
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} --test-transform-dialect-interpreter --test-transform-dialect-erase-schedule',
    ).read()
    
    return result

def apply_conv_2d_nchw_fchw_decomposition(code):
    code = code.strip()
    code = code[:-1] + "\ntransform.sequence failures(propagate) { ^bb1(%variant_op: !transform.any_op): %conv = transform.structured.match ops{[\"linalg.conv_2d_nchw_fchw\"]} in %variant_op : (!transform.any_op) -> !transform.any_op  %decomposed = transform.structured.decompose %conv: (!transform.any_op) -> !transform.any_op} \n }"    
    
    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"    
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} --test-transform-dialect-interpreter --test-transform-dialect-erase-schedule',
    ).read()
    
    return result

def apply_conv_2d_nhwc_fhwc_decomposition(code):
    code = code.strip()
    code = code[:-1] + "\ntransform.sequence failures(propagate) { ^bb1(%variant_op: !transform.any_op): %conv = transform.structured.match ops{[\"linalg.conv_2d_nhwc_fhwc\"]} in %variant_op : (!transform.any_op) -> !transform.any_op  %decomposed = transform.structured.decompose %conv: (!transform.any_op) -> !transform.any_op} \n }"    
    
    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"    
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} --test-transform-dialect-interpreter --test-transform-dialect-erase-schedule',
    ).read()
    
    return result


def apply_maxpool_decomposition(code):
    code = code.strip()
    code = code[:-1] + "\ntransform.sequence failures(propagate) { ^bb1(%variant_op: !transform.any_op): %conv = transform.structured.match ops{[\"linalg.pooling_nchw_max\"]} in %variant_op : (!transform.any_op) -> !transform.any_op  %decomposed = transform.structured.decompose %conv: (!transform.any_op) -> !transform.any_op} \n }"    
    
    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"    
    with open(tmp_file, "w") as file:
        file.write(code)
    
    result = os.popen(
        f'/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt {tmp_file} --test-transform-dialect-interpreter --test-transform-dialect-erase-schedule',
    ).read()
    
    return result


def evaluate_code(code, timeout=20):
    command_1 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -buffer-deallocation -convert-linalg-to-loops  -scf-foreach-thread-lowering -convert-vector-to-scf -convert-scf-to-openmp -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -reconcile-unrealized-casts"""
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

def evaluate_code_wrapper(code, return_list, state):
    res = evaluate_code(code)
    # if res:res /= 21
    # if state:
        # if 'conv' in state.operation_id:res /= 54
    return_list.append(res)

def evaluate_code_with_timeout(code, timeout, state=None):
    manager = multiprocessing.Manager()
    return_list = manager.list()
    process = multiprocessing.Process(target=evaluate_code_wrapper, args=(code, return_list, state))
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





    
def apply_transformation_wrapper(code, transformation, parameters, return_list):
    res = apply_transformation(code, transformation, parameters)
    return_list.append(res)
    
def apply_transformation_with_timeout(code, transformation, parameters, timeout):
    manager = multiprocessing.Manager()
    return_list = manager.list()
    process = multiprocessing.Process(target=apply_transformation_wrapper, args=(code, transformation, parameters, return_list))
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


# /scratch/nb3891/Script/tmp_llvm/llvm-project/build/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs=/scratch/nb3891/Script/tmp_llvm/llvm-project/build/lib/libmlir_runner_utils.so,/scratch/nb3891/Script/tmp_llvm/llvm-project/build/lib/libmlir_c_runner_utils.so,/scratch/nb3891/Script/tmp_llvm/llvm-project/build/lib/libomp.so