from rl_autoschedular import config as cfg
from rl_autoschedular.state import OperationFeatures, NestedLoopFeatures
import random
import re
import math
import os
import sys
import json
from tqdm import trange
import numpy as np
from mlir.ir import Context, Module
from mlir.execution_engine import ExecutionEngine, ctypes
from mlir.runtime import get_ranked_memref_descriptor
from mlir.passmanager import PassManager


output_dir = 'data/features'
inputs: dict[str, np.ndarray] = {}


pass_pipeline = """builtin.module(
    loop-invariant-code-motion,
    canonicalize,
    convert-vector-to-scf,
    convert-linalg-to-loops,
    buffer-deallocation-pipeline,
    convert-bufferization-to-memref,
    scf-forall-to-parallel,
    convert-scf-to-openmp,
    expand-strided-metadata,
    finalize-memref-to-llvm,
    convert-scf-to-cf,
    lower-affine,

    convert-openmp-to-llvm,
    convert-vector-to-llvm,
    convert-math-to-llvm,
    convert-func-to-llvm,
    convert-index-to-llvm,
    convert-arith-to-llvm,
    convert-cf-to-llvm,

    reconcile-unrealized-casts,
    canonicalize,
    cse
)"""


def gen_features() -> OperationFeatures:
    # Nested loops
    num_loops = random.randint(1, cfg.max_num_loops)
    reduction_count = random.randint(0, min(3, num_loops - 1))
    iterator_types = ['parallel'] * (num_loops - reduction_count) + ['reduction'] * reduction_count
    max_iterations = 10 ** 9
    max_per_loop = math.ceil(max_iterations ** (1 / num_loops))
    iterations = max_iterations
    while iterations >= max_iterations:
        iterations = 1
        upper_bounds = []
        for _ in range(num_loops):
            upper_bound = random.randint(2, min(max_per_loop * 2, 4096))
            upper_bounds.append(upper_bound)
            iterations *= upper_bound
    nested_loops = [
        NestedLoopFeatures(
            arg=f'd{i}',
            lower_bound=0,
            upper_bound=upper_bounds[i],
            step=1,
            iterator_type=iterator_types[i]
        )
        for i in range(num_loops)
    ]

    # Operators count
    total_op_count = 0
    while total_op_count == 0:
        op_count = {
            '+': random.randint(0, 10),
            '-': random.randint(0, 10),
            '*': random.randint(0, 10),
            '/': 0,  # TODO: Figure out how to handle division
            'exp': random.randint(0, 2),
        }
        total_op_count = sum(op_count.values())

    # Load data
    max_load_size = 2 ** 24
    num_loads = random.randint(1, cfg.max_num_stores_loads)
    load_data: list[list[str]] = []
    per_loop = max(math.ceil(iterations ** (1 / num_loops)), 2)
    max_dim = math.ceil(math.log(max_load_size) / math.log(per_loop))
    args_dict = {loop.arg: loop.upper_bound for loop in nested_loops}
    unseen_args = set(args_dict.keys())
    for _ in range(num_loads - 1):
        load_size = max_load_size
        while load_size >= max_load_size:
            dims_count = random.randint(1, min(cfg.max_num_load_store_dim, max_dim))
            zeros_count = random.randint(max(0, dims_count - num_loops), dims_count)
            load_args = random.sample(list(args_dict.keys()) + ['0'], dims_count, counts=[1] * num_loops + [zeros_count])
            load_size = 1
            for arg in load_args:
                if arg == '0':
                    load_size *= 5
                else:
                    load_size *= args_dict[arg]
        load_data.append(load_args)
        for arg in load_args:
            unseen_args.discard(arg)
    if unseen_args:
        load_data.append(list(unseen_args))

    # Store data
    p_args = [loop.arg for loop in nested_loops if loop.iterator_type == 'parallel']
    random.shuffle(p_args)
    store_data = p_args

    return OperationFeatures(
        raw_operation='',
        operation_type='generic',
        op_count=op_count,
        load_data=load_data,
        store_data=store_data,
        nested_loops=nested_loops,
        vectorizable=True
    )


def create_params(op_features: OperationFeatures) -> tuple[list[str], list[str]]:
    params = []
    shapes = []
    args_dict = {loop.arg: loop.upper_bound for loop in op_features.nested_loops}

    # Load params
    for i, load in enumerate(op_features.load_data):
        shape: list[int] = []
        for arg in load:
            if arg == '0':
                shape.append(random.randint(1, 5))
                continue
            shape.append(args_dict[arg])
        # inputs[f'arg{i}'] = np.random.rand(*shape) * 100
        inputs[f'arg{i}'] = np.empty(shape)
        params.append(f'%arg{i}')
        shapes.append(f"memref<{'x'.join(map(str, shape))}xf64>")

    # Store param
    shape = []
    for arg in op_features.store_data:
        if arg == '0':
            shape.append(random.randint(1, 5))
            continue
        shape.append(args_dict[arg])
    # inputs[f'arg{len(params)}'] = np.zeros(shape)
    inputs[f'arg{len(params)}'] = np.empty(shape)
    params.append(f'%arg{len(params)}')
    shapes.append(f"memref<{'x'.join(map(str, shape))}xf64>")

    return params, shapes


def create_raw_operation(op_features: OperationFeatures, params: list[str], shapes: list[str]) -> str:
    # Affine maps
    base_dims = ', '.join([loop.arg for loop in op_features.nested_loops])
    affine_maps = []
    for load in op_features.load_data:
        affine_maps.append(f"affine_map<({base_dims}) -> ({', '.join(load)})>")
    affine_maps.append(f"affine_map<({base_dims}) -> ({', '.join(op_features.store_data)})>")
    affine_maps_attr = f"[{', '.join(affine_maps)}]"

    # Iterators
    iterators = ', '.join([f'"{loop.iterator_type}"' for loop in op_features.nested_loops])
    iterators_attr = f'[{iterators}]'

    # Inputs / Outputs
    ins = f"ins({', '.join(params[:-1])}: {', '.join(shapes[:-1])})"
    outs = f"outs({params[-1]}: {shapes[-1]})"

    code = f"linalg.generic {{indexing_maps={affine_maps_attr}, iterator_types={iterators_attr}}} {ins} {outs} {{\n"
    block_args = [f"%in_{i}: f64" for i in range(len(op_features.load_data))] + ["%out: f64"]
    code += f"^bb0({', '.join(block_args)}):\n"

    # Linalg body
    block_params = [arg.split(':')[0] for arg in block_args]
    unused_block_params = set(block_params.copy())
    created_args: set[str] = set()
    tmp_count = 0
    op_count_copy = {op: count for op, count in op_features.op_count.items() if count > 0}
    assert all(op_count_copy.values())
    total_op_count = sum(op_count_copy.values())
    for _ in range(total_op_count):
        op = random.choice(list(op_count_copy.keys()))
        if op == 'exp':
            if len(unused_block_params) > 0:
                operands = random.sample(list(unused_block_params), 1)
                unused_block_params.difference_update(operands)
            else:
                operands = random.sample(list(created_args) + block_params, 1)
        else:
            if len(unused_block_params) > 1:
                operands = random.sample(list(unused_block_params), 2)
                unused_block_params.difference_update(operands)
            elif len(unused_block_params) == 1:
                operands = [unused_block_params.pop()]
                unused_block_params = set()
                operands += random.sample(list(created_args) + block_params, 1)
            else:
                operands = random.sample(list(created_args) + block_params, 2)

        result = f"%{tmp_count}"
        tmp_count += 1
        created_args.add(result)
        match op:
            case '+':
                code += f"{result} = arith.addf {operands[0]}, {operands[1]} fastmath<nnan, ninf, nsz, reassoc, contract, afn> : f64\n"
            case '-':
                code += f"{result} = arith.subf {operands[0]}, {operands[1]} fastmath<nnan, ninf, nsz, reassoc, contract, afn> : f64\n"
            case '*':
                code += f"{result} = arith.mulf {operands[0]}, {operands[1]} fastmath<nnan, ninf, nsz, reassoc, contract, afn> : f64\n"
            case '/':
                code += f"{result} = arith.divf {operands[0]}, {operands[1]} fastmath<nnan, ninf, nsz, reassoc, contract, afn> : f64\n"
            case 'exp':
                code += f"{result} = math.exp {operands[0]} fastmath<nnan, ninf, nsz, reassoc, contract, afn> : f64\n"

        op_count_copy[op] -= 1
        if op_count_copy[op] == 0:
            del op_count_copy[op]

    assert sum(op_count_copy.values()) == 0

    code += f"linalg.yield {result} : f64\n"
    code += "}\n"

    return code


def formatMLIRCode(code: str) -> str:
    """Util function that format the MLIR code by adding indents.

    Args:
        code (str): the MLIR code

    Returns:
        str: the formatted MLIR code
    """
    lines = re.sub(r'\n+', '\n', code).split('\n')
    result = ''
    indent = 0
    for line in lines:
        if len(line) > 0:
            if line[0] == '}':
                if indent > 0:
                    indent -= 1
                else:
                    indent = 0

        result += indent * '  ' + line + '\n'

        if len(line) > 0:
            if line[-1] == '{':
                indent += 1

    return result


def gen_full_code() -> str:
    op_features = gen_features()

    params, shapes = create_params(op_features)
    main_params = [f'{param}: {shape}' for param, shape in zip(params, shapes)]

    raw_operation = create_raw_operation(op_features, params, shapes)

    code = (
        f'func.func private @nanoTime() -> i64 attributes {{ llvm.emit_c_interface }}\n'
        f'func.func @main({", ".join(main_params)}) -> i64 attributes {{ llvm.emit_c_interface }} {{\n'
        f'%t0 = func.call @nanoTime() : () -> i64\n'
        f'{raw_operation}\n'
        f'%t1 = func.call @nanoTime() : () -> i64\n'
        f'%t2 = arith.subi %t1, %t0 : i64\n'
        f'return %t2 : i64\n'
        f'}}\n'
    )

    code = formatMLIRCode(code)

    return code


if __name__ == '__main__':
    with open('execution_times.json', 'r') as file:
        execution_times: dict[str, int] = json.load(file)
    last_count = max([int(k.split('_')[-1]) for k in execution_times.keys()]) + 1
    for i in trange(last_count, 10000, desc='Generating benchmarks', unit='bench'):
        bench_generated = False
        while not bench_generated:
            bench_name = f'generic_{i}'
            bench_output = os.path.join(output_dir, f"{bench_name}.mlir")

            inputs = {}
            code = gen_full_code()
            with Context():
                module = Module.parse(code)
                pm = PassManager.parse(pass_pipeline)
                pm.run(module.operation)
            execution_engine = ExecutionEngine(
                module,
                shared_libs=os.getenv("MLIR_SHARED_LIBS", "").split(","),
            )
            arg_names = sorted(inputs.keys())
            # np.savez(f"{bench_output}.npz", **inputs)

            c_args = []
            for arg_name in arg_names:
                c_args.append(ctypes.pointer(ctypes.pointer(
                    get_ranked_memref_descriptor(inputs[arg_name])
                )))
            delta_arg = (ctypes.c_int64 * 1)(0)
            c_args.append(delta_arg)

            try:
                execution_engine.invoke("main", *c_args)
                execution_engine.invoke("main", *c_args)
            except Exception as e:
                print(f"Failed, Bench: {bench_name}, error: {e}", file=sys.stderr)
                # os.remove(f'{bench_output}.npz')
                continue

            exec_time = delta_arg[0]
            if exec_time >= (1 * 10**9):
                # os.remove(f'{bench_output}.npz')
                continue

            with open(bench_output, 'w') as f:
                f.write(code)
            # expected = inputs[arg_names[-1]]
            # np.save(f"{bench_output}.npy", expected)

            execution_times[bench_name] = exec_time
            with open('execution_times.json', 'w') as file:
                json.dump(execution_times, file, indent=4)

            bench_generated = True
