"""
wrap_mlir.py
------------
Utilities for wrapping existing MLIR model files with a timed @main function,
making them ready for benchmarking via the MLIR execution engine.

Two wrapper variants:
  - nn_transform_wrapper()         — allocates tensors internally; @main has no
                                     args, prints timing via @printI64 (cmd backend).
  - nn_transform_wrapper_binding() — @main receives tensors as args and *returns*
                                     the timing as i64 (bindings backend).

main_wrapper() is the file-level helper that reads a model .mlir file, locates
the model forward function, and appends the appropriate wrapper.
"""

import re


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _remove_duplicate_args(args: list[str], shapes: list[str]):
    """Remove duplicate (arg, shape) pairs while preserving order."""
    seen = set()
    result = []
    for pair in zip(args, shapes):
        if pair not in seen:
            seen.add(pair)
            result.append(pair)
    args   = [a for a, _ in result]
    shapes = [s for _, s in result]
    return args, shapes


def _read_file_stream(filename: str):
    """Yield lines from a (potentially large) file one by one."""
    with open(filename, 'r') as fh:
        for line in fh:
            yield line


def _extract_element_shape(shape: str) -> str:
    matches = re.findall(r'tensor<(?:\d+x)*([fi]\d+)>', shape)
    if not matches:
        raise ValueError(f"Cannot extract element type from shape: {shape!r}")
    return matches[0]


# ---------------------------------------------------------------------------
# Wrapper generators
# ---------------------------------------------------------------------------

def nn_transform_wrapper(operation: str) -> str:
    """Generate a @main that allocates tensors and calls the model function,
    printing the elapsed time via @printI64 (suitable for the CMD backend).

    Args:
        operation: The function signature line of the model's forward function.

    Returns:
        MLIR source string for the @main function.
    """
    fields = re.findall(r"\s*\(([^()]+)\)\s*->\s*([^(]+)", operation)[0]

    args, shapes = [], []
    for f in fields[0].split(', '):
        arg, shape = f.split(':')
        shapes.append(shape.strip())
        args.append(arg.strip())

    args, shapes = _remove_duplicate_args(args, shapes)
    shapes.append(fields[1])

    dims = []
    for shape in shapes:
        if shape.startswith("tensor"):
            arg_dims = list(map(int, re.findall(r'\d+', shape[7:-5])))
            dims.append(arg_dims)
        else:
            dims.append(-1)

    func_name = re.search(r"@(\w+)", operation).group(1)
    func_call = (
        f"func.call @{func_name}({', '.join(args)}) : "
        f"({', '.join(shapes[:-1])}) -> {shapes[-1]}"
    )

    code  = "func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }\n"
    code += "func.func private @printI64(i64)\n"
    code += "func.func private @printNewline()\n\n\n"
    code += "func.func @main(){\n"
    code += "    %c1 = arith.constant 1: index\n"
    code += "    %c0 = arith.constant 0 : index\n"
    code += "    %n = arith.constant 2: index\n\n"
    code += "    %val_f32 = arith.constant 2.00000e+00 : f32\n"
    code += "    %val_i64 = arith.constant 2 : i64\n"
    code += "    %zero = arith.constant 0.00000e+00 : f32\n\n"

    for arg, shape, arg_dims in zip(args, shapes, dims):
        if arg_dims != -1:
            tmp_arg = f'%tmp_{arg[1:]}'
            etype = _extract_element_shape(shape)
            code += f"    {tmp_arg} = bufferization.alloc_tensor() : {shape}\n"
            code += (
                f"    {arg} = linalg.fill ins(%val_{etype} : {etype}) "
                f"outs({tmp_arg} : {shape}) -> {shape}\n"
            )
        else:
            code += f"    {arg} = arith.constant 2.00000e+00 : f32\n"

    code += "        \n"
    code += "    scf.for %i = %c0 to %n step %c1 {\n"
    code += "        %t0 = func.call @nanoTime() : () -> (i64)\n"
    code += "        \n"
    code += f"         %outputmain = {func_call}\n"
    code += "        \n"
    code += "        %t = func.call @nanoTime() : () -> (i64)\n"
    code += "        %delta = arith.subi %t, %t0 : i64\n"
    code += "        func.call @printI64(%delta) : (i64) -> ()\n"
    code += "        func.call @printNewline() : () -> ()\n"
    code += "        \n"
    code += "    }\n"
    code += "    return\n"
    code += "}\n"
    return code


def nn_transform_wrapper_binding(operation: str) -> str:
    """Generate a @main that receives tensors as arguments and *returns* the
    elapsed time as i64 (suitable for the Python bindings backend).

    Args:
        operation: The function signature line of the model's forward function.

    Returns:
        MLIR source string for the @main function.
    """
    # Handles multiple return values: -> (type1, type2, ...)
    fields = re.findall(r"\s*\(([^()]+)\)\s*->\s*\(([^(]+)\)", operation)[0]

    args, shapes = [], []
    for f in fields[0].split(', '):
        arg, shape = f.split(':')
        shapes.append(shape.strip())
        args.append(arg.strip())

    args, shapes = _remove_duplicate_args(args, shapes)
    shapes.append(fields[1])

    num_return = len(fields[1].split(", ")) if "," in fields[1] else 0

    func_name = re.search(r"@(\w+)", operation).group(1)
    func_call = (
        f"func.call @{func_name}({', '.join(args)}) : "
        f"({', '.join(shapes[:-1])}) -> ({shapes[-1]})"
    )

    arg_decls = ', '.join(f'{a}: {s}' for a, s in zip(args, shapes))

    code  = "func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }\n"
    code += "func.func private @printI64(i64)\n"
    code += "func.func private @printF32(f32)\n"
    code += "func.func private @printNewline()\n\n"
    code += (
        f"func.func @main({arg_decls}) -> i64 "
        f"attributes {{ llvm.emit_c_interface }} {{\n"
    )
    code += "    %c1 = arith.constant 1: index\n"
    code += "    %c0 = arith.constant 0 : index\n"
    code += "    %n = arith.constant 2: index\n"
    code += "    %init_delta = arith.constant 0 : i64\n \n"
    code += "    %final_delta = scf.for %i = %c0 to %n step %c1 iter_args(%d = %init_delta) -> (i64) {\n"
    code += "    %t0 = func.call @nanoTime() : () -> (i64)\n"

    if not num_return:
        code += f"    %outputmain = {func_call}\n"
    else:
        code += f"    %outputmain:{num_return} = {func_call}\n"

    code += "    %t = func.call @nanoTime() : () -> (i64)\n"
    code += "    %delta = arith.subi %t, %t0 : i64\n"
    code += "    scf.yield %delta : i64\n"
    code += "}\n"
    code += "    return %final_delta : i64\n"
    code += "}\n"
    return code


# ---------------------------------------------------------------------------
# File-level wrapper
# ---------------------------------------------------------------------------

def main_wrapper(filename: str, model_name: str, out: str) -> None:
    """Read a model's MLIR file, find the forward function, and append a
    timed @main using the bindings-compatible wrapper.

    Args:
        filename:   Path to the input .mlir file.
        model_name: Name (substring) of the model's forward function.
        out:        Path to write the wrapped output .mlir file.
    """
    wrapped_code = ''
    return_found = False
    wrapped_code_added = False

    with open(out, "w") as o:
        for line in _read_file_stream(filename):
            o.write(line)

            if not wrapped_code and model_name in line:
                signature = line[:-2].strip()
                wrapped_code = nn_transform_wrapper_binding(signature)

            if not return_found and "return" in line:
                return_found = True

            if (not wrapped_code_added and wrapped_code
                    and return_found and "}" in line):
                o.write(wrapped_code)
                wrapped_code_added = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Wrap an MLIR model file with a timed @main function."
    )
    parser.add_argument("--input",      required=True, help="Input .mlir file.")
    parser.add_argument("--model-name", required=True,
                        help="Name of the forward function to wrap (e.g. 'forward', 'main_graph').")
    parser.add_argument("--output",     required=True, help="Output .mlir file.")
    args = parser.parse_args()

    main_wrapper(args.input, args.model_name, args.output)
    print(f"Wrapped '{args.model_name}' from {args.input} → {args.output}")
