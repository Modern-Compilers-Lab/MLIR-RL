"""
add_timing_wrapper.py
---------------------
Modifies a full-model .mlir file so that `@main` returns (original_tensor, i64)
with the delta as the execution time measured via @nanoTime().

The resulting file is compatible with the Execution.execute_code() engine
which expects `@main -> (..., i64)` format.

Input:  func.func @main(%arg0: tensor<...>, ...) -> tensor<...>
Output: func.func @main(%arg0: tensor<...>, ...) -> (tensor<...>, i64)

Usage:
  python scripts/full_model/add_timing_wrapper.py --input data/nn/tagged/gcn_tagged.mlir --output data/nn/wrapped/gcn_wrapped.mlir
"""

import re
import argparse
import os


def extract_main_signature(code: str) -> tuple[str, str, str, str]:
    """Extract the @main function's arg string and return type string.

    Returns:
        (args_str, return_type_str, pre_body, post_body) where:
        - args_str: everything between @main(...) minus the leading 'func.func @main'
        - return_type_str: the return type (e.g. 'tensor<1x1000xf32>')
        - pre_body: everything before @main's opening brace
        - post_body: everything after @main's closing brace + module wrapper
    """
    # Find @main function declaration
    main_match = re.search(
        r'(func\.func\s+@main\s*\(([^)]*)\)\s*->\s*(\S+)\s*\{)',
        code
    )
    if not main_match:
        raise ValueError("Could not find func.func @main(…) -> … { in code")

    args_str = main_match.group(2).strip()
    return_type_str = main_match.group(3).strip()

    # Find the body: from the opening brace to the matching closing brace
    body_start = main_match.end()
    # Count braces to find matching close
    depth = 1
    pos = body_start
    while depth > 0 and pos < len(code):
        if code[pos] == '{':
            depth += 1
        elif code[pos] == '}':
            depth -= 1
        pos += 1
    body_end = pos  # position after matching '}'

    body_content = code[body_start:body_end - 1]  # strip the final '}'
    pre_body = code[:main_match.start()]
    post_body = code[body_end:]

    return args_str, return_type_str, pre_body, body_content, post_body


def find_return_line(body: str) -> tuple[str, str, str]:
    """Find the return statement in the function body and split around it."""
    # Match return with optional tensor and type
    return_match = re.search(r'(\s*return\s+)(%\S+)\s*:\s*(\S+)', body)
    if not return_match:
        raise ValueError("Could not find 'return %val : type' in @main body")

    pre_return = body[:return_match.start()]
    return_val = return_match.group(2)
    return_type = return_match.group(3).rstrip(')')
    post_return = body[return_match.end():]

    return pre_return, return_val, return_type, post_return


def wrap_model(code: str) -> str:
    """Add @nanoTime() timing wrapper to the @main function.

    Transforms:
        func.func @main(%args) -> tensor_ret {
            ...body...
            return %ret : tensor_ret
        }

    Into:
        func.func @main(%args) -> (tensor_ret, i64) {
            %t0 = func.call @nanoTime() : () -> i64
            ...body...
            %t1 = func.call @nanoTime() : () -> i64
            %delta = arith.subi %t1, %t0 : i64
            return %ret, %delta : tensor_ret, i64
        }
    """
    args_str, return_type_str, pre_body, body_content, post_body = extract_main_signature(code)

    try:
        pre_ret, ret_val, ret_type, post_ret = find_return_line(body_content)
    except ValueError:
        # If there's no return with explicit type, try simpler match
        # This handles cases where the body is complex
        pre_ret = body_content
        ret_val = ""
        ret_type = ""
        post_ret = ""

    # Build the timed body
    timed_body = (
        f'{pre_ret}\n'
        f'    %t0 = func.call @nanoTime() : () -> i64\n'
        f'    %t1 = func.call @nanoTime() : () -> i64\n'
        f'    %delta = arith.subi %t1, %t0 : i64\n'
        f'    return {ret_val}, %delta : {ret_type}, i64\n'
        f'{post_ret}'
    )

    # Build the new @main function signature
    new_main = (
        f'func.func @main({args_str}) -> ({return_type_str}, i64) {{\n'
        f'{timed_body}'
        f'}}'
    )

    # Reassemble the full file
    # Add @nanoTime declaration if not already present
    if '@nanoTime' not in pre_body:
        nano_decl = '  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}\n'
        # Insert after the module opening or first line
        if 'module' in pre_body:
            # Find module opening and insert after it
            pre_body = re.sub(
                r'(module\s*(?:attributes\s*\{[^}]*\})?\s*\{)',
                r'\1\n' + nano_decl,
                pre_body,
                count=1
            )
        else:
            pre_body = nano_decl + pre_body

    result = pre_body + new_main + post_body
    return result


def add_timing_wrapper_whole_body(code: str) -> str:
    """Alternative approach: wrap the entire main body with timing instead of
    trying to find the return statement. This is more robust for complex models.

    Wraps the entire body of @main between nanoTime calls and changes
    the return to include the delta.
    """
    # Find @main function
    main_match = re.search(
        r'(func\.func\s+@main\s*\(([^)]*)\)\s*->\s*(\S+)\s*\{)\n?',
        code
    )
    if not main_match:
        raise ValueError("Could not find func.func @main(…) -> … { in code")

    args_str = main_match.group(2).strip()
    return_type_str = main_match.group(3).strip()

    # Find matching closing brace
    body_start = main_match.end()
    depth = 1
    pos = body_start
    while depth > 0 and pos < len(code):
        if code[pos] == '{':
            depth += 1
        elif code[pos] == '}':
            depth -= 1
        pos += 1
    body_end = pos - 1  # position of matching '}'

    body = code[body_start:body_end]

    # Find the final return in the body and modify it
    # We look for the last 'return' that's not inside a nested block
    lines = body.split('\n')
    return_idx = None
    return_line = None
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith('return '):
            return_idx = i
            return_line = stripped
            break

    if return_idx is None:
        raise ValueError("Could not find 'return' in @main body")

    # Parse the return line: "return %val : type" or "return %val, %val2 : type1, type2"
    # Extract the value(s) and type(s)
    ret_content = return_line[len('return '):].strip()
    if ':' in ret_content:
        ret_vals, ret_types = ret_content.rsplit(':', 1)
        ret_vals = ret_vals.strip()
        ret_types = ret_types.strip()
    else:
        ret_vals = ret_content
        ret_types = return_type_str

    # Build new body
    new_lines = lines[:return_idx]
    new_body = '\n'.join(new_lines)

    timed_body = (
        f'{new_body}\n'
        f'    %t0 = func.call @nanoTime() : () -> i64\n'
        f'    %t1 = func.call @nanoTime() : () -> i64\n'
        f'    %delta = arith.subi %t1, %t0 : i64\n'
        f'    return {ret_vals}, %delta : {ret_types}, i64'
    )

    # Add @nanoTime declaration
    pre_body = code[:main_match.start()]
    post_body = code[pos:]

    if '@nanoTime' not in pre_body:
        nano_decl = '  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}\n'
        pre_body = re.sub(
            r'(module\s*(?:attributes\s*\{[^}]*\})?\s*\{)',
            r'\1\n' + nano_decl,
            pre_body,
            count=1
        )

    new_main = (
        f'func.func @main({args_str}) -> ({return_type_str}, i64) {{\n'
        f'{timed_body}\n'
        f'}}'
    )

    return pre_body + new_main + post_body


def main():
    parser = argparse.ArgumentParser(
        description="Add @nanoTime() timing wrapper to a full-model .mlir file."
    )
    parser.add_argument("--input", required=True, help="Path to input .mlir file.")
    parser.add_argument("--output", required=True, help="Path to write wrapped .mlir file.")
    parser.add_argument(
        "--whole-body",
        action="store_true",
        default=True,
        help="Wrap entire body with timing (default). More robust for complex models."
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        code = f.read()

    wrapped = add_timing_wrapper_whole_body(code)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(wrapped)
    print(f"Timing wrapper added → {args.output}")


if __name__ == "__main__":
    main()
