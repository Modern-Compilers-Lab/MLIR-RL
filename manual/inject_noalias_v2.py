"""Enhanced noalias injection with per-argument scopes.

Creates separate alias scopes for:
1. alloca-derived (promoted A/B tiles)
2. arg-A (%arg1 derived)
3. arg-B (%arg8 derived)
4. arg-C (%arg15 derived)

Each scope says it doesn't alias with any other scope.
This is stronger than the v1 injection which only has alloca vs arg.
"""

import re
import sys

# Alias scope attribute strings — 4 separate scopes
DOMAIN = '#llvm.alias_scope_domain<id = distinct[0]<>, description = "matmul_noalias_v2">'
ALLOCA_SCOPE = '#llvm.alias_scope<id = distinct[1]<>, domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "matmul_noalias_v2">, description = "alloca_buffers">'
ARG_A_SCOPE = '#llvm.alias_scope<id = distinct[2]<>, domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "matmul_noalias_v2">, description = "arg_A">'
ARG_B_SCOPE = '#llvm.alias_scope<id = distinct[3]<>, domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "matmul_noalias_v2">, description = "arg_B">'
ARG_C_SCOPE = '#llvm.alias_scope<id = distinct[4]<>, domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "matmul_noalias_v2">, description = "arg_C">'

# For alloca: in alloca scope, doesn't alias with any arg scope
ALLOCA_ALIAS = f'alias_scopes = [{ALLOCA_SCOPE}], noalias_scopes = [{ARG_A_SCOPE}, {ARG_B_SCOPE}, {ARG_C_SCOPE}]'
# For A: in A scope, doesn't alias with alloca, B, C
ARG_A_ALIAS = f'alias_scopes = [{ARG_A_SCOPE}], noalias_scopes = [{ALLOCA_SCOPE}, {ARG_B_SCOPE}, {ARG_C_SCOPE}]'
# For B: in B scope, doesn't alias with alloca, A, C
ARG_B_ALIAS = f'alias_scopes = [{ARG_B_SCOPE}], noalias_scopes = [{ALLOCA_SCOPE}, {ARG_A_SCOPE}, {ARG_C_SCOPE}]'
# For C: in C scope, doesn't alias with alloca, A, B
ARG_C_ALIAS = f'alias_scopes = [{ARG_C_SCOPE}], noalias_scopes = [{ALLOCA_SCOPE}, {ARG_A_SCOPE}, {ARG_B_SCOPE}]'


def inject_noalias_v2(mlir_asm: str) -> str:
    """Inject per-argument noalias scope metadata into MLIR LLVM dialect IR.

    The memref struct layout for 2D memrefs is:
    - arg0: alloc ptr, arg1: aligned data ptr, arg2: offset, arg3-4: sizes, arg5-6: strides
    - arg7: alloc ptr, arg8: aligned data ptr, ...
    - arg14: alloc ptr, arg15: aligned data ptr, ...

    So the data pointers are: %arg1 (A), %arg8 (B), %arg15 (C)
    """
    lines = mlir_asm.split('\n')

    # Build SSA def-use map for pointer derivation
    alloca_derived = set()
    arg_a_derived = set()
    arg_b_derived = set()
    arg_c_derived = set()

    # First pass: find allocas and identify data pointer args
    for line in lines:
        stripped = line.strip()
        m = re.match(r'(%\d+)\s*=\s*llvm\.alloca\b', stripped)
        if m:
            alloca_derived.add(m.group(1))

    # Mark specific function args
    # A data ptr: %arg1 (and %arg0 for alloc ptr)
    arg_a_derived.add('%arg0')
    arg_a_derived.add('%arg1')
    # B data ptr: %arg8 (and %arg7 for alloc ptr)
    arg_b_derived.add('%arg7')
    arg_b_derived.add('%arg8')
    # C data ptr: %arg15 (and %arg14 for alloc ptr)
    arg_c_derived.add('%arg14')
    arg_c_derived.add('%arg15')

    # Second pass: propagate through GEPs
    changed = True
    while changed:
        changed = False
        for line in lines:
            stripped = line.strip()
            m = re.match(r'(%\d+)\s*=\s*llvm\.getelementptr\b.*?(%\w+)', stripped)
            if m:
                result = m.group(1)
                base = m.group(2)
                if result not in alloca_derived and base in alloca_derived:
                    alloca_derived.add(result)
                    changed = True
                elif result not in arg_a_derived and base in arg_a_derived:
                    arg_a_derived.add(result)
                    changed = True
                elif result not in arg_b_derived and base in arg_b_derived:
                    arg_b_derived.add(result)
                    changed = True
                elif result not in arg_c_derived and base in arg_c_derived:
                    arg_c_derived.add(result)
                    changed = True

    # Third pass: annotate loads and stores
    new_lines = []
    stats = {'alloca_loads': 0, 'a_loads': 0, 'b_loads': 0, 'c_loads': 0,
             'alloca_stores': 0, 'a_stores': 0, 'b_stores': 0, 'c_stores': 0}

    for line in lines:
        stripped = line.strip()

        # Match loads
        load_match = re.match(r'(%\d+)\s*=\s*llvm\.load\s+(%\w+)', stripped)
        if load_match:
            ptr_operand = load_match.group(2)
            alias_attr = None
            if ptr_operand in alloca_derived:
                alias_attr = ALLOCA_ALIAS
                stats['alloca_loads'] += 1
            elif ptr_operand in arg_a_derived:
                alias_attr = ARG_A_ALIAS
                stats['a_loads'] += 1
            elif ptr_operand in arg_b_derived:
                alias_attr = ARG_B_ALIAS
                stats['b_loads'] += 1
            elif ptr_operand in arg_c_derived:
                alias_attr = ARG_C_ALIAS
                stats['c_loads'] += 1

            if alias_attr:
                if '{' in line and '}' in line:
                    line = line.replace('}', f', {alias_attr}}}', 1)
                else:
                    line = re.sub(
                        r'(llvm\.load\s+%\w+)\s*:',
                        rf'\1 {{{alias_attr}}} :',
                        line
                    )

        # Match stores
        store_match = re.match(r'llvm\.store\s+%\w+,\s*(%\w+)', stripped)
        if store_match:
            ptr_operand = store_match.group(1)
            alias_attr = None
            if ptr_operand in alloca_derived:
                alias_attr = ALLOCA_ALIAS
                stats['alloca_stores'] += 1
            elif ptr_operand in arg_a_derived:
                alias_attr = ARG_A_ALIAS
                stats['a_stores'] += 1
            elif ptr_operand in arg_b_derived:
                alias_attr = ARG_B_ALIAS
                stats['b_stores'] += 1
            elif ptr_operand in arg_c_derived:
                alias_attr = ARG_C_ALIAS
                stats['c_stores'] += 1

            if alias_attr:
                if '{' in line and '}' in line:
                    line = line.replace('}', f', {alias_attr}}}', 1)
                else:
                    line = re.sub(
                        r'(llvm\.store\s+%\w+,\s*%\w+)\s*:',
                        rf'\1 {{{alias_attr}}} :',
                        line
                    )

        new_lines.append(line)

    total_loads = sum(v for k, v in stats.items() if 'loads' in k)
    total_stores = sum(v for k, v in stats.items() if 'stores' in k)
    print(f'noalias_v2 injection: annotated {total_loads} loads, {total_stores} stores', file=sys.stderr)
    print(f'  alloca: {stats["alloca_loads"]}L/{stats["alloca_stores"]}S', file=sys.stderr)
    print(f'  arg_A:  {stats["a_loads"]}L/{stats["a_stores"]}S', file=sys.stderr)
    print(f'  arg_B:  {stats["b_loads"]}L/{stats["b_stores"]}S', file=sys.stderr)
    print(f'  arg_C:  {stats["c_loads"]}L/{stats["c_stores"]}S', file=sys.stderr)

    return '\n'.join(new_lines)
