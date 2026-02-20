"""Inject noalias scope metadata into lowered MLIR LLVM dialect IR.

Identifies alloca-derived memory accesses (promoted A/B tiles) vs
argument-derived accesses (C matrix) and adds alias_scopes/noalias_scopes
attributes to tell LLVM they don't alias.

Usage: Called from run_noalias.py after lower() and before ExecutionEngine.
"""

import re

# Alias scope attribute strings
DOMAIN = '#llvm.alias_scope_domain<id = distinct[0]<>, description = "matmul_noalias">'
ALLOCA_SCOPE = '#llvm.alias_scope<id = distinct[1]<>, domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "matmul_noalias">, description = "alloca_buffers">'
ARG_SCOPE = '#llvm.alias_scope<id = distinct[2]<>, domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "matmul_noalias">, description = "arg_ptrs">'

ALLOCA_ALIAS = f'alias_scopes = [{ALLOCA_SCOPE}], noalias_scopes = [{ARG_SCOPE}]'
ARG_ALIAS = f'alias_scopes = [{ARG_SCOPE}], noalias_scopes = [{ALLOCA_SCOPE}]'


def inject_noalias(mlir_asm: str) -> str:
    """Inject noalias scope metadata into MLIR LLVM dialect IR text.

    Strategy:
    1. Find alloca SSA values (e.g., %41, %58)
    2. Trace GEP chains: GEP from alloca → alloca-derived ptr
    3. Trace GEP chains: GEP from %argN → arg-derived ptr
    4. Annotate loads/stores based on their pointer operand's derivation

    Returns modified MLIR ASM string.
    """
    lines = mlir_asm.split('\n')

    # Build SSA def-use map for pointer derivation
    # Track which SSA values are alloca-derived vs arg-derived
    alloca_derived = set()  # SSA names derived from allocas
    arg_derived = set()     # SSA names derived from function args

    # First pass: find allocas and args
    for line in lines:
        stripped = line.strip()

        # Find alloca results: %N = llvm.alloca ...
        m = re.match(r'(%\d+)\s*=\s*llvm\.alloca\b', stripped)
        if m:
            alloca_derived.add(m.group(1))
            continue

        # Function args are arg-derived: %argN
        # We mark them as we see GEPs from them

    # Mark all %argN as arg-derived
    for m in re.finditer(r'%arg\d+', mlir_asm):
        arg_derived.add(m.group(0))

    # Second pass: propagate through GEPs
    # GEP results inherit the derivation of their base pointer
    # Need multiple iterations since GEPs can chain
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
                elif result not in arg_derived and base in arg_derived:
                    arg_derived.add(result)
                    changed = True

    # Third pass: annotate loads and stores
    new_lines = []
    annotated_loads = 0
    annotated_stores = 0

    for line in lines:
        stripped = line.strip()

        # Match vector loads: %N = llvm.load %ptr {alignment = ...} : !llvm.ptr -> vector<...>
        # or scalar loads: %N = llvm.load %ptr : !llvm.ptr -> f64
        load_match = re.match(r'(%\d+)\s*=\s*llvm\.load\s+(%\w+)', stripped)
        if load_match:
            ptr_operand = load_match.group(2)
            alias_attr = None
            if ptr_operand in alloca_derived:
                alias_attr = ALLOCA_ALIAS
            elif ptr_operand in arg_derived:
                alias_attr = ARG_ALIAS

            if alias_attr:
                # Insert alias attrs into the load's attribute dict
                if '{' in line and '}' in line:
                    # Has existing attributes - add before closing brace
                    line = line.replace('}', f', {alias_attr}}}', 1)
                else:
                    # No existing attributes - add before the colon type annotation
                    # Pattern: llvm.load %ptr : !llvm.ptr -> type
                    line = re.sub(
                        r'(llvm\.load\s+%\w+)\s*:',
                        rf'\1 {{{alias_attr}}} :',
                        line
                    )
                annotated_loads += 1

        # Match stores: llvm.store %val, %ptr {alignment = ...} : type, !llvm.ptr
        store_match = re.match(r'llvm\.store\s+%\w+,\s*(%\w+)', stripped)
        if store_match:
            ptr_operand = store_match.group(1)
            alias_attr = None
            if ptr_operand in alloca_derived:
                alias_attr = ALLOCA_ALIAS
            elif ptr_operand in arg_derived:
                alias_attr = ARG_ALIAS

            if alias_attr:
                if '{' in line and '}' in line:
                    line = line.replace('}', f', {alias_attr}}}', 1)
                else:
                    line = re.sub(
                        r'(llvm\.store\s+%\w+,\s*%\w+)\s*:',
                        rf'\1 {{{alias_attr}}} :',
                        line
                    )
                annotated_stores += 1

        new_lines.append(line)

    import sys
    print(f'noalias injection: annotated {annotated_loads} loads, {annotated_stores} stores', file=sys.stderr)
    print(f'  alloca-derived ptrs: {len(alloca_derived)}', file=sys.stderr)
    print(f'  arg-derived ptrs: {len(arg_derived)}', file=sys.stderr)

    return '\n'.join(new_lines)
