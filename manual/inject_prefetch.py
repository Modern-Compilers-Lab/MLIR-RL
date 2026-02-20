"""Inject software prefetch instructions into the K-loop of the matmul kernel.

Adds llvm.intr.prefetch for the next K iteration's A tile data.
The A tile (128KB) doesn't fit in L1 cache, so prefetching helps.
"""

import re
import sys


def inject_prefetch(mlir_asm: str, target: str = 'A', n_cachelines: int = 4) -> str:
    """Inject prefetch instructions into the K-loop body.

    Args:
        mlir_asm: MLIR LLVM dialect assembly string
        target: 'A' to prefetch A tile, 'B' to prefetch B tile
        n_cachelines: Number of cache lines to prefetch (1-8)

    Returns:
        Modified MLIR ASM with prefetch instructions.
    """
    lines = mlir_asm.split('\n')

    # Find allocas
    allocas = []
    for line in lines:
        m = re.match(r'\s+(%\d+)\s*=\s*llvm\.alloca\b', line)
        if m:
            allocas.append(m.group(1))

    if len(allocas) < 2:
        print('Warning: Not enough allocas found', file=sys.stderr)
        return mlir_asm

    a_alloca = allocas[0]
    b_alloca = allocas[1]
    target_alloca = a_alloca if target == 'A' else b_alloca
    print(f'prefetch: target={target} alloca={target_alloca}', file=sys.stderr)

    # Find step constant
    step_var = None
    for line in lines:
        m = re.match(r'\s+(%\d+)\s*=\s*llvm\.mlir\.constant\(8\s*:\s*index\)', line)
        if m:
            step_var = m.group(1)
            break

    # Find the target alloca's GEP in the K-loop and its K offset
    target_gep = None
    k_offset_var = None
    for line in lines:
        if f'llvm.getelementptr {target_alloca}[' in line:
            m = re.match(r'\s+(%\d+)\s*=\s*llvm\.getelementptr\s+\S+\[(%\d+)\]', line)
            if m:
                target_gep = m.group(1)
                k_offset_var = m.group(2)
                break

    if not target_gep or not k_offset_var:
        print(f'Warning: Could not find {target} GEP in K-loop', file=sys.stderr)
        return mlir_asm

    # Count vector loads from target GEP
    target_loads = 0
    for line in lines:
        if 'llvm.load' in line and 'vector<8xf64>' in line:
            m = re.match(r'\s+%\d+\s*=\s*llvm\.load\s+(%\d+)', line)
            if m:
                ptr = m.group(1)
                for line2 in lines:
                    if f'{ptr} = llvm.getelementptr {target_gep}[' in line2:
                        target_loads += 1
                        break

    print(f'prefetch: found {target_loads} {target} loads', file=sys.stderr)

    # Inject prefetches after last target load
    new_lines = []
    in_k_loop = False
    loads_seen = 0
    ssa_counter = 500

    for i, line in enumerate(lines):
        new_lines.append(line)

        if f'llvm.getelementptr {target_alloca}[' in line:
            in_k_loop = True
            loads_seen = 0

        if in_k_loop and 'llvm.load' in line and 'vector<8xf64>' in line:
            m = re.match(r'\s+%\d+\s*=\s*llvm\.load\s+(%\d+)', line)
            if m:
                ptr = m.group(1)
                for line2 in lines:
                    if f'{ptr} = llvm.getelementptr {target_gep}[' in line2:
                        loads_seen += 1
                        break

        if in_k_loop and loads_seen == target_loads and target_loads > 0:
            indent = '           '

            # Compute next K offset
            # For A: k_offset_var = row_base + K*stride (complex), simpler to just
            # prefetch relative to current position
            # For B: k_offset_var = K*8

            # Approach: compute target_alloca + (k_offset_var + step*step)
            # which is the next K iteration's starting position
            pf_step_sq = f'%pf_{ssa_counter}'
            ssa_counter += 1
            new_lines.append(f'{indent}{pf_step_sq} = llvm.mul {step_var}, {step_var} overflow<nsw> : i64')

            pf_next = f'%pf_{ssa_counter}'
            ssa_counter += 1
            new_lines.append(f'{indent}{pf_next} = llvm.add {k_offset_var}, {pf_step_sq} : i64')

            pf_base = f'%pf_{ssa_counter}'
            ssa_counter += 1
            new_lines.append(f'{indent}{pf_base} = llvm.getelementptr {target_alloca}[{pf_next}] : (!llvm.ptr, i64) -> !llvm.ptr, f64')

            cache_line_doubles = 8  # 64 bytes / 8 bytes per f64
            for cl in range(n_cachelines):
                if cl == 0:
                    pf_ptr = pf_base
                else:
                    offset_val = cl * cache_line_doubles
                    c1 = f'%pf_{ssa_counter}'
                    ssa_counter += 1
                    new_lines.append(f'{indent}{c1} = llvm.mlir.constant({offset_val} : i64) : i64')
                    c2 = f'%pf_{ssa_counter}'
                    ssa_counter += 1
                    new_lines.append(f'{indent}{c2} = llvm.getelementptr {pf_base}[{c1}] : (!llvm.ptr, i64) -> !llvm.ptr, f64')
                    pf_ptr = c2

                # hint=3 → T0 (L1), hint=2 → T1 (L2), hint=1 → T2 (L3)
                hint = 3 if cl < 2 else 2  # First 2 lines to L1, rest to L2
                new_lines.append(f'{indent}"llvm.intr.prefetch"({pf_ptr}) <{{cache = 1 : i32, hint = {hint} : i32, rw = 0 : i32}}> : (!llvm.ptr) -> ()')

            in_k_loop = False
            loads_seen = 0
            print(f'prefetch: inserted {n_cachelines} prefetch ops', file=sys.stderr)

    return '\n'.join(new_lines)
