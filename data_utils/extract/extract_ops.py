"""
extract_ops.py
--------------
Extract individual linalg operations from a full-model MLIR file, producing
one benchmark file per operation ready for RL training.

Each extracted file wraps a single linalg op in a timed @main function:
  - all tensor inputs/outputs become function arguments
  - dynamic batch dimensions (?) are replaced with a concrete value
  - scalar constants referenced in linalg.generic bodies are hoisted into @main
  - linalg.generic ops whose bodies reference elided tensor weights are skipped

Usage:
    python data_utils/extract_ops.py \\
        --input  data/nn/raw/resnet18_linalg.mlir \
        --output-dir data/nn/code_files/resnet18/
        --batch-size 1 \\
        --model-name resnet18

    # Batch-process all generated models:
    for f in data/nn/raw/*_linalg.mlir; do
        model=$(basename $f _linalg.mlir)
        python data_utils/extract_ops.py \\
            --input $f --output-dir data/nn/code_files/$model \\
            --batch-size 1 --model-name $model
    done
"""

import re
import argparse
import os
import textwrap
import json
import hashlib
import random


# Ops to extract — must be single-op linalg structured ops.
# linalg.broadcast, linalg.fill, linalg.transpose are intentionally excluded
# (too trivial for the scheduler).
TARGET_OPS = [
    "linalg.conv_2d_nchw_fchw",
    "linalg.conv_2d_nhwc_hwcf",
    "linalg.matmul",
    "linalg.batch_matmul",
    "linalg.pooling_nchw_max",
    "linalg.pooling_nhwc_max",
    "linalg.pooling_nchw_sum",
    "linalg.pooling_nhwc_sum",
    "linalg.add",
    "linalg.generic",
]

# Single-word ops: if the op keyword appears as a word inside a longer name
# (e.g. "linalg.generics"), skip that longer name.
_OP_KEYWORDS = {op.split("linalg.")[1] for op in TARGET_OPS}


# ---------------------------------------------------------------------------
# Text-parsing helpers
# ---------------------------------------------------------------------------

def _find_closing_paren(text: str, start: int) -> int:
    """Return the index of ')' that matches '(' at *start*. Returns -1 if not found."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_comma_angles(text: str) -> list[str]:
    """Split *text* by ',' while respecting nested '<>' angle brackets."""
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in text:
        if ch == "<":
            depth += 1
            buf.append(ch)
        elif ch == ">":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def _parse_section(op_text: str, keyword: str) -> tuple[list[str], list[str]]:
    """Extract (ssa_names, type_strings) from the *keyword*( ... ) block.

    The MLIR form is:  keyword(%a, %b : type_a, type_b)
    Returns empty lists when the section is not found.
    """
    # We prepend a space so that "ins" doesn't match inside "iterator_types"
    search = " " + keyword + "("
    idx = op_text.find(search)
    if idx == -1:
        return [], []
    paren_start = idx + len(search) - 1  # position of '('
    paren_end = _find_closing_paren(op_text, paren_start)
    if paren_end == -1:
        return [], []
    section = op_text[paren_start + 1 : paren_end]

    # The last occurrence of ' : ' separates ssa_names from types.
    # Tensor types do not contain ' : ', so rfind is safe.
    colon_idx = section.rfind(" : ")
    if colon_idx == -1:
        return [], []

    args_str = section[:colon_idx]
    types_str = section[colon_idx + 3 :]
    args = [a.strip() for a in args_str.split(",")]
    types = _split_comma_angles(types_str)
    return args, types


def _extract_result_type(op_text: str) -> str:
    """Extract the tensor result type from the trailing '-> tensor<...>'."""
    # For multi-line ops the '->' is on the last line; for single-line it's at the end.
    last_line = op_text.split("\n")[-1]
    m = re.search(r"->\s*(tensor<[^>]+(?:>[^>]*)*>)", last_line)
    if m:
        return m.group(1)
    return ""


def _specialize(t: str, batch_size: int) -> str:
    """Replace '?' with *batch_size* in a type string."""
    return t.replace("?", str(batch_size))


def _has_dynamic(types: list[str]) -> bool:
    return any("?" in t for t in types)


def _normalize_batch_candidates(candidates: list[int] | None, default_batch: int) -> list[int]:
    """Return sorted, unique, positive candidate batch sizes."""
    if not candidates:
        return [default_batch]
    clean = sorted({int(v) for v in candidates if int(v) > 0})
    return clean if clean else [default_batch]


def _tensor_numel(tensor_type: str) -> int:
    """Estimate element count for tensor<...> using 1 for dynamic dims."""
    m = re.search(r"tensor<([^>]+)>", tensor_type)
    if not m:
        return 1
    body = m.group(1)
    parts = body.split("x")
    if len(parts) < 2:
        return 1
    dims = parts[:-1]  # last token is element type (f32, i64, ...)
    numel = 1
    for d in dims:
        d = d.strip()
        if d == "?":
            numel *= 1
        elif d.isdigit():
            numel *= int(d)
        else:
            # Unknown dimension token; keep estimate conservative.
            numel *= 1
    return max(numel, 1)


def _estimate_op_footprint(ins_types: list[str], outs_types: list[str], result_type: str) -> int:
    """Estimate tensor element footprint (inputs + outputs + result)."""
    total = 0
    for t in ins_types + outs_types:
        total += _tensor_numel(t)
    if result_type:
        total += _tensor_numel(result_type)
    return total


def _choose_dynamic_batches(op_type: str,
                            ins_types: list[str], outs_types: list[str], result_type: str,
                            policy: str,
                            candidates: list[int],
                            default_batch: int,
                            max_variants: int) -> list[int]:
    """Choose one or more replacement batch sizes for dynamic dims.

    Policy:
      - static: always use default_batch
      - smart: one heuristic choice
      - hybrid: one heuristic + one contrastive size (bounded by max_variants)
    """
    if policy == "static":
        return [default_batch]

    pool = _normalize_batch_candidates(candidates, default_batch)
    if len(pool) == 1:
        return pool

    footprint = _estimate_op_footprint(ins_types, outs_types, result_type)
    heavy_ops = {"conv_2d_nchw_fchw", "conv_2d_nhwc_hwcf", "matmul", "batch_matmul"}
    light_ops = {
        "add",
        "pooling_nchw_max", "pooling_nhwc_max", "pooling_nchw_sum", "pooling_nhwc_sum",
        "generic",
    }

    lo = pool[0]
    hi = pool[-1]
    mid = pool[len(pool) // 2]

    # Heuristic 1: memory-heavy structured ops prefer smaller batches.
    # Heuristic 2: light elementwise/pooling ops can use larger batches.
    # Heuristic 3: medium footprint defaults to middle batch.
    if footprint >= 5_000_000 or (op_type in heavy_ops and footprint >= 500_000):
        base = lo
    elif footprint <= 200_000 and op_type in light_ops:
        base = hi
    else:
        base = mid

    if policy == "smart":
        return [base]

    variants = [base]
    if max_variants > 1:
        contrast = hi if base != hi else lo
        variants.append(contrast)

    # Keep stable order and cap length.
    out = []
    seen = set()
    for b in variants:
        if b not in seen:
            out.append(b)
            seen.add(b)
        if len(out) >= max_variants:
            break
    return out or [default_batch]


def _count_parallel_loops(op_text: str) -> int:
    """Count 'parallel' entries in iterator_types attribute."""
    m = re.search(r'iterator_types\s*=\s*\[([^\]]*)\]', op_text)
    if not m:
        return 0
    return m.group(1).count('"parallel"')


def _count_reduction_loops(op_text: str) -> int:
    """Count 'reduction' entries in iterator_types attribute."""
    m = re.search(r'iterator_types\s*=\s*\[([^\]]*)\]', op_text)
    if not m:
        return 0
    return m.group(1).count('"reduction"')


# ---------------------------------------------------------------------------
# Op collection
# ---------------------------------------------------------------------------

def _collect_generic(lines: list[str], start: int) -> tuple[str, int]:
    """Collect a linalg.generic from its header line through '} -> type'.

    Returns (full_op_text, last_line_index).
    """
    buf = [lines[start]]
    for i in range(start + 1, len(lines)):
        buf.append(lines[i])
        stripped = lines[i].strip()
        # The closing line of the region body is '} -> <type>'
        if stripped.startswith("} ->"):
            return "\n".join(buf), i
    # Fallback — malformed input
    return "\n".join(buf), len(lines) - 1


# ---------------------------------------------------------------------------
# Affine-map helpers
# ---------------------------------------------------------------------------

def _get_referenced_maps(op_text: str, map_defs: dict[str, str]) -> list[str]:
    """Return affine_map declaration lines that are referenced by *op_text*."""
    result = []
    for name, decl in map_defs.items():
        # Match '#name' as a whole token (not a prefix of another map name)
        if re.search(r"#" + re.escape(name) + r"\b", op_text):
            result.append(decl)
    return result


# ---------------------------------------------------------------------------
# Outer-constant helpers (for linalg.generic bodies)
# ---------------------------------------------------------------------------

def _get_needed_constants(op_text: str, const_map: dict[str, str]) -> list[str]:
    """Return constant definition lines needed by the op body.

    Only returns scalar (non-tensor) constants; ops that need tensor constants
    with elided weights will be filtered out by the caller.
    """
    needed = []
    for ssa, defn in const_map.items():
        # Match the bare name after '%'
        if re.search(r"%" + re.escape(ssa[1:]) + r"\b", op_text):
            needed.append(defn)
    return needed


def _has_elided_tensor(const_lines: list[str]) -> bool:
    """True if any constant is a tensor with dense_resource<__elided__>."""
    for line in const_lines:
        if "dense_resource<__elided__>" in line and "tensor<" in line:
            return True
    return False


# ---------------------------------------------------------------------------
# Op text patching
# ---------------------------------------------------------------------------

def _replace_section_inplace(text: str, keyword: str,
                              new_args: list[str], new_types: list[str]) -> str:
    """Replace the *keyword*(...) block with rebuilt content."""
    search = " " + keyword + "("
    idx = text.find(search)
    if idx == -1:
        return text
    paren_start = idx + len(search) - 1
    paren_end = _find_closing_paren(text, paren_start)
    if paren_end == -1:
        return text
    new_block = f" {keyword}({', '.join(new_args)} : {', '.join(new_types)})"
    return text[:idx] + new_block + text[paren_end + 1 :]


def _patch_op(op_text: str,
              ins_args: list[str], spec_ins: list[str],
              outs_args: list[str], spec_outs: list[str],
              spec_result: str,
              batch_size: int = 1) -> str:
    """Return op_text with:
      - result SSA renamed to %result
      - ins SSA args renamed %arg0, %arg1, ...
      - outs SSA args renamed %arg<N>, ...
      - all tensor types specialized (? → batch_size)
      - trailing '-> type' updated
    """
    # 1. Rename result SSA on the first line
    text = re.sub(r"^(\s*)%\w+(\s*=\s*)", r"\1%result\2", op_text, count=1)

    # 2. Rebuild ins() block
    n_ins = len(ins_args)
    new_ins_args = [f"%arg{i}" for i in range(n_ins)]
    text = _replace_section_inplace(text, "ins", new_ins_args, spec_ins)

    # 3. Rebuild outs() block
    new_outs_args = [f"%arg{n_ins + i}" for i in range(len(outs_args))]
    text = _replace_section_inplace(text, "outs", new_outs_args, spec_outs)

    # 4. Replace result type on the trailing '-> type'
    if "\n" in text:
        lines = text.split("\n")
        lines[-1] = re.sub(r"->\s*tensor<[^>]+>", f"-> {spec_result}", lines[-1])
        text = "\n".join(lines)
    else:
        text = re.sub(r"->\s*tensor<[^>]+>\s*$", f"-> {spec_result}", text)

    # 5. Final sweep: replace any remaining '?' with the concrete batch size
    text = text.replace('?', str(batch_size))

    return text


def _normalized_op_signature(op_text: str) -> str:
    """Normalize an op text blob for deduplication/signature hashing."""
    normalized = textwrap.dedent(op_text)
    normalized = re.sub(r"%\w+", "%v", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


# ---------------------------------------------------------------------------
# Benchmark module builder
# ---------------------------------------------------------------------------

def _build_benchmark(affine_maps: list[str],
                     constants: list[str],
                     op_text: str,
                     all_types: list[str],
                     result_type: str) -> str:
    """Assemble a complete benchmark MLIR module."""
    args_decl = ", ".join(f"%arg{i}: {t}" for i, t in enumerate(all_types))

    header = ""
    if affine_maps:
        header = "\n".join(affine_maps) + "\n"

    const_block = ""
    if constants:
        const_block = "\n" + "\n".join(f"    {c.strip()}" for c in constants)

    # Dedent op_text to remove shared leading whitespace, then re-indent to 4 spaces.
    # This preserves relative indentation (e.g. generic region body is indented further).
    dedented = textwrap.dedent(op_text)
    indented_op = "\n".join(
        "    " + line if line.strip() else ""
        for line in dedented.split("\n")
    )

    return (
        f"{header}"
        f"module {{\n"
        f"  func.func private @nanoTime() -> i64 attributes {{llvm.emit_c_interface}}\n"
        f"  func.func @main({args_decl}) -> ({result_type}, i64)"
        f" attributes {{llvm.emit_c_interface}} {{\n"
        f"    %t0 = call @nanoTime() : () -> i64"
        f"{const_block}\n"
        f"{indented_op}\n"
        f"    %t1 = call @nanoTime() : () -> i64\n"
        f"    %delta = arith.subi %t1, %t0 : i64\n"
        f"    return %result, %delta : {result_type}, i64\n"
        f"  }}\n"
        f"}}\n"
    )


# ---------------------------------------------------------------------------
# Single-file extraction (callable from batch mode and CLI)
# ---------------------------------------------------------------------------

def extract_from_file(input_path: str, output_dir: str, batch_size: int = 1,
                       model_name: str | None = None, min_parallel_loops: int = 2,
                       require_reduction: bool = True,
                       generic_ratio: float | None = None,
                       clean: bool = False,
                      dynamic_shape_policy: str = "static",
                      dynamic_batch_candidates: list[int] | None = None,
                      dynamic_max_variants: int = 2,
                      append_batch_tag: bool = False,
                      real_output_dir: str | None = None,
                      dynamic_output_dir: str | None = None,
                      manifest_path: str | None = None) -> tuple[int, int, dict[str, int]]:
    """Extract ops from one *_linalg.mlir file.

        Returns (written, skipped, op_counts).

        dynamic_shape_policy:
            - static: replace all '?' with batch_size
            - smart: one heuristic batch per dynamic op
            - hybrid: one heuristic + one contrastive batch per dynamic op (capped)

        real_output_dir / dynamic_output_dir:
            Optional destination directories for splitting outputs:
            - static ops (no '?' in original op types) -> real_output_dir
            - dynamic ops (contained '?') -> dynamic_output_dir
    """
    os.makedirs(output_dir, exist_ok=True)
    if real_output_dir:
        os.makedirs(real_output_dir, exist_ok=True)
    if dynamic_output_dir:
        os.makedirs(dynamic_output_dir, exist_ok=True)

    clean_dirs = [output_dir]
    if real_output_dir:
        clean_dirs.append(real_output_dir)
    if dynamic_output_dir:
        clean_dirs.append(dynamic_output_dir)

    if clean:
        for cdir in dict.fromkeys(clean_dirs):
            if not os.path.isdir(cdir):
                continue
            for f in os.listdir(cdir):
                if f.endswith(".mlir"):
                    os.remove(os.path.join(cdir, f))
    if model_name is None:
        model_name = os.path.basename(input_path).replace("_linalg.mlir", "")

    with open(input_path) as fh:
        raw_lines = fh.readlines()
    lines = [l.rstrip() for l in raw_lines]

    # ------------------------------------------------------------------
    # Pass 1: collect affine map declarations and scalar outer constants
    # ------------------------------------------------------------------
    affine_map_defs: dict[str, str] = {}   # map-name (without #) → declaration line
    outer_constants: dict[str, str] = {}   # %ssa_name → full definition line (stripped)

    for line in lines:
        # Affine map: '#mapN = affine_map<...>'
        m = re.match(r"^(#\w+)\s*=\s*affine_map<", line)
        if m:
            name = m.group(1)[1:]  # strip leading '#'
            affine_map_defs[name] = line
            continue

        # arith.constant (scalar and tensor, we'll filter later)
        m = re.match(r"\s+(%\w+)\s*=\s*(arith\.constant\s+.+)", line)
        if m:
            ssa = m.group(1)
            outer_constants[ssa] = f"    {ssa} = {m.group(2)}"

    # ------------------------------------------------------------------
    # Pass 2: iterate lines, find target ops, extract benchmarks
    # ------------------------------------------------------------------
    seen_shapes: set[tuple] = set()          # normalized dedup key for real/dynamic outputs
    op_counts: dict[str, int] = {}           # op_type → next index
    written = 0
    skipped = 0
    dynamic_ops_seen = 0
    dynamic_variants_written = 0
    real_written = 0
    manifest_entries: list[dict] = []

    # Buffer for linalg.generic ops when generic_ratio is set.
    # Each entry: (op_text, result_type_raw, ins_args, ins_types, outs_args, outs_types,
    #              is_reduction, op_has_dynamic, dedup_key)
    pending_generics: list[tuple] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Quick filter: must contain 'linalg.'
        if "linalg." not in line:
            i += 1
            continue

        # Skip linalg.yield (inside region bodies)
        if "linalg.yield" in line:
            i += 1
            continue

        # Identify which target op is on this line
        matched_op: str | None = None
        for op in TARGET_OPS:
            if op in line:
                # Make sure it's a whole-word match for the op keyword part
                kw = op.split("linalg.")[1]
                if re.search(r"linalg\." + re.escape(kw) + r"\b", line):
                    matched_op = op
                    break

        if matched_op is None:
            i += 1
            continue

        # Skip excluded simple ops
        if any(excl in line for excl in (
            "linalg.broadcast", "linalg.fill", "linalg.transpose",
            "linalg.reduce", "linalg.index",
        )):
            i += 1
            continue

        # Collect full op text (multi-line for linalg.generic)
        if matched_op == "linalg.generic":
            op_text, end_i = _collect_generic(lines, i)
        else:
            op_text = line
            end_i = i

        i = end_i + 1

        # Skip / buffer trivial linalg.generic ops
        is_generic = matched_op == "linalg.generic"
        is_reduction = False
        if is_generic:
            is_reduction = _count_reduction_loops(op_text) > 0
            if _count_parallel_loops(op_text) < min_parallel_loops:
                skipped += 1
                continue
            if require_reduction and not is_reduction and generic_ratio is None:
                skipped += 1
                continue

        # Skip multi-result ops: '-> (type1, type2)' tuple result
        last_op_line = op_text.split("\n")[-1]
        if re.search(r"->\s*\(", last_op_line):
            skipped += 1
            continue

        # -- Parse ins/outs -----------------------------------------------
        ins_args, ins_types = _parse_section(op_text, "ins")
        outs_args, outs_types = _parse_section(op_text, "outs")

        if not ins_types and not outs_types:
            skipped += 1
            continue

        result_type_raw = _extract_result_type(op_text)
        if not result_type_raw and outs_types:
            result_type_raw = outs_types[-1]

        # -- Specialize dynamic dims --------------------------------------
        op_type = matched_op.split("linalg.")[1]
        raw_types = ins_types + outs_types + [result_type_raw]
        op_has_dynamic = _has_dynamic(raw_types)
        if op_has_dynamic:
            dynamic_ops_seen += 1
            chosen_batches = _choose_dynamic_batches(
                op_type,
                ins_types,
                outs_types,
                result_type_raw,
                dynamic_shape_policy,
                dynamic_batch_candidates or [batch_size],
                batch_size,
                max(1, dynamic_max_variants),
            )
        else:
            chosen_batches = [batch_size]

        # Build one benchmark per chosen batch variant.
        variant_written = False
        for chosen_batch in chosen_batches:
            spec_ins = [_specialize(t, chosen_batch) for t in ins_types]
            spec_outs = [_specialize(t, chosen_batch) for t in outs_types]
            spec_result = _specialize(result_type_raw, chosen_batch)

            all_spec = spec_ins + spec_outs + [spec_result]
            if _has_dynamic(all_spec):
                continue

            # -- Skip ops with unsupported element types (e.g. i1) ---------
            if re.search(r'tensor<[^>]*xi1>', ' '.join(spec_ins + spec_outs + [spec_result])):
                continue

            # -- Outer constants needed in generic body -------------------
            needed_constants: list[str] = []
            if matched_op == "linalg.generic":
                needed_constants = _get_needed_constants(op_text, outer_constants)
                if _has_elided_tensor(needed_constants):
                    continue
                # Skip if the region body references SSA values not defined locally
                # (i.e. results of other ops in the original function — unresolvable).
                # Collect everything that IS in scope for the body:
                #   1. ins/outs SSA args of the op itself
                #   2. hoisted outer arith.constants
                #   3. ^bb0 block-argument names  (e.g. %in, %out, %in_313)
                #   4. body-local definitions:     %N = <op>
                local_defs: set[str] = set(ins_args) | set(outs_args) | set(outer_constants)
                for bline in op_text.split("\n"):
                    if bline.strip().startswith("^bb0"):
                        for m in re.finditer(r'(%\w+)\s*:', bline):
                            local_defs.add(m.group(1))
                    for m in re.finditer(r'(%\w+)\s*=', bline):
                        local_defs.add(m.group(1))

                dangling = False
                in_bb0 = False
                for bline in op_text.split("\n"):
                    if bline.strip().startswith("^bb0"):
                        in_bb0 = True
                    if not in_bb0 or bline.strip().startswith("linalg.yield"):
                        continue
                    for ref in re.findall(r'(%\w+)', bline):
                        if ref in local_defs:
                            continue
                        dangling = True
                        break
                    if dangling:
                        break
                if dangling:
                    continue

            # -- Deduplicate by (op_type, shapes) -------------------------
            all_types = spec_ins + spec_outs

            # Build a normalized op signature to avoid duplicate writes across
            # real/static and dynamic-specialized outputs.
            op_patched = _patch_op(
                op_text,
                ins_args, spec_ins,
                outs_args, spec_outs,
                spec_result,
                batch_size=chosen_batch,
            )
            op_signature = _normalized_op_signature(op_patched)
            dedup_key = (op_type, tuple(all_types + [spec_result]), op_signature)
            if dedup_key in seen_shapes:
                continue
            seen_shapes.add(dedup_key)

            # -- Affine maps ----------------------------------------------
            ref_maps = _get_referenced_maps(op_text, affine_map_defs)

            # -- Build benchmark (may be deferred if generic_ratio applies) --
            idx = op_counts.get(op_type, 0)
            op_counts[op_type] = idx + 1

            mlir = _build_benchmark(ref_maps, needed_constants, op_patched, all_types, spec_result)

            target_dir = output_dir
            if op_has_dynamic and dynamic_output_dir:
                target_dir = dynamic_output_dir
            elif (not op_has_dynamic) and real_output_dir:
                target_dir = real_output_dir

            batch_suffix = f"_bs{chosen_batch}" if (append_batch_tag and op_has_dynamic) else ""
            out_path = os.path.join(target_dir, f"{model_name}_{op_type}{batch_suffix}_{idx}.mlir")

            signature_blob = "|".join([
                op_type,
                str(chosen_batch),
                ";".join(all_types),
                spec_result,
                op_signature,
            ])
            signature_hash = hashlib.sha1(signature_blob.encode("utf-8")).hexdigest()
            manifest_entry = {
                "source_file": os.path.abspath(input_path),
                "model": model_name,
                "op_type": op_type,
                "op_index": idx,
                "output_file": os.path.abspath(out_path),
                "origin": "dynamic" if op_has_dynamic else "real",
                "was_dynamic": op_has_dynamic,
                "chosen_batch": chosen_batch,
                "policy": dynamic_shape_policy if op_has_dynamic else "static",
                "input_types": all_types,
                "result_type": spec_result,
                "signature_hash": signature_hash,
            }

            # When generic_ratio is set, defer all generic ops for capped sampling.
            if generic_ratio is not None and is_generic:
                pending_generics.append((
                    mlir, out_path, manifest_entry, is_reduction,
                ))
                variant_written = True
                continue

            with open(out_path, "w") as fh:
                fh.write(mlir)

            manifest_entries.append(manifest_entry)

            written += 1
            if op_has_dynamic:
                dynamic_variants_written += 1
            else:
                real_written += 1
            variant_written = True

        if not variant_written:
            skipped += 1

    # ------------------------------------------------------------------
    # Flush deferred generic ops with ratio-based cap
    # ------------------------------------------------------------------
    if generic_ratio is not None and pending_generics:
        total_generics = len(pending_generics)
        keep_count = max(1, int(total_generics * generic_ratio))

        # Prioritise reduction-containing generics, then fill with element-wise.
        reduction_entries = [g for g in pending_generics if g[3]]
        elementwise_entries = [g for g in pending_generics if not g[3]]

        selected = reduction_entries[:keep_count]
        remaining_slots = keep_count - len(selected)
        if remaining_slots > 0:
            selected += random.sample(
                elementwise_entries,
                min(remaining_slots, len(elementwise_entries)),
            )

        skipped += total_generics - len(selected)

        for mlir, out_path, manifest_entry, _is_red in selected:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as fh:
                fh.write(mlir)
            manifest_entries.append(manifest_entry)
            written += 1
            if manifest_entry["was_dynamic"]:
                dynamic_variants_written += 1
            else:
                real_written += 1

    print(f"Model : {model_name}")
    print(f"Written: {written} benchmark file(s) → {output_dir}/")
    print(f"Skipped: {skipped} op(s)")
    if real_output_dir or dynamic_output_dir:
        print(f"Split outputs: real={real_written} dynamic={dynamic_variants_written}")
    if dynamic_ops_seen:
        print(
            f"Dynamic policy: {dynamic_shape_policy} | "
            f"dynamic ops seen: {dynamic_ops_seen} | "
            f"dynamic variants written: {dynamic_variants_written}"
        )

    if manifest_path:
        manifest_parent = os.path.dirname(manifest_path)
        if manifest_parent:
            os.makedirs(manifest_parent, exist_ok=True)
        manifest = {
            "model": model_name,
            "source_file": os.path.abspath(input_path),
            "policy": dynamic_shape_policy,
            "batch_size": batch_size,
            "dynamic_batch_candidates": _normalize_batch_candidates(dynamic_batch_candidates, batch_size),
            "dynamic_max_variants": max(1, dynamic_max_variants),
            "written": written,
            "skipped": skipped,
            "real_written": real_written,
            "dynamic_variants_written": dynamic_variants_written,
            "entries": manifest_entries,
        }
        with open(manifest_path, "w") as mf:
            json.dump(manifest, mf, indent=2)
        print(f"Wrote extraction manifest: {manifest_path}")

    print(f"Op breakdown: {dict(sorted(op_counts.items()))}")
    return written, skipped, op_counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract individual linalg ops from full-model MLIR files."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Input *_linalg.mlir file (single-file mode)")
    group.add_argument(
        "--input-dir",
        help="Directory of *_linalg.mlir files (batch mode — processes all files)"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for benchmark files")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1],
        help="One or more batch sizes to substitute for '?' dimensions. "
             "Each size produces its own set of benchmark files. Default: 1. "
             "Example: --batch-sizes 1 4 16"
    )
    parser.add_argument(
        "--dynamic-shape-policy",
        choices=["static", "smart", "hybrid"],
        default="static",
        help="How to specialize dynamic-shape ops: static (use a fixed batch), "
             "smart (one heuristic batch), hybrid (heuristic + contrastive batch)."
    )
    parser.add_argument(
        "--dynamic-max-variants", type=int, default=2,
        help="Maximum variants per dynamic op when using --dynamic-shape-policy hybrid. "
             "Default: 2"
    )
    parser.add_argument(
        "--real-output-dir", default=None,
        help="If set, write ops with no dynamic dims ('?') to this directory."
    )
    parser.add_argument(
        "--dynamic-output-dir", default=None,
        help="If set, write ops that originally had dynamic dims ('?') to this directory."
    )
    parser.add_argument(
        "--manifest-dir", default=None,
        help="If set, writes one extraction manifest JSON file per model into this directory."
    )
    parser.add_argument(
        "--model-name", default=None,
        help="Model name prefix for output files (default: derived from input filename)"
    )
    parser.add_argument(
        "--min-parallel-loops", type=int, default=2,
        help="Minimum parallel loop count for linalg.generic ops (default: 2)"
    )
    parser.add_argument(
        "--no-require-reduction", dest="require_reduction", action="store_false", default=True,
        help="Keep linalg.generic ops with no reduction loops (elementwise). "
             "Default: require at least one reduction loop."
    )
    parser.add_argument(
        "--generic-ratio", type=float, default=None,
        help="If set, keep at most this fraction of linalg.generic ops (0.0-1.0). "
             "Reduction-containing generics are prioritised, element-wise fill remaining "
             "slots via random sampling. Overrides --no-require-reduction for element-wise "
             "generics; still respects --min-parallel-loops."
    )
    parser.add_argument(
        "--clean", action="store_true", default=False,
        help="Remove all existing .mlir files in the output directory before extracting"
    )
    args = parser.parse_args()

    if args.input:
        if args.dynamic_shape_policy == "static":
            # Single-file mode — loop over each requested batch size
            for i, bs in enumerate(args.batch_sizes):
                base_name = args.model_name or os.path.basename(args.input).replace("_linalg.mlir", "")
                effective_name = f"{base_name}_bs{bs}" if len(args.batch_sizes) > 1 else base_name
                extract_from_file(
                    args.input, args.output_dir,
                    batch_size=bs,
                    model_name=effective_name,
                    min_parallel_loops=args.min_parallel_loops,
                    require_reduction=args.require_reduction,
                    generic_ratio=args.generic_ratio,
                    clean=args.clean and i == 0,
                    dynamic_shape_policy="static",
                    dynamic_batch_candidates=args.batch_sizes,
                    dynamic_max_variants=max(1, args.dynamic_max_variants),
                    append_batch_tag=False,
                    real_output_dir=args.real_output_dir,
                    dynamic_output_dir=args.dynamic_output_dir,
                    manifest_path=(os.path.join(args.manifest_dir, f"{effective_name}_extract_manifest.json")
                                   if args.manifest_dir else None),
                )
        else:
            base_name = args.model_name or os.path.basename(args.input).replace("_linalg.mlir", "")
            extract_from_file(
                args.input, args.output_dir,
                batch_size=args.batch_sizes[0],
                model_name=base_name,
                min_parallel_loops=args.min_parallel_loops,
                require_reduction=args.require_reduction,
                generic_ratio=args.generic_ratio,
                clean=args.clean,
                dynamic_shape_policy=args.dynamic_shape_policy,
                dynamic_batch_candidates=args.batch_sizes,
                dynamic_max_variants=max(1, args.dynamic_max_variants),
                append_batch_tag=True,
                real_output_dir=args.real_output_dir,
                dynamic_output_dir=args.dynamic_output_dir,
                manifest_path=(os.path.join(args.manifest_dir, f"{base_name}_extract_manifest.json")
                               if args.manifest_dir else None),
            )
    else:
        # Batch mode: process all *_linalg.mlir files in the directory
        input_files = sorted(
            f for f in os.listdir(args.input_dir)
            if f.endswith("_linalg.mlir")
        )
        if not input_files:
            print(f"No *_linalg.mlir files found in {args.input_dir}")
            return
        total_written = 0
        total_skipped = 0
        for fname in input_files:
            model = fname.replace("_linalg.mlir", "")
            model_output_dir = os.path.join(args.output_dir, model)
            if args.dynamic_shape_policy == "static":
                for i, bs in enumerate(args.batch_sizes):
                    effective_name = f"{model}_bs{bs}" if len(args.batch_sizes) > 1 else model
                    w, s, _ = extract_from_file(
                        os.path.join(args.input_dir, fname),
                        model_output_dir,
                        batch_size=bs,
                        model_name=effective_name,
                        min_parallel_loops=args.min_parallel_loops,
                        require_reduction=args.require_reduction,
                        generic_ratio=args.generic_ratio,
                        clean=args.clean and i == 0,
                        dynamic_shape_policy="static",
                        dynamic_batch_candidates=args.batch_sizes,
                        dynamic_max_variants=max(1, args.dynamic_max_variants),
                        append_batch_tag=False,
                        real_output_dir=(os.path.join(args.real_output_dir, model)
                                         if args.real_output_dir else None),
                        dynamic_output_dir=(os.path.join(args.dynamic_output_dir, model)
                                            if args.dynamic_output_dir else None),
                        manifest_path=(os.path.join(args.manifest_dir, f"{effective_name}_extract_manifest.json")
                                   if args.manifest_dir else None),
                    )
                    total_written += w
                    total_skipped += s
            else:
                w, s, _ = extract_from_file(
                    os.path.join(args.input_dir, fname),
                    model_output_dir,
                    batch_size=args.batch_sizes[0],
                    model_name=model,
                    min_parallel_loops=args.min_parallel_loops,
                    require_reduction=args.require_reduction,
                    generic_ratio=args.generic_ratio,
                    clean=args.clean,
                    dynamic_shape_policy=args.dynamic_shape_policy,
                    dynamic_batch_candidates=args.batch_sizes,
                    dynamic_max_variants=max(1, args.dynamic_max_variants),
                    append_batch_tag=True,
                    real_output_dir=(os.path.join(args.real_output_dir, model)
                                     if args.real_output_dir else None),
                    dynamic_output_dir=(os.path.join(args.dynamic_output_dir, model)
                                        if args.dynamic_output_dir else None),
                    manifest_path=(os.path.join(args.manifest_dir, f"{model}_extract_manifest.json")
                                   if args.manifest_dir else None),
                )
                total_written += w
                total_skipped += s
        print(f"\nBatch complete: {total_written} files written, {total_skipped} ops skipped "
              f"across {len(input_files)} model(s)")


if __name__ == "__main__":
    main()
