from tqdm import tqdm
import sys
import os
import json

import re
from typing import List

from rl_autoschedular.state import extract_bench_features_from_file, extract_bench_features_from_code
from rl_autoschedular.transforms import transform_img2col

def split_convolution_operations(img2col_conv_code: str) -> List[str]:
    """
    Deterministic splitter that:
      - preserves original SSA names when safe,
      - detects how many #map lines belong to op0 by scanning op0 body,
      - keeps tensor.collapse_shape lines in op1 (but avoids SSA redefinition by
        mapping conflicting function-arg names to the collapse source),
      - includes tensor.expand_shape (if present) and makes op1 return the expanded SSA/type,
      - removes empty lines when parsing.

    Returns [mlir_op0_text, mlir_op1_text].
    """
    # remove empty lines immediately
    lines = [ln for ln in img2col_conv_code.splitlines() if ln.strip()]

    # 1) collect contiguous maps
    maps: List[str] = []
    idx = 0
    while idx < len(lines) and lines[idx].strip().startswith("#map"):
        maps.append(lines[idx])
        idx += 1

    # store map names in order for detection
    map_names: List[str] = []
    for mline in maps:
        mm = re.match(r'^\s*(#map\d*|\#map)\b', mline)
        map_names.append(mm.group(1) if mm else "")

    # 2) find the linalg.generic lines that have the tags
    op0_line_idx = next(i for i, ln in enumerate(lines) if 'tag = "operation_0"' in ln)
    op1_line_idx = next(i for i, ln in enumerate(lines) if 'tag = "operation_1"' in ln)

    # 3) helper: extract balanced brace block starting idx
    def extract_block_from(start_idx: int) -> List[str]:
        depth = 0
        block = []
        for ln in lines[start_idx:]:
            depth += ln.count("{")
            depth -= ln.count("}")
            block.append(ln)
            if depth == 0:
                break
        return block

    op0_block = extract_block_from(op0_line_idx)
    op1_block = extract_block_from(op1_line_idx)

    op0_text = "\n".join(op0_block)
    op1_text = "\n".join(op1_block)

    # 4) determine which maps are referenced by op0 (dynamic)
    referenced_maps_in_op0 = set(re.findall(r"#map\d*", op0_text))
    if "#map" in op0_text:
        referenced_maps_in_op0.add("#map")

    last_ref_idx = -1
    for i, name in enumerate(map_names):
        if name and name in referenced_maps_in_op0:
            last_ref_idx = i

    if last_ref_idx >= 0:
        maps_op0 = maps[: last_ref_idx + 1]
        maps_op1 = maps[last_ref_idx + 1 :]
    else:
        fallback = min(8, len(maps))
        maps_op0 = maps[:fallback]
        maps_op1 = maps[fallback:]

    # 5) find any tensor.collapse_shape lines (they appear before op0/op1)
    #    Parse them into collapse_infos so we can reason about 'result' and 'src'.
    collapse_infos: List[dict] = []
    for j in range(idx, op1_line_idx):
        ln = lines[j]
        if "tensor.collapse_shape" in ln:
            # full parse (preferred)
            m = re.match(
                r'\s*(%[A-Za-z0-9_]+)\s*=\s*tensor\.collapse_shape\s+(%[A-Za-z0-9_]+).*:\s*(tensor<[^>]+>)\s+into\s+(tensor<[^>]+>)',
                ln)
            if m:
                collapse_infos.append({
                    'result': m.group(1),
                    'src': m.group(2),
                    'src_type': m.group(3),
                    'out_type': m.group(4),
                    'line': ln.rstrip()
                })
            else:
                # looser parse: try only names
                mm = re.match(r'\s*(%[A-Za-z0-9_]+)\s*=\s*tensor\.collapse_shape\s+(%[A-Za-z0-9_]+).*', ln)
                if mm:
                    collapse_infos.append({
                        'result': mm.group(1),
                        'src': mm.group(2),
                        'src_type': None,
                        'out_type': None,
                        'line': ln.rstrip()
                    })

    # 6) find any tensor.expand_shape lines after op1 that consume op1 result
    expand_info = None

    def find_linalg_result_ssa(block_text: str) -> str:
        for ln in block_text.splitlines():
            m = re.match(r"\s*(%[0-9]+)\s*=\s*linalg\.generic", ln)
            if m:
                return m.group(1)
        return None

    linalg_ssa = find_linalg_result_ssa(op1_text)
    for j in range(op1_line_idx + len(op1_block), len(lines)):
        ln = lines[j]
        if "tensor.expand_shape" in ln and linalg_ssa and linalg_ssa in ln:
            m = re.match(
                r'\s*(%[A-Za-z0-9_]+)\s*=\s*tensor\.expand_shape\s+(' + re.escape(linalg_ssa) +
                r').*:\s*(tensor<[^>]+>)\s+into\s+(tensor<[^>]+>)', ln)
            if m:
                expand_info = {
                    'result': m.group(1),
                    'src': m.group(2),
                    'src_type': m.group(3),
                    'out_type': m.group(4),
                    'line': ln.rstrip()
                }
            else:
                mm = re.match(r'\s*(%[A-Za-z0-9_]+)\s*=\s*tensor\.expand_shape\s+([%A-Za-z0-9_]+).*', ln)
                if mm:
                    expand_info = {'result': mm.group(1), 'line': ln.rstrip()}
            break

    # 7) parse original function args (preserve order and names)
    func_line = next((ln for ln in lines if "func.func @main" in ln), "")
    arg_match = re.search(r"@main\((.*?)\)\s*->", func_line)
    original_arg_list: List[str] = []
    if arg_match:
        original_args = arg_match.group(1).strip()
        original_arg_list = [a.strip() for a in original_args.split(",") if a.strip()]

    # 8) Build op0 argument list (prefer original %arg0)
    op0_args = [a for a in original_arg_list if a.split(":")[0].strip() == "%arg0"]
    if not op0_args:
        m_tex = re.search(r"tensor<[^>]+>", op0_text)
        img_type = m_tex.group(0) if m_tex else "tensor<?>"
        op0_args = [f"%arg0: {img_type}"]

    # op0 body: include any tensor.empty() lines nearest before op0 (keeps naming)
    tensor_empty_line = None
    for j in range(op0_line_idx - 1, -1, -1):
        if "tensor.empty" in lines[j]:
            tensor_empty_line = lines[j].rstrip()
            break
    op0_setup_lines = [tensor_empty_line] if tensor_empty_line else []
    op0_text_block = "\n".join(op0_setup_lines + op0_block)

    # 9) Build op1 argument list carefully while avoiding SSA redefinition:
    #    - if original args include a collapse-result name, *replace* that arg with
    #      the collapse *source* (preserve source type when available). This allows us
    #      to keep the collapse line in op1 and avoid redefinition while keeping naming traceable.
    #    - preserve original arg order as much as possible.
    # parse op1 ins(...) line to learn im2col inputs order and types
    op1_ins_line = next((ln for ln in op1_block if "ins(" in ln), "")
    in_names: List[str] = []
    in_types: List[str] = []
    m_ins = re.search(r"ins\((.*?)\)", op1_ins_line)
    if m_ins:
        inside = m_ins.group(1)
        if ":" in inside:
            names_part, types_part = inside.split(":", 1)
            in_names = re.findall(r"%[A-Za-z0-9_]+", names_part)
            in_types = re.findall(r"tensor<[^>]+>", types_part)
        else:
            in_names = re.findall(r"%[A-Za-z0-9_]+", inside)

    # Map collapse result -> src for quick lookup
    collapse_result_to_src = {ci['result']: ci for ci in collapse_infos}

    # Build op1_arg_list starting from original_arg_list but remap any arg that equals a collapse result
    op1_arg_list: List[str] = []
    seen_args = set()
    for orig in original_arg_list:
        name = orig.split(":")[0].strip()
        # If this original argument *is* a collapse result, replace it with the collapse source
        if name in collapse_result_to_src:
            ci = collapse_result_to_src[name]
            src_name = ci['src']
            # try to preserve original arg type for the src if it's in original_arg_list
            src_orig_entry = next((a for a in original_arg_list if a.split(":")[0].strip() == src_name), None)
            if src_orig_entry:
                entry = src_orig_entry
            else:
                # fallback to parsed src_type if present
                if ci.get('src_type'):
                    entry = f"{src_name}: {ci['src_type']}"
                else:
                    entry = f"{src_name}: tensor<?>"
            # append only if not already added
            if entry not in op1_arg_list:
                op1_arg_list.append(entry)
                seen_args.add(entry.split(":")[0].strip())
        else:
            # normal case: keep the original arg if it's relevant (we'll filter later)
            if orig not in op1_arg_list:
                op1_arg_list.append(orig)
                seen_args.add(name)

    # Now ensure the im2col inputs from ins(...) are present in op1_arg_list:
    for idx_name, nm in enumerate(in_names):
        # if nm is already in op1_arg_list (by SSA name), skip
        if any(nm == (a.split(":")[0].strip()) for a in op1_arg_list):
            continue
        t = in_types[idx_name] if idx_name < len(in_types) else "tensor<?>"
        op1_arg_list.append(f"{nm}: {t}")

    # If we still ended up empty (very unlikely), fallback to original args containing %arg*
    if not op1_arg_list:
        op1_arg_list = [a for a in original_arg_list if a.startswith("%arg")]

    # Deduplicate preserving order
    final_op1_arg_list: List[str] = []
    seen = set()
    for a in op1_arg_list:
        name = a.split(":")[0].strip()
        if name not in seen:
            final_op1_arg_list.append(a)
            seen.add(name)
    op1_arg_list = final_op1_arg_list

    # 10) Build op1 body text: include collapse lines first (preserve exact text),
    #     then the op1 linalg block, then the expand line (if any).
    #     Because we replaced collapse-arg names with their sources in signature above,
    #     including collapse lines verbatim is safe (no redefinition).
    op1_body_parts = []
    for ci in collapse_infos:
        # If, for some reason, collapse result name is still present as an argument name,
        # we must avoid redefining the arg. But our remap step above should prevent that.
        res_name = ci['result']
        arg_names_set = {a.split(":")[0].strip() for a in op1_arg_list}
        if res_name in arg_names_set:
            # As a safety net: rename the collapse result inside the line to a fresh SSA and
            # update any subsequent references. (This should rarely occur due to remapping.)
            base = res_name.lstrip("%")
            new_name = f"%{base}_from_collapse"
            collapse_line = ci['line'].replace(res_name, new_name, 1)
            op1_body_parts.append(collapse_line)
            # also replace uses of res_name inside op1_text with new_name
            op1_text = re.sub(rf'(?<![A-Za-z0-9_]){re.escape(res_name)}(?![A-Za-z0-9_])', new_name, op1_text)
            if expand_info and expand_info.get('src') == res_name:
                expand_info['src'] = new_name
                expand_info['line'] = expand_info['line'].replace(res_name, new_name, 1)
        else:
            op1_body_parts.append(ci['line'])

    op1_body_parts.append(op1_text)
    if expand_info:
        op1_body_parts.append(expand_info['line'])
    op1_body_text = "\n".join(op1_body_parts)

    # 11) determine op1 return SSA and type (expand wins if present)
    def extract_linalg_result_type(block_lines: List[str]) -> str:
        joined = "\n".join(block_lines)
        m = re.search(r"->\s*(tensor<[^>]+>)", joined)
        return m.group(1) if m else "tensor<?>"

    if expand_info:
        op1_result_ssa = expand_info['result']
        op1_result_type = expand_info.get('out_type') or extract_linalg_result_type(op1_block)
    else:
        op1_result_ssa = find_linalg_result_ssa(op1_text) or "%result"
        op1_result_type = extract_linalg_result_type(op1_block)

    # for op0, extract its result SSA and type
    op0_result_ssa = find_linalg_result_ssa(op0_text) or "%result"
    op0_result_type = extract_linalg_result_type(op0_block)

    # 12) Build final MLIR module texts (op0 and op1) preserving naming
    def build_module(maps: List[str], arg_list: List[str], result_type: str, body_text: str, result_ssa: str) -> str:
        arg_sig = "(" + ", ".join(arg_list) + ")"
        ret_sig = "(" + result_type + ", i64)"
        return (
            "\n".join(maps) + "\n\n"
            "module {\n"
            "  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}\n\n"
            f"  func.func @main{arg_sig}\n"
            f"      -> {ret_sig} attributes {{llvm.emit_c_interface}} {{\n\n"
            "    %0 = call @nanoTime() : () -> i64\n"
            f"{body_text}\n"
            "    %4 = call @nanoTime() : () -> i64\n"
            "    %5 = arith.subi %4, %0 : i64\n\n"
            f"    return {result_ssa}, %5 : {result_type}, i64\n"
            "  }\n"
            "}\n"
        )

    mlir_op0 = build_module(maps_op0, op0_args, op0_result_type, op0_text_block, op0_result_ssa)
    mlir_op1 = build_module(maps_op1, op1_arg_list, op1_result_type, op1_body_text, op1_result_ssa)

    return [mlir_op0, mlir_op1]

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python utils/data_utils.py <path_to_folder>")
    sys.exit(1)
  path_to_folder = sys.argv[1]
  if not os.path.isdir(path_to_folder):
    print(f"Error: {path_to_folder} is not a valid directory.")
    sys.exit(1)
    
  with open(f"{path_to_folder}/../benchmarks_split.json", 'r') as f:
    benchmarks_split = json.load(f)

  code_files = [f for f in os.listdir(path_to_folder) if f.endswith('.mlir')]
  files_tqdm = tqdm(code_files, unit='file')
  for code_file in files_tqdm:
    bench_name = code_file.replace('.mlir', '')
    files_tqdm.set_postfix_str(bench_name)
    full_path = os.path.join(path_to_folder, code_file)
    with open(full_path, 'r') as f:
        bench_features = extract_bench_features_from_file(bench_name, full_path, 0)
        # print(f"EXTRACT FEATUERES FROM FILE: \n", bench_features.code)
        code_i2c = transform_img2col(bench_features.code, "operation_0")
        # print(f"I2C Code: \n", code_i2c)
        bench_features_i2c = extract_bench_features_from_code(bench_name, code_i2c, 0)
        # print(f"EXTRACT FEATUERES FROM CODE: \n", bench_features_i2c.code)
        with open(f"{path_to_folder}/{bench_name}_i2c.mlir", 'w') as f:
          f.write(bench_features_i2c.code)
        
    try:
        op0, op1 = split_convolution_operations(bench_features_i2c.code)
        
        with open(f"{path_to_folder}/{bench_name}_op0.mlir", 'w') as f:
            f.write(op0)
        with open(f"{path_to_folder}/{bench_name}_op1.mlir", 'w') as f:
            f.write(op1)
            
        if bench_name in benchmarks_split['train']:
          benchmarks_split['train'].append(f"{bench_name}_i2c")
          benchmarks_split['train'].append(f"{bench_name}_op0")
          benchmarks_split['train'].append(f"{bench_name}_op1")
        else:
          benchmarks_split['eval'].append(f"{bench_name}_i2c")
          benchmarks_split['eval'].append(f"{bench_name}_op0")
          benchmarks_split['eval'].append(f"{bench_name}_op1")
          
    except Exception as e:
        print(f"Failed to split convolution code `{bench_name}`: {e}")
        
  with open(f"{path_to_folder}/../benchmarks_split.json", 'w') as f:
    json.dump(benchmarks_split, f, indent=4)
                 