"""
preprocess_model.py
-------------------
Runs the C++ AST dumper on a full model .mlir file to inject
{tag = "operation_NNN"} attributes on every linalg op.

The AST dumper outputs:
   <operation graph and features info>
   ########################################
   <tagged full code with affine maps, constants, operations>

This script extracts the tagged code portion and writes it out.

Usage:
  python scripts/full_model/preprocess_model.py --input data/nn/raw_bench/gcn_linalg.mlir --output data/nn/tagged/gcn_tagged.mlir
"""

import os
import subprocess
import argparse


def run_ast_dumper(file_path: str) -> str:
    ast_dumper = os.getenv("AST_DUMPER_BIN_PATH")
    if not ast_dumper:
        raise RuntimeError("AST_DUMPER_BIN_PATH is not set. Source .env first.")

    proc = subprocess.run(
        f"{ast_dumper} {file_path}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    out = proc.stdout.decode("utf-8")
    err = proc.stderr.decode("utf-8")

    if proc.returncode != 0:
        raise RuntimeError(f"AST dumper failed for {file_path}: {err}")

    return out


def extract_tagged_code(raw_ast_info: str) -> str:
    if "########################################" in raw_ast_info:
        _, tagged_code = raw_ast_info.split("########################################", 1)
        return tagged_code.strip()
    else:
        raise RuntimeError(
            "AST dumper output missing '########################################' separator. "
            "The file may not contain any linalg operations."
        )


def extract_operation_tags(raw_ast_info: str) -> list[str]:
    """Extract operation tags from AST dumper output for downstream use."""
    if "########################################" in raw_ast_info:
        info, _ = raw_ast_info.split("########################################", 1)
    else:
        info = raw_ast_info

    if "#BEGIN_GRAPH" not in info:
        return []

    operations_lines, _ = info.split("#BEGIN_GRAPH", 1)
    tags = []
    for block in operations_lines.split("#START_OPERATION"):
        block = block.strip()
        if not block:
            continue
        if "#START_TAG" in block:
            _, tag_part = block.split("#START_TAG", 1)
            tag = tag_part.strip().split("\n")[0]
            tags.append(tag)
    return tags


def main():
    parser = argparse.ArgumentParser(
        description="Inject {tag} attributes into linalg ops via AST dumper."
    )
    parser.add_argument("--input", required=True, help="Path to full model .mlir file.")
    parser.add_argument("--output", required=True, help="Path to write tagged .mlir file.")
    parser.add_argument("--list-tags", action="store_true", help="Print operation tags found.")
    args = parser.parse_args()

    raw_ast_info = run_ast_dumper(args.input)
    tagged_code = extract_tagged_code(raw_ast_info)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write(tagged_code)
    print(f"Tagged code written to {args.output}")

    if args.list_tags:
        tags = extract_operation_tags(raw_ast_info)
        print(f"Found {len(tags)} operations:")
        for t in tags:
            print(f"  {t}")


if __name__ == "__main__":
    main()
