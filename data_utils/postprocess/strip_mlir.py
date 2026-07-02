#!/usr/bin/env python3
"""
strip_mlir.py
-------------
Post-process MLIR files to remove large weight constants, reducing file size
significantly while keeping the computational structure intact.

Two stripping strategies:
  1. Hex-encoded constants  : dense<"0xABCD..."> → dense_resource<__elided__>
  2. Long numeric arrays    : dense<[1.0, 2.0, ... ]> (≥500 chars) → same

This script is a fallback for when the --mlir-elide-elementsattrs-if-larger
flag is unavailable or insufficient during torch-mlir-opt lowering.
"""

import re
import sys
import os
import argparse


def strip_weights(input_file: str, output_file: str = None,
                  verbose: bool = False) -> float:
    """Strip large dense constants from an MLIR file.

    Args:
        input_file:  Path to the source .mlir file.
        output_file: Path for the stripped output. Defaults to
                     <input>_stripped.mlir.
        verbose:     Print size statistics when True.

    Returns:
        Percentage reduction in file size.
    """
    if output_file is None:
        output_file = input_file.replace('.mlir', '_stripped.mlir')

    if verbose:
        print(f"Reading {input_file}...")

    with open(input_file, 'r') as f:
        content = f.read()

    original_size = len(content)

    # Strategy 1: hex-encoded dense constants  dense<"0xABCD...">
    stripped = re.sub(
        r'dense<"0x[0-9A-Fa-f]+">',
        'dense_resource<__elided__>',
        content
    )

    # Strategy 2: long numeric dense arrays (>= 500 characters inside brackets)
    stripped = re.sub(
        r'dense<\[([0-9eE.+\-,\s]{500,})\]>',
        'dense_resource<__elided__>',
        stripped
    )

    # Strategy 3: torch_tensor dense_resource references + remove binary resource section.
    # These appear as dense_resource<torch_tensor_NNN_torch.floatXX> in the IR and the actual
    # binary data lives in a {-# dialect_resources: { ... } #-} section at the end of the file.
    # Step 3a: replace references in-line
    stripped = re.sub(
        r'dense_resource<torch_tensor_[^>]+>',
        'dense_resource<__elided__>',
        stripped
    )
    # Step 3b: truncate at the resource section delimiter (avoids DOTALL regex on huge strings)
    resource_marker = stripped.find('{-#')
    if resource_marker != -1:
        stripped = stripped[:resource_marker].rstrip() + '\n'

    new_size = len(stripped)
    reduction = (1 - new_size / original_size) * 100 if original_size > 0 else 0.0

    if verbose:
        print(f"Original : {original_size:,} bytes")
        print(f"Stripped : {new_size:,} bytes")
        print(f"Reduction: {reduction:.1f}%")

    with open(output_file, 'w') as f:
        f.write(stripped)

    if verbose:
        print(f"Saved to {output_file}")

    return reduction


def main():
    parser = argparse.ArgumentParser(
        description="Strip large weight constants from MLIR files."
    )
    parser.add_argument("input",  help="Input .mlir file.")
    parser.add_argument("-o", "--output",
                        help="Output file (default: <input>_stripped.mlir).")
    parser.add_argument("-v", "--verbose",  action="store_true")
    parser.add_argument("-r", "--replace",  action="store_true",
                        help="Overwrite the input file with the stripped version.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: file not found: {args.input}")
        sys.exit(1)

    output_file = args.output
    if args.replace:
        output_file = args.input + ".tmp"

    try:
        reduction = strip_weights(args.input, output_file, args.verbose)

        if args.replace:
            os.replace(output_file, args.input)
            if args.verbose:
                print(f"Replaced {args.input} with stripped version.")

        print(f"Done ({reduction:.1f}% reduction).")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
