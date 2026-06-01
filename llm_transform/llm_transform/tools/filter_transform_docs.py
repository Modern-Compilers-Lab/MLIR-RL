#!/usr/bin/env python3
"""Build a markdown reference containing only the documentation for a curated
set of MLIR transform-dialect operations.

The source is the local ``Transform.md`` file, which is the full transform
dialect dump. Every operation appears as a level-3 heading of the form::

    ### `transform.structured.tile_using_for` (transform::TileUsingForOp)

with level-4 sub-sections (``#### Operands:`` etc.) underneath it, grouped by
level-2 category headings (``## Structured (Linalg) Transform Operations``).

This script extracts the section for each operation listed in ``OPS`` (keeping
the requested order) and writes them all into one markdown file. No network
access and no third-party dependencies are required.

Usage:
    python build_transform_docs.py [input.md] [output.md]

Defaults: input=Transform.md, output=transform_docs.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PARENT_DIR = Path(__file__).parents[2]

# Operations to include.
OPS = (PARENT_DIR / "resources" / "whitelist.txt").read_text().splitlines()

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def heading_op_name(text: str) -> str | None:
    """Return the transform op name from a heading line's text, or None.

    Heading text looks like ``\`transform.structured.tile_using_for\`
    (transform::TileUsingForOp)``; the op name is the first token once the
    backticks are stripped. Sub-headings such as ``Operands:`` return None.
    """
    cleaned = text.replace("`", "").strip()
    if not cleaned:
        return None
    first = cleaned.split()[0]
    return first if first.startswith("transform.") else None


def split_sections(markdown: str) -> dict[str, list[str]]:
    """Map each transform op name to its section's lines (heading + body).

    A section runs from its operation heading until the next heading whose
    level is the same or shallower (so the level-4 ``#### Operands:`` blocks
    stay inside their operation's section).
    """
    lines = markdown.splitlines()

    # Pre-scan every heading's position and level.
    headings = []  # (line_index, level, op_name_or_None)
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            headings.append((i, len(m.group(1)), heading_op_name(m.group(2))))

    sections: dict[str, list[str]] = {}
    for idx, (start, level, op) in enumerate(headings):
        if op is None:
            continue
        end = len(lines)
        for j in range(idx + 1, len(headings)):
            if headings[j][1] <= level:
                end = headings[j][0]
                break
        sections.setdefault(op, lines[start:end])  # keep first occurrence
    return sections


def normalize_section(section_lines: list[str]) -> str:
    """Re-level a section so its operation heading becomes a level-2 (##)
    heading, shifting nested sub-headings to match."""
    if not section_lines:
        return ""
    m = _HEADING_RE.match(section_lines[0])
    base_level = len(m.group(1)) if m else 2

    out = []
    for line in section_lines:
        hm = _HEADING_RE.match(line)
        if hm:
            new_level = max(2, len(hm.group(1)) - base_level + 2)
            out.append("#" * new_level + " " + hm.group(2))
        else:
            out.append(line)
    return "\n".join(out).strip()


def build(input_path: Path, output_path: Path) -> None:
    with open(input_path, encoding="utf-8") as f:
        markdown = f.read()
    sections = split_sections(markdown)

    chunks = [
        "# MLIR Transform Dialect — Selected Operations",
        "",
        f"Documentation extracted from `{input_path.name}`.",
        "",
    ]

    missing = []
    for op in OPS:
        section = sections.get(op)
        if not section:
            missing.append(op)
            chunks.append(f"## `{op}`\n\n*(not found in {input_path})*\n")
            continue
        chunks.append(normalize_section(section))
        chunks.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(chunks).rstrip() + "\n")

    found = len(OPS) - len(missing)
    print(f"Wrote {found}/{len(OPS)} operations to {output_path}")
    if missing:
        print("Missing (not matched in source):")
        for op in missing:
            print(f"  - {op}")


if __name__ == "__main__":
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (PARENT_DIR / "resources" / "Transform.md")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else (PARENT_DIR / "resources" / "transform_filtered.md")
    build(in_path, out_path)
