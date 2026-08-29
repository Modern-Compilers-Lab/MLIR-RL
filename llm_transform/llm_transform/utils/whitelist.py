"""Verify that an LLM-generated transform schedule only uses whitelisted ops.

The whitelist (``resources/whitelist.txt``) lists the transform-dialect
operations the LLM is allowed to emit, one ``<code> <op-name>`` per line. This
module extracts the operations actually used by a schedule and checks them
against that list, so a schedule using a non-whitelisted op is rejected before
it is ever applied.
"""

from __future__ import annotations

from pathlib import Path

from mlir.ir import Context, Module, Operation, WalkResult

PARENT_DIR = Path(__file__).parents[2]
_WHITELIST_FILE = PARENT_DIR / "resources" / "whitelist.txt"

# Structural ops that form the schedule scaffolding. They are required by the
# named-sequence interpreter (see ``transform_module``) rather than chosen by
# the LLM as optimizations, so they are always permitted. The
# ``transform.with_named_sequence`` module attribute and ``!transform.*`` types
# are not operations, so they never appear when walking the parsed module.
_STRUCTURAL_OPS = {"transform.named_sequence"}


def _extract_transform_ops(op: Operation):
    """Yield ``op`` and every operation nested within its regions (pre-order)."""
    transform_ops: list[str] = []
    def _add_transform_op(op: Operation):
        if op.name.startswith("transform."):
            transform_ops.append(op.name)
        return WalkResult.ADVANCE
    op.walk(_add_transform_op)
    return transform_ops


def load_whitelist(path: Path = _WHITELIST_FILE) -> set[str]:
    """Return the set of allowed transform op names from ``whitelist.txt``.

    Each non-blank line is ``<code> <op-name>``; only the op name is kept.
    """
    ops: set[str] = set()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        _code, _, op = line.partition(" ")
        op = op.strip()
        if op:
            ops.add(op)
    return ops


def extract_transform_ops(schedule: str) -> list[str]:
    """Return the transform-dialect op names used in a schedule, in source order.

    The schedule is parsed with MLIR and every nested operation is walked, so
    types (``!transform.any_op``) and attributes (``{transform.readonly}``) are
    not mistaken for operations. Raises ``ValueError`` if the schedule does not
    parse as valid MLIR.
    """
    try:
        with Context() as ctx:
            ctx.load_all_available_dialects()
            module = Module.parse(schedule)
            return _extract_transform_ops(module.operation)
    except ValueError:
        raise
    except Exception as exc:  # MLIR parse / diagnostic errors
        raise ValueError(f"Transformation schedule is not valid MLIR: {exc}") from exc


def verify_schedule(schedule: str, path: Path = _WHITELIST_FILE) -> None:
    """Raise ``ValueError`` if the schedule uses any non-whitelisted transform op.

    Structural scaffolding ops (see ``_STRUCTURAL_OPS``) are always allowed.
    """
    whitelist = load_whitelist(path)
    allowed = whitelist | _STRUCTURAL_OPS
    disallowed = sorted({op for op in extract_transform_ops(schedule) if op not in allowed})
    if disallowed:
        raise ValueError(
            "Transformation schedule uses operations that are not whitelisted: "
            f"{', '.join(disallowed)}. "
            f"Allowed operations: {', '.join(sorted(whitelist))}."
        )
