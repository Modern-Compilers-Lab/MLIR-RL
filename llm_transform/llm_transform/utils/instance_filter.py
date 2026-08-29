"""Shared parsing for the `INSTANCE_FILTER` environment variable.

Tokens are space- or comma-separated and may be benchmark names (e.g. `matmul`)
or full `{name}_{instance}` IDs (e.g. `matmul_2`).
"""
import os
import re


def parse_instance_filter(raw: str | None = None) -> set[str] | None:
    """Return a set of allowed tokens, or None when no filter is configured.

    If `raw` is None, the value is read from the `INSTANCE_FILTER` environment
    variable.
    """
    if raw is None:
        raw = os.environ.get("INSTANCE_FILTER", "")
    raw = raw.strip()
    if not raw:
        return None
    return {tok for tok in re.split(r"[,\s]+", raw) if tok}
