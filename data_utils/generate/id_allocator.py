#!/usr/bin/env python3
"""
id_allocator.py
---------------
Utilities for deterministic numeric ID allocation across legacy and new datasets.

Use cases:
- Continue synthetic single IDs after legacy `single_bench_*.mlir`
- Continue synthetic bench IDs after legacy `bench_*.mlir`
- Keep one consistent ID space even after multiple generation runs
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Iterable


SINGLE_PATTERN = re.compile(r"^single_bench_(\d+)\.mlir$")
BENCH_PATTERN = re.compile(r"^bench_(\d+)\.mlir$")


def _scan_ids(directory: str, pattern: re.Pattern[str]) -> list[int]:
    """Return sorted IDs extracted from file names in *directory*."""
    if not directory or not os.path.isdir(directory):
        return []

    ids: list[int] = []
    for name in os.listdir(directory):
        match = pattern.match(name)
        if match:
            ids.append(int(match.group(1)))
    ids.sort()
    return ids


def _max_or_none(values: Iterable[int]) -> int | None:
    values = list(values)
    return max(values) if values else None


@dataclass(frozen=True)
class IdSpaceState:
    single_legacy_max: int | None
    single_current_max: int | None
    single_next: int

    bench_legacy_max: int | None
    bench_current_max: int | None
    bench_next: int


def build_id_space_state(
    legacy_single_dir: str,
    legacy_bench_dir: str,
    synthetic_single_dir: str,
    synthetic_bench_dir: str,
) -> IdSpaceState:
    """
    Compute next IDs by considering both legacy sources and already-generated files.

    This guarantees that new synthetic files continue after the maximum ID seen in
    either location.
    """
    legacy_single_ids = _scan_ids(legacy_single_dir, SINGLE_PATTERN)
    legacy_bench_ids = _scan_ids(legacy_bench_dir, BENCH_PATTERN)

    current_single_ids = _scan_ids(synthetic_single_dir, SINGLE_PATTERN)
    current_bench_ids = _scan_ids(synthetic_bench_dir, BENCH_PATTERN)

    single_legacy_max = _max_or_none(legacy_single_ids)
    single_current_max = _max_or_none(current_single_ids)
    bench_legacy_max = _max_or_none(legacy_bench_ids)
    bench_current_max = _max_or_none(current_bench_ids)

    single_next = max(v for v in [single_legacy_max, single_current_max, -1] if v is not None) + 1
    bench_next = max(v for v in [bench_legacy_max, bench_current_max, -1] if v is not None) + 1

    return IdSpaceState(
        single_legacy_max=single_legacy_max,
        single_current_max=single_current_max,
        single_next=single_next,
        bench_legacy_max=bench_legacy_max,
        bench_current_max=bench_current_max,
        bench_next=bench_next,
    )


def reserve_ids(start_id: int, count: int) -> list[int]:
    """Return contiguous IDs [start_id, start_id + count)."""
    if count < 0:
        raise ValueError("count must be >= 0")
    return [start_id + i for i in range(count)]


def single_name(file_id: int) -> str:
    return f"single_bench_{file_id}.mlir"


def bench_name(file_id: int) -> str:
    return f"bench_{file_id}.mlir"
