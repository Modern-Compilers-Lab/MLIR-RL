#!/usr/bin/env python3
"""
batch_policy.py
---------------
Batch-size selection policies for extracted block benchmarks.

Policy contract:
- Default is heuristic-only (data-driven from block op-count distribution).
- No fallback is applied unless explicitly requested.
- Optional fallback: mean (deterministic mean-based batch set).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class BatchPolicyError(RuntimeError):
    """Raised when batch policy selection cannot proceed."""


@dataclass(frozen=True)
class HeuristicModel:
    q25: float
    q50: float
    q75: float
    batch_candidates: list[int]


@dataclass(frozen=True)
class BatchSelectionResult:
    selected_batches: list[int]
    method: str
    details: dict


def sanitize_batch_candidates(values: list[int]) -> list[int]:
    """Return unique sorted positive batch candidates."""
    clean = sorted({int(v) for v in values if int(v) > 0})
    if not clean:
        raise BatchPolicyError("Batch candidate list is empty after sanitization")
    return clean


def block_complexity_from_op_counts(op_counts: list[dict[str, int]]) -> int:
    """Compute deterministic complexity score from block op-count dictionaries."""
    total = 0
    for op_count in op_counts:
        for _, value in op_count.items():
            total += int(value)
    return total


def _percentile(sorted_vals: list[int], q: float) -> float:
    """Linear-interpolated percentile for 0<=q<=1."""
    if not sorted_vals:
        raise BatchPolicyError("Cannot compute percentile on empty list")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])

    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return (1.0 - frac) * sorted_vals[lo] + frac * sorted_vals[hi]


def fit_heuristic_model(complexities: list[int], batch_candidates: list[int]) -> HeuristicModel:
    """Fit thresholds from complexity distribution."""
    if not complexities:
        raise BatchPolicyError("No block complexities were provided")

    positives = [c for c in complexities if c > 0]
    if not positives:
        raise BatchPolicyError(
            "Heuristic batch selection failed: all block complexities are zero"
        )

    candidates = sanitize_batch_candidates(batch_candidates)
    sorted_vals = sorted(positives)

    return HeuristicModel(
        q25=_percentile(sorted_vals, 0.25),
        q50=_percentile(sorted_vals, 0.50),
        q75=_percentile(sorted_vals, 0.75),
        batch_candidates=candidates,
    )


def _heuristic_select_one(complexity: int, model: HeuristicModel) -> int:
    """Select one batch value for a single complexity score."""
    cands = model.batch_candidates
    n = len(cands)

    # Heavier blocks -> smaller batch; lighter blocks -> larger batch.
    if complexity >= model.q75:
        idx = 0
    elif complexity >= model.q50:
        idx = min(1, n - 1)
    elif complexity >= model.q25:
        idx = min(n // 2, n - 1)
    else:
        idx = n - 1

    return cands[idx]


def _mean_fallback_set(
    complexities: list[int],
    batch_candidates: list[int],
) -> tuple[list[int], float]:
    """Build deterministic mean-based batch set."""
    if not complexities:
        raise BatchPolicyError("Mean fallback cannot run on empty complexities")

    candidates = sanitize_batch_candidates(batch_candidates)
    positives = [c for c in complexities if c > 0]
    if not positives:
        positives = [1]

    mean_val = float(sum(positives)) / float(len(positives))
    max_val = float(max(positives))
    ratio = mean_val / max_val if max_val > 0 else 1.0

    # Mean complexity selects center; heavier mean -> smaller center batch.
    if ratio >= 0.66:
        center_idx = 0
    elif ratio >= 0.33:
        center_idx = len(candidates) // 2
    else:
        center_idx = len(candidates) - 1

    idx_set = sorted({
        max(0, center_idx - 1),
        center_idx,
        min(len(candidates) - 1, center_idx + 1),
    })
    batch_set = [candidates[i] for i in idx_set]
    return batch_set, mean_val


def select_batches(
    complexities: list[int],
    batch_candidates: list[int],
    fallback: Optional[str] = None,
) -> BatchSelectionResult:
    """
    Select one batch per block complexity.

    Args:
        complexities: complexity score per block.
        batch_candidates: allowed batch values.
        fallback: None or "mean".

    Returns:
        BatchSelectionResult with per-block selected batch values.
    """
    try:
        model = fit_heuristic_model(complexities, batch_candidates)
        selected = [_heuristic_select_one(c, model) for c in complexities]
        return BatchSelectionResult(
            selected_batches=selected,
            method="heuristic",
            details={
                "q25": model.q25,
                "q50": model.q50,
                "q75": model.q75,
                "batch_candidates": model.batch_candidates,
            },
        )
    except BatchPolicyError as exc:
        if fallback != "mean":
            raise

        # Explicit fallback path only.
        batch_set, mean_val = _mean_fallback_set(complexities, batch_candidates)

        # Deterministic assignment around mean: heavy -> smallest, light -> largest.
        lo = batch_set[0]
        hi = batch_set[-1]
        mid = batch_set[len(batch_set) // 2]

        positives = [c for c in complexities if c > 0]
        ref_mean = (sum(positives) / len(positives)) if positives else 0.0

        selected = []
        for c in complexities:
            if c > ref_mean:
                selected.append(lo)
            elif c < ref_mean:
                selected.append(hi)
            else:
                selected.append(mid)

        return BatchSelectionResult(
            selected_batches=selected,
            method="fallback:mean",
            details={
                "reason": str(exc),
                "mean_complexity": ref_mean,
                "batch_set": batch_set,
                "batch_candidates": sanitize_batch_candidates(batch_candidates),
            },
        )
