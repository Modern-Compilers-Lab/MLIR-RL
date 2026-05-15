import argparse

from llm_action.src.config import MAX_PARAM_SLOTS, L
from llm_action.src.env.action_registry import load_action_registry
from llm_action.src.env.action_space import build_action_space, build_action_masks

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--action-version", default="v10")
    args = p.parse_args()

    reg = load_action_registry(args.action_version)
    space, slot_map = build_action_space(reg, max_n_loops=L)

    print(f"=== Action Space ({args.action_version}) ===")
    print(f"  Config: MAX_PARAM_SLOTS={MAX_PARAM_SLOTS}, L(max_n_loops)={L}")
    print(f"  Actions: {reg.num_actions} + done = {reg.total_actions}")
    print(f"  MultiDiscrete dims: {len(space.nvec)}  nvec={list(space.nvec)}")
    print()

    # Per-action slot breakdown
    print("--- Per-action slot breakdown ---")
    print(f"  {'Action':<20} {'params_size':>11} {'slots':>5}  classes_per_slot({L})")
    print(f"  {'─'*20} {'─'*11} {'─'*5}  {'─'*30}")
    total_slots = 0
    for idx, cls in enumerate(reg.action_classes):
        cps = cls.classes_per_slot(L)
        start, end = slot_map[idx]
        total_slots += len(cps)
        print(f"  {cls.__name__:<20} {cls.params_size():>11} {len(cps):>5}  {cps}  (dims {start}..{end-1})")
    print(f"  {'done':<20} {'—':>11} {'0':>5}  []  (no params)")
    print(f"\n  Total param dims: {total_slots}  |  Full space: 1 (selector) + {total_slots} (params) = {1 + total_slots}")
    assert len(space.nvec) == 1 + total_slots, "FAIL: space dims mismatch"
    print("  # Space dimensions match\n")

    # Slot map integrity
    print("--- Slot map integrity ---")
    ranges = []
    for idx in range(reg.num_actions):
        assert idx in slot_map, f"FAIL: action {idx} missing from slot_map"
        s, e = slot_map[idx]
        assert e >= s
        ranges.append((s, e))
    s, e = slot_map[reg.done_idx]
    assert s == e, f"FAIL: done should have empty range, got ({s},{e})"
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            s1, e1 = ranges[i]
            s2, e2 = ranges[j]
            assert e1 <= s2 or e2 <= s1, f"FAIL: overlap between action {i} and {j}"
    print("  # All actions covered, no overlaps\n")

    # No-padding check
    print("--- No-padding check ---")
    for cls in reg.action_classes:
        cps = cls.classes_per_slot(L)
        assert len(cps) <= MAX_PARAM_SLOTS, f"FAIL: {cls.__name__} has {len(cps)} slots > MAX_PARAM_SLOTS"
        has_trailing_ones = len(cps) > 1 and cps[-1] == 1
        status = "⚠ trailing [1]" if has_trailing_ones else "# clean"
        print(f"  {cls.__name__:<20} {status}")
    print()

    # decode_params for all vocab values
    print("--- decode_params (all vocab values) ---")
    for cls in reg.action_classes:
        cps = cls.classes_per_slot(L)
        if not cps:
            print(f"  {cls.__name__:<20} (no params)")
            continue
        n_combos = 0
        for slot_i, n_classes in enumerate(cps):
            for val in range(n_classes):
                raw = [0] * len(cps)
                raw[slot_i] = val
                params = cls.decode_params(raw, n_loops=L)
                assert isinstance(params, dict) and len(params) > 0
                n_combos += 1
        # Show one example
        example_raw = [c // 2 for c in cps]
        example_params = cls.decode_params(example_raw, n_loops=L)
        print(f"  {cls.__name__:<20} {n_combos} combos OK  |  example: raw={example_raw} → {example_params}")
    print()

    # Action masks
    print("--- Action masks ---")
    masks_full = build_action_masks(
        reg, slot_map, n_loops=L, max_n_loops=L,
        used_action_counts={},
    )
    print(f"  n_loops={L}: mask length={len(masks_full)}, True={masks_full.sum()}/{len(masks_full)}")
    assert len(masks_full) == sum(space.nvec)

    if L > 1:
        masks_1 = build_action_masks(
            reg, slot_map, n_loops=1, max_n_loops=L,
            used_action_counts={},
        )
        print(f"  n_loops=1: mask length={len(masks_1)}, True={masks_1.sum()}/{len(masks_1)}")
        assert masks_1.sum() <= masks_full.sum()
        print(f"  Shrinkage: {masks_full.sum()} → {masks_1.sum()} ({masks_full.sum() - masks_1.sum()} fewer valid choices)")

    masks_used = build_action_masks(
        reg, slot_map, n_loops=L, max_n_loops=L,
        used_action_counts={idx: 1 for idx in range(reg.num_actions)},
    )
    assert masks_used[reg.done_idx], "FAIL: done should always be unmasked"
    print(f"  All actions used once: done still available (per-action unique_execution honored) #")

    # New cap-coverage assertion: at MAX_ACTION_EXECUTIONS, even unique_execution=False
    # action indices must be masked.
    from llm_action.src.config import MAX_ACTION_EXECUTIONS
    masks_cap = build_action_masks(
        reg, slot_map, n_loops=L, max_n_loops=L,
        used_action_counts={idx: MAX_ACTION_EXECUTIONS for idx in range(reg.num_actions)},
    )
    for idx in range(reg.num_actions):
        if idx == reg.done_idx:
            continue
        assert masks_cap[idx] == False, (
            f"FAIL: action {idx} should be masked once count == MAX_ACTION_EXECUTIONS "
            f"(unique_execution={reg.action_classes[idx].unique_execution})"
        )
    assert masks_cap[reg.done_idx], "FAIL: done should always be unmasked"
    print(f"  All actions at cap ({MAX_ACTION_EXECUTIONS} uses): all masked except done #")
    print()

    print("=== ALL CHECKS PASSED ===")

if __name__ == "__main__":
    main()
