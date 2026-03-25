import math
import re
import subprocess
from enum import Enum

import numpy as np

from llm_action.src.keys import AST_DUMPER_BIN_PATH
from llm_action.src.config import ARITH_OPS, L, LS, LSD, NUM_OP_TYPES, OP_FEATURES_SIZE

class OperationType(Enum):
    Generic = "generic"
    Matmul = "matmul"
    # Conv = "conv"
    # Pooling = "pooling"
    # Add = "add"

OP_TYPE_LIST = list(OperationType)
assert len(OP_TYPE_LIST) == NUM_OP_TYPES, (
    f"OperationType enum has {len(OP_TYPE_LIST)} members but config.NUM_OP_TYPES={NUM_OP_TYPES}"
)

def observation_size(total_actions: int, max_steps: int) -> int:
    return OP_FEATURES_SIZE + max_steps * total_actions + 1

def extract_observation(code: str, action_indices: list[int], step: int, total_actions: int, max_steps: int) -> np.ndarray:
    obs_size = observation_size(total_actions, max_steps)
    obs = np.zeros(obs_size, dtype=np.float32)

    try:
        obs[:OP_FEATURES_SIZE] = _extract_op_features(code)
    except Exception:
        pass

    offset = OP_FEATURES_SIZE
    for i, act_idx in enumerate(action_indices[:max_steps]):
        obs[offset + i * total_actions + act_idx] = 1.0

    obs[-1] = step / max_steps
    return obs

def _extract_op_features(code: str) -> np.ndarray:
    raw = subprocess.run(
        f"{AST_DUMPER_BIN_PATH} -", shell=True,
        input=code.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
    ).stdout.decode()

    info, _ = raw.split("########################################")
    operations_lines, _ = info.split("#BEGIN_GRAPH")
    blocks = [b.strip() for b in operations_lines.split("#START_OPERATION") if b.strip()]
    if not blocks:
        return np.zeros(OP_FEATURES_SIZE, dtype=np.float32)

    block = blocks[0]
    for b in blocks:
        if "operation_0" in b:
            block = b
            break

    return _encode_block(block)

def count_loops(code: str) -> int:
    """Count the number of loop dimensions in the tagged operation. Fast fallback: 3."""
    try:
        raw = subprocess.run(
            f"{AST_DUMPER_BIN_PATH} -", shell=True,
            input=code.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
        ).stdout.decode()
        info, _ = raw.split("########################################")
        operations_lines, _ = info.split("#BEGIN_GRAPH")
        blocks = [b.strip() for b in operations_lines.split("#START_OPERATION") if b.strip()]
        block = blocks[0]
        for b in blocks:
            if "operation_0" in b:
                block = b
                break
        rest, _ = block.split("#START_TAG")
        _, rest = rest.split("#START_VECTORIZABLE")
        _, rest = rest.split("#START_NESTED_LOOPS")
        loops_str, _ = rest.split("#START_LOAD_DATA")
        n = sum(1 for line in loops_str.strip().split("\n") if line.strip())
        return max(1, n)
    except Exception:
        return L

def _encode_block(block: str) -> np.ndarray:
    vec = np.zeros(OP_FEATURES_SIZE, dtype=np.float32)
    rest, _ = block.split("#START_TAG")
    op_name, rest = rest.split("#START_VECTORIZABLE")

    offset = 0
    op_type = _get_op_type(op_name.strip())
    for i, ot in enumerate(OP_TYPE_LIST):
        if op_type == ot:
            vec[offset + i] = 1.0
    offset += len(OP_TYPE_LIST)

    _, rest = rest.split("#START_NESTED_LOOPS")
    loops_str, rest = rest.split("#START_LOAD_DATA")
    loops = []
    for line in loops_str.strip().split("\n"):
        if not line.strip():
            continue
        p = line.strip().split(" ")
        loops.append((f"%{p[0]}", int(p[1]), int(p[2]), int(p[3]), p[4]))

    idx_map = {nl[0]: i for i, nl in enumerate(loops)}

    for i, nl in enumerate(loops[:L]):
        vec[offset + i] = math.log2(max(nl[2], 1))
    offset += L

    for i, nl in enumerate(loops[:L]):
        vec[offset + i] = 1.0 if nl[4] == "parallel" else 0.0
    offset += L

    loads_str, rest = rest.split("#START_STORE_DATA")
    offset = _encode_access(vec, offset, loads_str, idx_map)

    stores_str, ops_str = rest.split("#START_OP_COUNT")
    offset = _encode_access(vec, offset, stores_str, idx_map)

    op_count = {}
    for line in ops_str.strip().split("\n"):
        if line.strip():
            p = line.strip().split(" ")
            op_count[p[0]] = int(p[1])
    for i, sym in enumerate(ARITH_OPS):
        vec[offset + i] = op_count.get(sym, 0)

    return vec

def _encode_access(vec, offset, data_str, idx_map):
    data_str = re.sub(r"d\d+", lambda m: f"%{m.group()}", data_str)
    rows = [line.split(", ") for line in data_str.strip().split("\n") if line.strip()]
    for ri, row in enumerate(rows[:LS]):
        terms = [_parse_formula(t) for t in row]
        for m, dim_terms in enumerate(terms[:LSD]):
            for idx, factor in dim_terms:
                if idx in idx_map and idx_map[idx] < L:
                    vec[offset + ri * LSD * L + m * L + idx_map[idx]] = factor
    return offset + LS * LSD * L

def _get_op_type(name: str) -> OperationType:
    for ot in OperationType:
        if ot.value and ot.value in name:
            if ot.value == "conv" and any(s in name for s in ["op0", "op1", "i2c"]):
                return OperationType.Generic
            return ot
    return OperationType.Generic

def _parse_formula(formula: str) -> list[tuple[str, int]]:
    terms = (formula + " +").split(" ")
    factor, var, result = 1, None, []
    for t in terms:
        if t.startswith("%"):
            var = t
        elif t == "+":
            result.append((var, factor))
            factor = 1
        elif t == "-":
            result.append((var, factor))
            factor = -1
        elif t.isnumeric():
            factor *= int(t)
    if result and result[0][0] is None:
        result = result[1:]
    return result
