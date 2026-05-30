import math
import re
import subprocess
from enum import Enum

import numpy as np

from llm_action.src.keys import AST_DUMPER_BIN_PATH
from llm_action.src.config import ARITH_OPS, L, LS, LSD, NUM_OP_TYPES, OP_FEATURES_SIZE, AST_DUMPER_TIMEOUT

class OperationType(Enum):
    Generic = "generic"
    Matmul = "matmul"
    Conv = "conv"
    Pooling = "pooling"
    Add = "add"
    Relu = "relu"

OP_TYPE_LIST = list(OperationType)
assert len(OP_TYPE_LIST) == NUM_OP_TYPES, (
    f"OperationType enum has {len(OP_TYPE_LIST)} members but config.NUM_OP_TYPES={NUM_OP_TYPES}"
)

_RELU_BODY_SIGNATURE = ("arith.cmpf ugt", "arith.select")

# Match `linalg.<name>` whose attribute brace carries `tag = "operation_0"`.
# Used to identify the family of the tagged op directly from the MLIR source,
# uniformly across all op types (matmul, conv, pool, add, generic/relu).
_TAGGED_OP_RE = re.compile(
    r'linalg\.([\w_]+)[^{]*\{[^}]*tag\s*=\s*"operation_0"',
    re.DOTALL,
)

def observation_size(total_actions: int, max_steps: int, history_mode: str = "success-encoding") -> int:
    if history_mode == "success-encoding":
        per_step = total_actions + 1  # one-hot action + success flag
    else:  # "include-all" or "ignore-failed"
        per_step = total_actions
    return OP_FEATURES_SIZE + max_steps * per_step + 1

def extract_observation(code: str, action_indices: list[tuple[int, bool]], step: int,
                        total_actions: int, max_steps: int,
                        history_mode: str = "success-encoding",
                        loop_bound_encoding: str = "log", bound_norm: float = 1.0) -> np.ndarray:
    obs_size = observation_size(total_actions, max_steps, history_mode)
    obs = np.zeros(obs_size, dtype=np.float32)

    try:
        obs[:OP_FEATURES_SIZE] = _extract_op_features(
            code, loop_bound_encoding=loop_bound_encoding, bound_norm=bound_norm)
    except Exception:
        pass

    offset = OP_FEATURES_SIZE
    if history_mode == "success-encoding":
        stride = total_actions + 1
        for i, (act_idx, success) in enumerate(action_indices[:max_steps]):
            obs[offset + i * stride + act_idx] = 1.0
            obs[offset + i * stride + total_actions] = 1.0 if success else 0.0
    else:  # "include-all" or "ignore-failed"
        slot = 0
        for act_idx, success in action_indices[:max_steps]:
            if history_mode == "ignore-failed" and not success:
                continue
            obs[offset + slot * total_actions + act_idx] = 1.0
            slot += 1

    obs[-1] = step / max_steps
    return obs

def _extract_op_features(code: str, loop_bound_encoding: str = "log", bound_norm: float = 1.0) -> np.ndarray:
    result = subprocess.run(
        f"{AST_DUMPER_BIN_PATH} -", shell=True,
        input=code.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=AST_DUMPER_TIMEOUT,
    )
    if result.returncode != 0:
        return np.zeros(OP_FEATURES_SIZE, dtype=np.float32)
    raw = result.stdout.decode()

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

    return _encode_block(block, code=code, loop_bound_encoding=loop_bound_encoding, bound_norm=bound_norm)

def _parse_loop_info(code: str) -> tuple[int, list[int]]:
    """Run AST dumper once and return (n_loops, [upper_bound_per_loop]).

    Falls back to (L, []) on any error so callers degrade gracefully.
    """
    try:
        result = subprocess.run(
            f"{AST_DUMPER_BIN_PATH} -", shell=True,
            input=code.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=AST_DUMPER_TIMEOUT,
        )
        if result.returncode != 0:
            return L, []
        raw = result.stdout.decode()
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
        loops = []
        for line in loops_str.strip().split("\n"):
            if not line.strip():
                continue
            p = line.strip().split(" ")
            loops.append(int(p[2]))  # upper bound is index 2: (var, lower, upper, step, flag)
        n = max(1, len(loops))
        return n, loops
    except Exception:
        return L, []


def count_loops(code: str) -> int:
    """Count the number of loop dimensions in the tagged operation. Fast fallback: L."""
    n, _ = _parse_loop_info(code)
    return n


def extract_loop_bounds(code: str) -> list[int]:
    """Upper bounds for each loop in the tagged operation. Returns [] on error (safe fallback)."""
    _, bounds = _parse_loop_info(code)
    return bounds

def _encode_block(block: str, code: str = "", loop_bound_encoding: str = "log", bound_norm: float = 1.0) -> np.ndarray:
    vec = np.zeros(OP_FEATURES_SIZE, dtype=np.float32)
    rest, _ = block.split("#START_TAG")
    op_name, rest = rest.split("#START_VECTORIZABLE")

    offset = 0
    op_type = _get_op_type(op_name.strip(), code=code)
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
        if loop_bound_encoding == "max":
            vec[offset + i] = nl[2] / bound_norm if bound_norm > 0 else 0.0
        else:  # "log"
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

def _detect_op_type_from_code(code: str) -> OperationType | None:
    """Identify the tagged op's family by parsing the `linalg.<name>` carrying the tag.

    Returns None when the source has no tagged linalg op (e.g. when only the
    AST-dumper output is available). The relu refinement of `generic` is the
    op-type analogue of the legacy `conv → Generic` refinement for `op0/op1/i2c`
    post-transform variants.
    """
    if not code:
        return None
    m = _TAGGED_OP_RE.search(code)
    if not m:
        return None
    linalg_name = m.group(1)

    if linalg_name == "matmul":
        return OperationType.Matmul
    if linalg_name.startswith("conv"):
        return OperationType.Conv
    if linalg_name.startswith("pooling"):
        return OperationType.Pooling
    if linalg_name == "add":
        return OperationType.Add
    if linalg_name == "generic":
        if all(s in code for s in _RELU_BODY_SIGNATURE):
            return OperationType.Relu
        return OperationType.Generic
    return OperationType.Generic


def _get_op_type(name: str, code: str = "") -> OperationType:
    """Identify the tagged op's family.

    Prefers source-based detection (uniform across all op families). Falls back
    to AST-dumper-name detection when the code is unavailable, preserving
    legacy behavior including the `conv → Generic` post-transform refinement.
    """
    detected = _detect_op_type_from_code(code)
    if detected is not None:
        return detected
    for ot in OperationType:
        if ot is OperationType.Relu:
            continue  # body-required; cannot identify relu from a name alone
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
