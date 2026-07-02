from rl_autoschedular_v4_5.actions import ActionSpace
from rl_autoschedular_v4_5.state import OperationState, OperationType, IteratorType, OperationFeatures
import torch
import math
import os
import re

from utils.config import Config

L = Config().max_num_loops
LSD = Config().max_num_load_store_dim
LS = Config().max_num_stores_loads


def _parse_size_to_kb(raw_size: str) -> float:
    token = raw_size.strip().upper()
    match = re.fullmatch(r"(\d+)([KMG])", token)
    if not match:
        return 0.0

    value = float(match.group(1))
    unit = match.group(2)
    if unit == "K":
        return value
    if unit == "M":
        return value * 1024.0
    if unit == "G":
        return value * 1024.0 * 1024.0
    return 0.0


def _read_cache_level_kb(level: int) -> float:
    base = "/sys/devices/system/cpu/cpu0/cache"
    if not os.path.isdir(base):
        return 0.0

    for entry in os.listdir(base):
        level_path = os.path.join(base, entry, "level")
        size_path = os.path.join(base, entry, "size")
        if not os.path.isfile(level_path) or not os.path.isfile(size_path):
            continue
        try:
            with open(level_path, "r") as f_level:
                entry_level = int(f_level.read().strip())
            if entry_level != level:
                continue
            with open(size_path, "r") as f_size:
                return _parse_size_to_kb(f_size.read())
        except (OSError, ValueError):
            continue
    return 0.0


def _read_cpuinfo_text() -> str:
    try:
        with open("/proc/cpuinfo", "r") as f:
            return f.read()
    except OSError:
        return ""


def _detect_physical_cores(cpuinfo_text: str, logical_cores: int) -> int:
    if not cpuinfo_text:
        return logical_cores

    blocks = [b for b in cpuinfo_text.strip().split("\n\n") if b.strip()]
    pairs: set[tuple[str, str]] = set()
    for block in blocks:
        physical_id = None
        core_id = None
        for line in block.splitlines():
            if line.startswith("physical id"):
                physical_id = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":", 1)[1].strip()
        if physical_id is not None and core_id is not None:
            pairs.add((physical_id, core_id))

    if pairs:
        return len(pairs)

    match = re.search(r"cpu cores\s*:\s*(\d+)", cpuinfo_text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return logical_cores


def _detect_simd_width(cpuinfo_text: str) -> int:
    flags_match = re.search(r"^flags\s*:\s*(.+)$", cpuinfo_text, flags=re.MULTILINE)
    if not flags_match:
        return 0

    flags = set(flags_match.group(1).split())
    if "avx512f" in flags:
        return 512
    if "avx2" in flags or "avx" in flags:
        return 256
    if "sse2" in flags or "sse" in flags or "neon" in flags:
        return 128
    return 0


def _detect_clock_mhz(cpuinfo_text: str) -> float:
    match = re.search(r"cpu MHz\s*:\s*([0-9]+(?:\.[0-9]+)?)", cpuinfo_text)
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


def _resolve_feature(config_value: float | int, detected_value: float | int, auto_detect: bool) -> float:
    if config_value > 0:
        return float(config_value)
    if auto_detect:
        return float(detected_value)
    return 0.0


def _build_hardware_vector() -> torch.Tensor:
    cfg = Config()
    cpuinfo_text = _read_cpuinfo_text() if cfg.hardware_auto_detect else ""

    logical_cores = os.cpu_count() or 0
    # Prefer Slurm CPU allocation over physical machine count (V4.6+)
    slurm_cpus = os.getenv('SLURM_CPUS_PER_TASK')
    if slurm_cpus and slurm_cpus.isdigit():
        logical_cores = int(slurm_cpus)
    physical_cores = _detect_physical_cores(cpuinfo_text, logical_cores) if cfg.hardware_auto_detect else 0
    l1_kb = _read_cache_level_kb(1) if cfg.hardware_auto_detect else 0.0
    l2_kb = _read_cache_level_kb(2) if cfg.hardware_auto_detect else 0.0
    l3_kb = _read_cache_level_kb(3) if cfg.hardware_auto_detect else 0.0
    simd_width = _detect_simd_width(cpuinfo_text) if cfg.hardware_auto_detect else 0
    clock_mhz = _detect_clock_mhz(cpuinfo_text) if cfg.hardware_auto_detect else 0.0

    features = torch.tensor([
        _resolve_feature(cfg.hardware_l1_kb, l1_kb, cfg.hardware_auto_detect) / 256.0,
        _resolve_feature(cfg.hardware_l2_kb, l2_kb, cfg.hardware_auto_detect) / 4096.0,
        _resolve_feature(cfg.hardware_l3_kb, l3_kb, cfg.hardware_auto_detect) / 65536.0,
        _resolve_feature(cfg.hardware_physical_cores, physical_cores, cfg.hardware_auto_detect) / 256.0,
        _resolve_feature(cfg.hardware_logical_cores, logical_cores, cfg.hardware_auto_detect) / 512.0,
        _resolve_feature(cfg.hardware_simd_width, simd_width, cfg.hardware_auto_detect) / 1024.0,
        _resolve_feature(cfg.hardware_clock_mhz, clock_mhz, cfg.hardware_auto_detect) / 6000.0,
    ], dtype=torch.float32)

    return features


HARDWARE_VECTOR = _build_hardware_vector()


class ObservationPart:
    @classmethod
    def size(cls) -> int:
        raise NotImplementedError

    @classmethod
    def from_state(cls, state: OperationState) -> torch.Tensor:
        """Create the observation part from the current state."""
        raise NotImplementedError


class OpFeatures(ObservationPart):
    """Class representing operation features in the observation"""

    arith_ops = ['+', '-', '*', '/', 'exp']

    @classmethod
    def size(cls) -> int:
        return len(OperationType) + L + L + LS * LSD * L + LS * LSD * L + len(cls.arith_ops)

    @classmethod
    def from_state(cls, state: OperationState) -> torch.Tensor:
        return cls._from_features(state.original_operation_features)

    @classmethod
    def _from_features(cls, op_features: OperationFeatures) -> torch.Tensor:
        indices_dim = {nested_loop.arg: i for i, nested_loop in enumerate(op_features.nested_loops)}

        # Operation type
        op_type = torch.tensor([op_features.operation_type == ot for ot in OperationType])

        # Nested loop features: (upper bounds, iterator types)
        nested_loops = torch.zeros(L)
        iterator_types = torch.zeros(L)
        for i, nested_loop in enumerate(op_features.nested_loops):
            if i == L:
                break
            ub = nested_loop.upper_bound
            match Config().normalize_bounds:
                case 'max':
                    ub = ub / 4096
                case 'log':
                    ub = math.log2(ub)
            nested_loops[i] = ub
            iterator_types[i] = nested_loop.iterator_type == IteratorType.Parallel

        # # Vectorizable
        # vectorizable = torch.tensor([op_features.vectorizable])

        # load access matrices:
        load_access_matrices = torch.zeros((LS, LSD, L))

        for load_i, load in enumerate(op_features.load_data):
            if load_i == LS:
                break
            dimensions_terms = [cls.__formula_str_to_list(term) for term in load]
            for m, dimension_term in enumerate(dimensions_terms):
                if m == LSD:
                    break
                for index, factor in dimension_term:
                    if index not in indices_dim:
                        continue
                    n = indices_dim[index]
                    if n >= L:
                        continue
                    load_access_matrices[load_i, m, n] = factor

        # store access matrices:
        store_access_matrices = torch.zeros((LS, LSD, L))

        for store_i, store in enumerate(op_features.store_data):
            if store_i == LS:
                break
            dimensions_terms = [cls.__formula_str_to_list(term) for term in store]
            for m, dimension_term in enumerate(dimensions_terms):
                if m == LSD:
                    break
                for index, factor in dimension_term:
                    if index not in indices_dim:
                        continue
                    n = indices_dim[index]
                    if n >= L:
                        continue
                    store_access_matrices[store_i, m, n] = factor

        # Operations count:
        operations_count = torch.tensor([op_features.op_count[s] for s in cls.arith_ops])

        feature_vector = torch.cat((
            op_type,
            nested_loops,
            iterator_types,
            # vectorizable,
            load_access_matrices.reshape(-1),
            store_access_matrices.reshape(-1),
            operations_count
        ))

        return feature_vector

    @staticmethod
    def __formula_str_to_list(formula: str) -> list[tuple[str, int]]:
        """Turns assignement formula to a list of (index, factor)
        Example:
            formula = "%x1 - %x2 + %x3 * 5 - %x5 * 3"
            return [('%x1', 1), ('%x2', -1), ('%x3', 5), ('%x5', -3)]

        Args:
            formula (str): the formula as a string input

        Returns:
            list: list of (index, factor) pairs
        """
        formula = formula + ' +'
        terms = formula.split(' ')

        running_factor = 1
        running_term = None

        save = []

        for term in terms:

            if term.startswith('%'):
                running_term = term
            elif term == '+':
                save.append((running_term, running_factor))
                running_factor = 1
            elif term == '-':
                save.append((running_term, running_factor))
                running_factor = -1
            elif term.isnumeric():
                running_factor *= int(term)

        if save[0][0] is None:
            save = save[1:]

        return save


class ProducerOpFeatures(OpFeatures):
    @classmethod
    def from_state(cls, state: OperationState) -> torch.Tensor:
        if state.producer_features:
            return cls._from_features(state.producer_features)

        return torch.zeros(cls.size())


class ActionHistory(ObservationPart):
    """Class representing action history in the observation"""

    @classmethod
    def size(cls) -> int:
        return ActionSpace.cumulative_history_sizes()[-1]

    @classmethod
    def from_state(cls, state: OperationState) -> torch.Tensor:
        return ActionSpace.action_history(state.current_history)


class ActionMask(ObservationPart):
    """Class representing action mask in the observation"""

    @classmethod
    def size(cls) -> int:
        return ActionSpace.cumulative_mask_sizes()[-1]

    @classmethod
    def from_state(cls, state: OperationState) -> torch.Tensor:
        return ActionSpace.action_mask(state)


class NumLoops(ObservationPart):
    """Class representing number of loops in the observation"""

    @classmethod
    def size(cls) -> int:
        return 1

    @classmethod
    def from_state(cls, state: OperationState) -> torch.Tensor:
        return torch.tensor([len(state.operation_features.nested_loops)])


class HardwareFeatures(ObservationPart):
    """Hardware descriptors injected into every observation."""

    @classmethod
    def size(cls) -> int:
        return HARDWARE_VECTOR.numel()

    @classmethod
    def from_state(cls, state: OperationState) -> torch.Tensor:
        return HARDWARE_VECTOR


class Observation:
    """Class to manage creation and use of observations"""

    parts: list[type[ObservationPart]] = [
        OpFeatures,
        ProducerOpFeatures,
        ActionHistory,
        NumLoops,
        HardwareFeatures,
        ActionMask
    ]

    @classmethod
    def cumulative_sizes(cls) -> list[int]:
        """Get cumulative sizes of all observation parts."""
        sizes = [0]
        for part in cls.parts:
            sizes.append(sizes[-1] + part.size())
        return sizes

    @classmethod
    def part_number(cls, part: type[ObservationPart]) -> int:
        """Get the index of a part in the observation."""
        return cls.parts.index(part)

    @classmethod
    def get_part(cls, obs: torch.Tensor, part: type[ObservationPart], squeeze: bool = True) -> torch.Tensor:
        """Get a specific part of the observation."""
        part_idx = cls.part_number(part)
        cum_sizes = cls.cumulative_sizes()
        start = cum_sizes[part_idx]
        if part.size() == 1 and squeeze:
            return obs[:, start]
        end = cum_sizes[part_idx + 1]
        return obs[:, start:end]

    @classmethod
    def get_parts(cls, obs: torch.Tensor, *parts: type[ObservationPart]) -> torch.Tensor:
        """Get multiple parts of the observation in a single tensor."""
        return torch.cat([cls.get_part(obs, part, False) for part in parts], dim=1)

    @classmethod
    def from_state(cls, state: OperationState) -> torch.Tensor:
        """Create the full observation from the current state."""
        obs_parts = [part.from_state(state) for part in cls.parts]
        return torch.cat(obs_parts).unsqueeze(0)

    @classmethod
    def from_states(cls, states: list[OperationState]) -> torch.Tensor:
        """Create the full observation for all the states."""
        return torch.cat([cls.from_state(s) for s in states])
