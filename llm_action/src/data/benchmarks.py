from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

from llm_action.src.config import DATA_DIR

BENCHMARKS_ROOT: Path = DATA_DIR
TEMPLATES_DIR: Path = DATA_DIR / "templates"
STANDARD_SET_NAME: str = "standard"

Split = Literal["train", "eval", "all"]

# Family detection. Each entry is (family_name, regex matched against the file stem).
_FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("img2col_conv_2d_nchw_fchw", re.compile(r"^img2col_conv_2d_nchw_fchw_.+$")),
    ("conv_2d_nchw_fchw", re.compile(r"^conv_2d_nchw_fchw_.+$")),
    ("matmul",            re.compile(r"^matmul_.+$")),
    ("pooling_nchw",      re.compile(r"^pooling_nchw_.+$")),
    ("add",               re.compile(r"^add.+$")),
    ("relu",              re.compile(r"^relu_.+$")),
    ("generic",           re.compile(r"^generic_.+$")),
)

SUPPORTED_FAMILIES: tuple[str, ...] = tuple(name for name, _ in _FAMILY_PATTERNS)

@dataclass(frozen=True)
class BenchmarkInstance:
    name: str          # filename stem, e.g. "matmul_128_256_128"
    family: str        # one of SUPPORTED_FAMILIES, or "unknown"
    code: str          # full MLIR source
    path: Path         # absolute path to the .mlir file
    split: str         # "train" | "eval" | "flat"


def detect_family(name: str) -> str | None:
    """Return the op family for a benchmark stem, or None if no pattern matches."""
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.match(name):
            return family
    return None


def template_path(family: str) -> Path:
    return TEMPLATES_DIR / f"{family}_template.mlir"


def load_template(family: str) -> str:
    path = template_path(family)
    if not path.exists():
        raise FileNotFoundError(f"No template for family '{family}' at {path}")
    return path.read_text()


def _resolve_set_dir(name: str) -> Path:
    bdir = BENCHMARKS_ROOT / name
    if not bdir.exists():
        raise FileNotFoundError(f"Benchmark set not found: {bdir}")
    return bdir


def _is_split_layout(set_dir: Path) -> bool:
    return (set_dir / "train").is_dir() or (set_dir / "eval").is_dir()


def _iter_split_dirs(set_dir: Path, split: Split) -> Iterable[tuple[str, Path]]:
    """Yield (split_label, dir) pairs to scan for *.mlir files."""
    if _is_split_layout(set_dir):
        candidates = ("train", "eval") if split == "all" else (split,)
        for s in candidates:
            sub = set_dir / s
            if sub.is_dir():
                yield s, sub
    else:
        yield "flat", set_dir


def load_benchmark_set(
    name: str = STANDARD_SET_NAME,
    split: Split = "train",
) -> list[BenchmarkInstance]:
    """Load every *.mlir in the given benchmark set.

    For split sets (with `train/` and/or `eval/` subdirs), `split` selects which.
    For flat sets, `split` is ignored.
    """
    set_dir = _resolve_set_dir(name)
    instances: list[BenchmarkInstance] = []
    for split_label, sdir in _iter_split_dirs(set_dir, split):
        for f in sorted(sdir.glob("*.mlir")):
            stem = f.stem
            family = detect_family(stem) or "unknown"
            instances.append(BenchmarkInstance(
                name=stem,
                family=family,
                code=f.read_text(),
                path=f,
                split=split_label,
            ))
    if not instances:
        raise FileNotFoundError(f"No .mlir files found in set '{name}' (split={split})")
    return instances


def group_by_family(
    instances: list[BenchmarkInstance],
) -> dict[str, list[BenchmarkInstance]]:
    """Group instances by op family, preserving sort order within each group."""
    groups: dict[str, list[BenchmarkInstance]] = {}
    for inst in instances:
        groups.setdefault(inst.family, []).append(inst)
    return groups


def load_baselines(name: str, split: Split = "train") -> dict[str, dict[str, float]]:
    """Read `baselines.json` for a benchmark set, normalized to {bname: {mlir, torch}}.

    Supports both schemas:
      - flat:   {bname: {mlir, torch}}
      - nested: {train: {bname: {...}}, eval: {bname: {...}}}
    For nested + split="all", train and eval entries are merged (eval overrides on collision).
    """
    set_dir = _resolve_set_dir(name)
    path = set_dir / "baselines.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not raw:
        return {}
    # Detect nested schema: top-level keys are split labels.
    if set(raw.keys()) <= {"train", "eval"}:
        if split == "all":
            merged: dict[str, dict[str, float]] = {}
            merged.update(raw.get("train", {}))
            merged.update(raw.get("eval", {}))
            return merged
        return raw.get(split, {})
    return raw


def save_baselines(
    name: str,
    baselines: dict[str, dict[str, float]],
    split: Split = "train",
) -> None:
    """Write `baselines.json` honoring the existing schema (flat or nested).

    For split layouts, writes back into the nested schema under the requested split.
    """
    set_dir = _resolve_set_dir(name)
    path = set_dir / "baselines.json"
    nested = _is_split_layout(set_dir)
    if nested:
        existing: dict[str, dict[str, dict[str, float]]] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text())
                if set(loaded.keys()) <= {"train", "eval"}:
                    existing = loaded
            except (json.JSONDecodeError, OSError):
                pass
        if split == "all":
            existing.setdefault("train", {}).update(baselines)
            existing.setdefault("eval", {}).update(baselines)
        else:
            existing.setdefault(split, {}).update(baselines)
        path.write_text(json.dumps(existing, indent=2, sort_keys=True))
    else:
        path.write_text(json.dumps(baselines, indent=2, sort_keys=True))


def _default_picker(group: list[BenchmarkInstance]) -> BenchmarkInstance:
    return group[0]  # already sorted by load_benchmark_set


def format_for_prompt(
    name: str = STANDARD_SET_NAME,
    split: Split = "train",
    instance_picker: Callable[[list[BenchmarkInstance]], BenchmarkInstance] | None = None,
    annotate_baselines: bool = False,
    limit: int = 10
) -> str:
    """Render the structured per-family representation embedded in LLM prompts.

    For each op family present in the set, emits:
      ## <family>
      Template:
      <full template code>

      Example instance — <name>:
      <full instance code>

      Other <family> shapes in this set (N): name1, name2, ...

    Set `annotate_baselines=True` to append `(mlir baseline: X.XX ms)` after each
    listed name when baselines.json provides one — used by Layer 3.
    """
    picker = instance_picker or _default_picker
    instances = load_benchmark_set(name, split=split)
    groups = group_by_family(instances)
    baselines = load_baselines(name, split=split) if annotate_baselines else {}

    blocks: list[str] = [f"Benchmark set: {name} (split: {split}, {len(instances)} instance(s))"]
    for family in list(SUPPORTED_FAMILIES) + sorted(g for g in groups if g not in SUPPORTED_FAMILIES):
        group = groups.get(family)
        if not group:
            continue
        example = picker(group)
        others = [b for b in group if b.name != example.name]

        blocks.append(f"\n- {family}")

        try:
            tmpl = load_template(family)
            blocks.append(f"\nTemplate ({template_path(family).name}):\n```mlir\n{tmpl.rstrip()}\n```")
        except FileNotFoundError:
            blocks.append(f"\n(no template at {template_path(family)})")

        blocks.append(f"\nExample instance — {example.name}:\n```mlir\n{example.code.rstrip()}\n```")

        if others:
            def _label(b: BenchmarkInstance) -> str:
                if annotate_baselines and b.name in baselines and "mlir" in baselines[b.name]:
                    return f"{b.name} (mlir baseline: {baselines[b.name]['mlir']:.2f} ms)"
                return b.name
            total = len(others)
            if limit is not None and total > limit:
                sampled = random.sample(others, limit)
                sampled.sort(key=lambda b: b.name)
                other_lines = "\n".join(f"- {_label(b)}" for b in sampled)
                blocks.append(
                    f"\nOther {family} shapes in this set ({limit} sampled out of {total}):\n{other_lines}"
                )
            else:
                other_lines = "\n".join(f"- {_label(b)}" for b in others)
                blocks.append(f"\nOther {family} shapes in this set ({total}):\n{other_lines}")
        else:
            blocks.append(f"\n(no other {family} shapes in this set)")

    return "\n".join(blocks)
