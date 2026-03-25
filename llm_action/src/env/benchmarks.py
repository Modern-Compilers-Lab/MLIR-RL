import json
import logging
from pathlib import Path
from dataclasses import dataclass

from llm_action.src.config import DATA_DIR

logger = logging.getLogger(__name__)

@dataclass
class Benchmark:
    name: str
    code: str
    base_exec_time_ms: float

def load_benchmarks(benchmarks_dir=DATA_DIR, name="matmul", executor=None) -> list[Benchmark]:
    bdir = Path(benchmarks_dir) / name
    if not bdir.exists():
        raise FileNotFoundError(f"Benchmarks directory not found: {bdir}")

    mlir_files = sorted(bdir.glob("*.mlir"))
    if not mlir_files:
        raise FileNotFoundError(f"No .mlir files in {bdir}")

    cache_path = bdir / "baselines.json"
    cached: dict[str, float] = {}
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    benchmarks = []
    needs_measurement = []

    for f in mlir_files:
        name = f.stem
        code = f.read_text()
        if name in cached:
            benchmarks.append(Benchmark(name, code, cached[name]))
        else:
            benchmarks.append(Benchmark(name, code, -1.0))
            needs_measurement.append(len(benchmarks) - 1)

    if needs_measurement and executor is not None:
        logger.info(f"Measuring baselines for {len(needs_measurement)} benchmarks...")
        for idx in needs_measurement:
            b = benchmarks[idx]
            try:
                t, ok = executor.execute(b.code)
                b.base_exec_time_ms = t if ok and t > 0 else 1.0
                cached[b.name] = b.base_exec_time_ms
                logger.info(f"  {b.name}: {b.base_exec_time_ms:.2f} ms")
            except Exception as e:
                logger.warning(f"  {b.name}: baseline failed: {e}")
                b.base_exec_time_ms = 1.0
                cached[b.name] = 1.0

        try:
            cache_path.write_text(json.dumps(cached, indent=2))
        except OSError:
            pass

    elif needs_measurement:
        logger.warning(f"{len(needs_measurement)} benchmarks need baseline measurement (deferred to reset)")

    return benchmarks
