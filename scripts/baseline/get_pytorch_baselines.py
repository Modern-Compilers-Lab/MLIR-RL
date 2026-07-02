#!/usr/bin/env python3
"""
get_pytorch_baselines.py — canonical PyTorch eager + JIT baseline measurement.

Measures all models registered in pytorch_models.json.  Supports both full
Slurm runs and ad-hoc debugging (--models gpt2 vit_b_16).  GPT2 variants
use a JIT-compatible wrapper (utils/gpt2_jit_compat.py); all other models
use the standard HuggingFace / torchvision loaders.

Outputs:
  JSON  — {model: {eager_ns, jit_ns, ...}, ...}  (incremental, primary)
  CSV   — model,mlir_baseline_ns,pytorch_eager_ns,pytorch_jit_ns,mlir_rl_ns

Usage:
  python scripts/baseline/get_pytorch_baselines.py                                    # all models
  python scripts/baseline/get_pytorch_baselines.py --models gpt2 vit_b_16            # ad-hoc
  python scripts/baseline/get_pytorch_baselines.py --output results/full_model_1/pytorch_times.json
"""

import os, sys, json, time, gc, csv, argparse
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DEVICE = torch.device("cpu")
WARMUP = 10
MEASURE = 20


# =========================================================================
#  Vision models (torchvision)
# =========================================================================

def _vision_input():
    return (torch.randn(1, 3, 224, 224),)


def _load_vision(model_name, cfg):
    import torchvision.models as tv_models
    fn = getattr(tv_models, cfg["fn"])
    weights_parts = cfg["weights"].split(".")
    weights_obj = tv_models
    for part in weights_parts:
        weights_obj = getattr(weights_obj, part)
    model = fn(weights=weights_obj).eval()
    return model, _vision_input()


# =========================================================================
#  Transformer models (HuggingFace)
# =========================================================================

class _HFWrapper(torch.nn.Module):
    """Wrap HuggingFace model to return tensor only (not dict)."""
    def __init__(self, m, has_mask=True):
        super().__init__()
        self.m = m
        self.has_mask = has_mask

    def forward(self, ids, mask=None):
        if self.has_mask:
            out = self.m(input_ids=ids, attention_mask=mask)
        else:
            out = self.m(ids)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out


def _load_transformer(model_name, cfg):
    transformers = __import__("transformers")
    tokenizer_cls = getattr(transformers, cfg["tokenizer"])
    model_cls = getattr(transformers, cfg["cls"])

    tokenizer = tokenizer_cls.from_pretrained(cfg["hf"])
    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model_cls.from_pretrained(cfg["hf"]).eval()

    if "gpt2" in model_name and "use_cache" in dir(model.config):
        model.config.use_cache = False

    text = "Studies show that owning a dog is good for you"
    enc = tokenizer(
        text, padding="max_length", max_length=16, truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    mask = enc.get("attention_mask", None)

    if mask is not None:
        model = _HFWrapper(model, has_mask=True)
        inputs = (input_ids, mask)
    else:
        model = _HFWrapper(model, has_mask=False)
        inputs = (input_ids,)

    return model, inputs


# =========================================================================
#  LSTM (custom)
# =========================================================================

def _load_lstm(model_name, cfg):
    class LSTMSeq2Seq(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.LSTM(512, 768, 2, batch_first=True)
            self.decoder = torch.nn.LSTM(768, 768, 2, batch_first=True)
            self.proj = torch.nn.Linear(768, 512)

        def forward(self, x):
            out, (h, c) = self.encoder(x)
            dec, _ = self.decoder(out, (h, c))
            return self.proj(dec)

    return LSTMSeq2Seq().eval(), (torch.randn(1, 100, 512),)


# =========================================================================
#  GNN (local)
# =========================================================================

def _load_gnn(model_name, cfg):
    sys.path.insert(0, str(PROJECT_ROOT))
    mod = __import__(cfg["src"], fromlist=[cfg["cls"]])
    cls = getattr(mod, cfg["cls"])
    model = cls().eval()
    return model, (torch.randn(128, 64), torch.randn(128, 128))


# =========================================================================
#  GPT2 JIT-compatible (uses utils/gpt2_jit_compat.py)
# =========================================================================

def _load_gpt2_jit(model_name, cfg):
    """Load gpt2 variant via JIT-compatible wrapper."""
    from utils.gpt2_jit_compat import make_gpt2_jit_compat
    return make_gpt2_jit_compat(cfg["hf"])


# =========================================================================
#  Dispatcher
# =========================================================================

LOADERS = {
    "vision":      _load_vision,
    "transformer": _load_transformer,
    "lstm":        _load_lstm,
    "gnn":         _load_gnn,
}

# Models that use the GPT2 JIT-compat wrapper instead of _load_transformer
GPT2_JIT_MODELS = {"gpt2", "gpt2-medium", "gpt2-large"}


# =========================================================================
#  Timing helpers
# =========================================================================

def _run_model(model, inputs):
    if isinstance(inputs, tuple):
        return model(*inputs)
    return model(inputs)


def _time_model(model, inputs):
    for _ in range(WARMUP):
        _run_model(model, inputs)
    times = []
    for _ in range(MEASURE):
        start = time.perf_counter_ns()
        _run_model(model, inputs)
        times.append(time.perf_counter_ns() - start)
    return int(np.median(times))


def _time_jit(model, inputs):
    """Trace then script fallback. Returns median ns or None."""
    for method in ["trace", "script"]:
        try:
            if method == "trace":
                jitted = torch.jit.trace(model, inputs, check_trace=False)
            else:
                jitted = torch.jit.script(model)
            jitted = jitted.eval()
            for _ in range(WARMUP + 10):
                _run_model(jitted, inputs)
            return _time_model(jitted, inputs)
        except Exception:
            continue
    return None


# =========================================================================
#  Single-model processing
# =========================================================================

def process_model(model_name, cfg):
    """Load + time a single model. Returns (eager_ns, jit_ns | None)."""
    # Find the model in the config
    model_type = None
    model_cfg = None
    for mt, models in cfg.items():
        if isinstance(models, dict) and model_name in models:
            model_type = mt
            model_cfg = models[model_name]
            break
    if model_type is None or model_cfg is None:
        raise ValueError(f"Model '{model_name}' not found in config")

    # GPT2 uses JIT-compat wrapper
    if model_name in GPT2_JIT_MODELS:
        model, inputs = _load_gpt2_jit(model_name, model_cfg)
    else:
        loader = LOADERS[model_type]
        model, inputs = loader(model_name, model_cfg)

    model = model.to(DEVICE)
    if isinstance(inputs, tuple):
        inputs = tuple(x.to(DEVICE) for x in inputs)
    else:
        inputs = inputs.to(DEVICE)

    print(f"    Eager  ({WARMUP} warmup + {MEASURE} measure) ...", end=" ", flush=True)
    eager_ns = _time_model(model, inputs)
    print(f"{eager_ns:,} ns")

    print(f"    JIT    ({WARMUP} warmup + {MEASURE} measure) ...", end=" ", flush=True)
    jit_ns = _time_jit(model, inputs)
    if jit_ns:
        print(f"{jit_ns:,} ns")
    else:
        print("FAILED")

    del model
    gc.collect()
    return eager_ns, jit_ns


# =========================================================================
#  Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyTorch eager + JIT baseline measurement"
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Models to measure (default: all in config)",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "results/full_model_0/pytorch_models.json"),
        help="Config JSON mapping models to loaders",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results/full_model_0/pytorch_times.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--csv-output", default=None,
        help="Optional CSV output path",
    )
    parser.add_argument(
        "--mlir-baselines", default=None,
        help="JSON file with cached MLIR baseline times",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    # Collect all model names from config
    all_models = set()
    for model_type, models in cfg.items():
        if isinstance(models, dict):
            all_models.update(models.keys())

    target_models = args.models if args.models else sorted(all_models)
    unknown = set(target_models) - all_models
    if unknown:
        print(f"Unknown models (not in config): {unknown}")
        sys.exit(1)

    # Load MLIR baselines if provided
    mlir_baselines = {}
    if args.mlir_baselines:
        mlir_path = Path(args.mlr_baselines)
        if mlir_path.exists():
            raw = json.loads(mlir_path.read_text())
            for k, v in raw.items():
                if isinstance(v, dict) and v.get("baseline_ns"):
                    mlir_baselines[k] = v["baseline_ns"]
                elif isinstance(v, (int, float)):
                    mlir_baselines[k] = v

    # Load existing JSON results
    output_path = Path(args.output)
    results = {}
    if output_path.exists():
        try:
            results = json.loads(output_path.read_text())
        except Exception:
            pass
    already = set(results.keys()) & set(target_models)
    if already:
        print(f"Skipping {len(already)} models already measured: {sorted(already)}")

    pending = sorted(set(target_models) - set(results.keys()))
    if not pending:
        print("All models already measured.")
        return

    print(f"Models to measure: {len(pending)}")
    print(f"Device: {DEVICE}")

    for model_name in pending:
        print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
        try:
            eager_ns, jit_ns = process_model(model_name, cfg)
            results[model_name] = {
                "eager_ns": eager_ns,
                "jit_ns": jit_ns if jit_ns else None,
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {
                "eager_ns": None, "jit_ns": None, "error": str(e),
            }

        # Save JSON incrementally
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Model':<22s} {'Eager (ns)':>15s} {'JIT (ns)':>15s}")
    print("-" * 55)
    for model_name in sorted(results):
        r = results[model_name]
        e = f"{r['eager_ns']:>15,}" if r.get("eager_ns") else "           FAIL"
        j = f"{r['jit_ns']:>15,}" if r.get("jit_ns") else "           FAIL"
        print(f"{model_name:<22s} {e} {j}")

    print(f"\nSaved -> {output_path}")

    # CSV output (optional)
    if args.csv_output:
        csv_path = Path(arg.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for model_name in sorted(results):
            r = results[model_name]
            rows.append({
                "model": model_name,
                "mlir_baseline_ns": mlir_baselines.get(model_name, ""),
                "pytorch_eager_ns": r.get("eager_ns", ""),
                "pytorch_jit_ns": r.get("jit_ns", ""),
                "mlir_rl_ns": "",
            })
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "model", "mlir_baseline_ns", "pytorch_eager_ns",
                "pytorch_jit_ns", "mlir_rl_ns",
            ])
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV saved -> {csv_path}")


if __name__ == "__main__":
    main()
