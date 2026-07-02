#!/usr/bin/env python3
"""
transformers2mlir.py
--------------------
Convert HuggingFace / sequence transformer models to linalg-on-tensors MLIR.

Conversion pipeline (primary — ONNX route):
  PyTorch model → ONNX (torch.onnx.export)
                → shape inference (onnxruntime symbolic_shape_infer)
                → torch MLIR (torch-mlir-import-onnx / torch_mlir.tools.import_onnx)
                → linalg MLIR (torch-mlir-opt lowering passes)

Fallback (when CLI tools are unavailable):
  PyTorch model → linalg MLIR via torch_mlir.compile or torch_mlir.fx.export_and_import

Supported models: see data_utils.model_catalog.TRANSFORMER_MODELS.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import argparse
import os
import re
import subprocess
import sys
import traceback

from data_utils.model_catalog import TRANSFORMER_MODELS, MODEL_REGISTRY, TRANSFORMER_INPUT_SPECS

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NON_STRIPPED_DIR = os.path.join(PROJECT_ROOT, "data", "nn", "non_stripped_models")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "generated", "code_files")


# ── Post-processing helpers ─────────────────────────────────────────────────

def specialize_tensor_types(content: str, static_batch_size: int) -> tuple[str, int]:
    replaced = 0

    def _repl(m: re.Match) -> str:
        nonlocal replaced
        token = m.group(0)
        count = token.count("?")
        if count:
            replaced += count
            token = token.replace("?", str(static_batch_size))
        return token

    content = re.sub(r"tensor<[^>]+>", _repl, content)
    content = re.sub(r"!torch\.vtensor<\[[^\]]+\],[^>]+>", _repl, content)
    return content, replaced


def postprocess_linalg_file(linalg_path: str, static_batch_size: int = 0) -> None:
    with open(linalg_path, "r") as fh:
        content = fh.read()

    changed = False
    if "@main_graph" in content:
        content = content.replace("@main_graph", "@main")
        changed = True
        print("  Renamed @main_graph -> @main")

    if static_batch_size > 0 and "?" in content:
        content, replaced = specialize_tensor_types(content, static_batch_size)
        if replaced:
            changed = True
            print(f"  Specialized dynamic dims with static batch="
                  f"{static_batch_size} ({replaced} replacement(s))")

    if changed:
        with open(linalg_path, "w") as fh:
            fh.write(content)


# ── Weight stripping / backup ────────────────────────────────────────────────

def backup_and_strip(output_path: str, strip_weights: bool,
                     verbose: bool = False) -> None:
    import shutil
    os.makedirs(NON_STRIPPED_DIR, exist_ok=True)
    backup_path = os.path.join(NON_STRIPPED_DIR, os.path.basename(output_path))
    shutil.copy2(output_path, backup_path)
    print(f"  Non-stripped backup: {backup_path}")

    if not strip_weights:
        return
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    if size_mb > 1:
        print(f"  File is {size_mb:.1f} MB — stripping weights...")
        from data_utils.postprocess.strip_mlir import strip_weights as _strip
        reduction = _strip(output_path, output_path, verbose=verbose)
        print(f"  Stripped: {reduction:.1f}% reduction.")


# ── ONNX → linalg pipeline ─────────────────────────────────────────────────

def _cleanup_onnx_intermediates(base: str, keep_onnx: bool) -> None:
    if keep_onnx:
        return
    for ext in (".onnx", ".onnx.data", "_inferred.onnx"):
        p = base + ext
        if os.path.exists(p):
            os.remove(p)
            print(f"  Removed intermediate: {os.path.basename(p)}")


def _shape_infer_subprocess(base: str) -> str:
    onnx_input = f"{base}.onnx"
    result = subprocess.run(
        [
            sys.executable, "-m", "onnxruntime.tools.symbolic_shape_infer",
            "--input", f"{base}.onnx",
            "--output", f"{base}_inferred.onnx",
            "--auto_merge",
        ],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return f"{base}_inferred.onnx"
    print(f"  Shape inference warning: {result.stderr}")
    return onnx_input


def _import_onnx(base: str, onnx_input: str, opset: int) -> None:
    torch_mlir_path = f"{base}_torch.mlir"

    result = subprocess.run(
        ["torch-mlir-import-onnx", onnx_input, "-o", torch_mlir_path,
         "--opset-version", str(opset)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return

    result2 = subprocess.run(
        [sys.executable, "-m", "torch_mlir.tools.import_onnx",
         onnx_input, "-o", torch_mlir_path,
         "--opset-version", str(opset)],
        capture_output=True, text=True,
    )
    if result2.returncode != 0:
        raise RuntimeError(
            f"ONNX import failed:\n{result.stderr}\n{result2.stderr}"
        )


def _lower_to_linalg(base: str, method: str = "flags") -> None:
    torch_mlir_path = f"{base}_torch.mlir"
    linalg_path = f"{base}_linalg.mlir"

    opt_flags = [
        "--convert-torch-onnx-to-torch",
        "--torch-decompose-complex-ops",
        "--convert-torch-to-linalg",
        "--torch-backend-to-linalg-on-tensors-backend-pipeline",
    ]
    result = subprocess.run(
        ["torch-mlir-opt", torch_mlir_path] + opt_flags + ["-o", linalg_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"torch-mlir-opt failed:\n{result.stderr}")


def onnx_route(
    model: torch.nn.Module,
    model_name: str,
    example_inputs,
    output_dir: str,
    *,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    opset: int = 17,
    keep_onnx: bool = False,
    static_batch_size: int = 0,
    shape_infer: str = "subprocess",
    lowering_method: str = "flags",
    do_postprocess: bool = True,
) -> str:
    base = os.path.join(output_dir, model_name)
    if input_names is None:
        input_names = [f"input_{i}" for i in range(len(example_inputs)
                      if isinstance(example_inputs, tuple) else 1)]
    if output_names is None:
        output_names = ["output"]

    print(f"  Exporting {model_name} to ONNX (opset {opset})...")
    torch.onnx.export(
        model, example_inputs, f"{base}.onnx",
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
    )

    print("  Running shape inference...")
    onnx_input = _shape_infer_subprocess(base)

    print("  Importing ONNX to torch-mlir...")
    _import_onnx(base, onnx_input, opset)

    print("  Lowering to linalg MLIR...")
    _lower_to_linalg(base, method=lowering_method)

    linalg_path = f"{base}_linalg.mlir"

    if do_postprocess:
        postprocess_linalg_file(linalg_path, static_batch_size)

    _cleanup_onnx_intermediates(base, keep_onnx)

    return linalg_path


def direct_route(
    model: torch.nn.Module,
    model_name: str,
    example_inputs,
    output_dir: str,
    *,
    func_name: str = "main",
    try_compile: bool = True,
    static_batch_size: int = 0,
    do_postprocess: bool = True,
) -> str:
    from torch_mlir.compiler_utils import OutputType

    print(f"  Using direct torch_mlir route for {model_name}...")
    mlir_module = None

    if try_compile:
        try:
            import torch_mlir
            if hasattr(torch_mlir, "compile") and callable(torch_mlir.compile):
                try:
                    mlir_module = torch_mlir.compile(
                        model, example_inputs,
                        output_type=OutputType.LINALG_ON_TENSORS,
                        use_tracing=True, func_name=func_name,
                    )
                except TypeError:
                    mlir_module = torch_mlir.compile(
                        model, example_inputs,
                        output_type=OutputType.LINALG_ON_TENSORS,
                        func_name=func_name,
                    )
        except Exception as e:
            print(f"  torch_mlir.compile failed: {e}")

    if mlir_module is None:
        from torch_mlir.fx import export_and_import
        kwargs = dict(
            output_type=OutputType.LINALG_ON_TENSORS,
            func_name=func_name,
        )
        if isinstance(example_inputs, tuple):
            mlir_module = export_and_import(model, *example_inputs, **kwargs)
        else:
            mlir_module = export_and_import(model, example_inputs, **kwargs)

    linalg_path = os.path.join(output_dir, f"{model_name}_linalg.mlir")
    with open(linalg_path, "w") as f:
        f.write(str(mlir_module))

    if do_postprocess:
        postprocess_linalg_file(linalg_path, static_batch_size)

    return linalg_path


# ---------------------------------------------------------------------------
# Model + input factory
# ---------------------------------------------------------------------------

def _hf_repo(model_name: str) -> str:
    """Get HuggingFace repo ID from the catalog."""
    entry = MODEL_REGISTRY.get(model_name, {})
    return entry.get("hf_repo", model_name)


def _load_model_and_inputs(model_name: str):
    """Return (model, example_inputs, input_names, output_names)."""
    name = model_name.lower()
    repo = _hf_repo(name)

    if name == "bert":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).eval()
        inputs = tokenizer(
            "Hello from MLIR", return_tensors="pt",
            padding="max_length", max_length=16, truncation=True,
        )
        example_inputs = (inputs["input_ids"], inputs["attention_mask"])
        return model, example_inputs, ["input_ids", "attention_mask"], ["last_hidden_state"]

    if name == "distilbert":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).eval()
        encoded = tokenizer(
            "Hello from MLIR-RL!", return_tensors="pt",
            padding="max_length", max_length=16, truncation=True,
        )
        example_inputs = (encoded["input_ids"], encoded["attention_mask"])
        return model, example_inputs, ["input_ids", "attention_mask"], ["last_hidden_state"]

    if name == "roberta":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).eval()
        encoded = tokenizer(
            "Hello from MLIR-RL!", return_tensors="pt",
            padding="max_length", max_length=16, truncation=True,
        )
        example_inputs = (encoded["input_ids"], encoded["attention_mask"])
        return model, example_inputs, ["input_ids", "attention_mask"], ["last_hidden_state"]

    if name == "albert":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).eval()
        encoded = tokenizer(
            "Hello from MLIR-RL!", return_tensors="pt",
            padding="max_length", max_length=16, truncation=True,
        )
        example_inputs = (encoded["input_ids"], encoded["attention_mask"])
        return model, example_inputs, ["input_ids", "attention_mask"], ["last_hidden_state"]

    if name == "gpt2":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(repo)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModel.from_pretrained(repo).eval()
        base_model.config.use_cache = False

        class _GPT2ExportWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids, attention_mask):
                cache_position = torch.arange(
                    0, input_ids.shape[1], device=input_ids.device
                )
                return self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                ).last_hidden_state

        model = _GPT2ExportWrapper(base_model).eval()
        encoded = tokenizer(
            "Hello from MLIR-RL!", return_tensors="pt",
            padding="max_length", max_length=16, truncation=True,
        )
        example_inputs = (encoded["input_ids"], encoded["attention_mask"])
        return model, example_inputs, ["input_ids", "attention_mask"], ["last_hidden_state"]

    if name == "t5":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).eval()
        input_ids = tokenizer(
            "Studies show that owning a dog is good for you",
            padding="max_length", max_length=16, truncation=True,
            return_tensors="pt",
        ).input_ids
        return model, (input_ids,), ["input_ids"], ["last_hidden_state"]

    if name == "bart":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModel.from_pretrained(repo).eval()
        model.config.use_cache = False
        enc = tokenizer(
            ["The cat sat on the mat."],
            return_tensors="pt", padding="max_length",
            truncation=True, max_length=16,
        )
        example_inputs = (enc["input_ids"], enc["attention_mask"])
        return model, example_inputs, ["input_ids", "attention_mask"], ["last_hidden_state"]

    if name == "llama3_2_1b":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(repo)
        model = AutoModelForCausalLM.from_pretrained(
            repo, dtype=torch.float32
        ).eval()
        model.config.use_cache = False
        encoded = tokenizer(
            "Hello from MLIR-RL!", return_tensors="pt",
            padding="max_length", max_length=16, truncation=True,
        )
        return model, (encoded["input_ids"],), ["input_ids"], ["logits"]

    if name == "whisper_base":
        from transformers import WhisperModel
        model = WhisperModel.from_pretrained(repo).eval()
        encoder = model.encoder
        dummy_input = torch.randn(1, 80, 3000)
        return encoder, (dummy_input,), ["input_features"], ["last_hidden_state"]

    if name == "lstm":
        class LSTMSeq2Seq(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.LSTM(512, 768, 2, batch_first=True)
                self.decoder = nn.LSTM(768, 768, 2, batch_first=True)
                self.proj    = nn.Linear(768, 512)
            def forward(self, x):
                out, (h, c) = self.encoder(x)
                dec, _      = self.decoder(out, (h, c))
                return self.proj(dec)
        model = LSTMSeq2Seq().eval()
        example_inputs = torch.randn(1, 100, 512)
        return model, example_inputs, ["input"], ["output"]

    raise ValueError(f"Unknown model: {model_name}. Choose from: {TRANSFORMER_MODELS}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace / transformer models to linalg MLIR."
    )
    parser.add_argument(
        "--model", type=str, default="distilbert", choices=TRANSFORMER_MODELS,
        help="Transformer model to convert.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated files.",
    )
    parser.add_argument(
        "--backend", choices=["onnx", "direct"], default="onnx",
        help="Conversion backend.",
    )
    parser.add_argument(
        "--strip-weights", action="store_true", default=True,
        help="Strip large weight constants (default: enabled).",
    )
    parser.add_argument(
        "--keep-onnx", action="store_true", default=False,
        help="Keep intermediate .onnx/.onnx.data/_inferred.onnx files after a "
             "successful pipeline run (default: delete on success).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading {args.model}...")
    model, example_inputs, input_names, output_names = _load_model_and_inputs(args.model)

    output_path = None
    if args.backend == "onnx":
        try:
            output_path = onnx_route(
                model, args.model, example_inputs, args.output_dir,
                input_names=input_names, output_names=output_names,
                opset=17, keep_onnx=args.keep_onnx,
                shape_infer="subprocess",
            )
        except Exception as e:
            print(f"  ONNX route failed: {e}")
            traceback.print_exc()
            print("  Falling back to direct torch_mlir route...")
            output_path = direct_route(
                model, args.model, example_inputs, args.output_dir,
            )
    else:
        output_path = direct_route(
            model, args.model, example_inputs, args.output_dir,
        )

    backup_and_strip(output_path, args.strip_weights, args.verbose)
    print(f"\nDone: {output_path}")


if __name__ == "__main__":
    main()
