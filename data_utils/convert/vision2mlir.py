#!/usr/bin/env python3
"""
vision2mlir.py
--------------
Convert vision models to linalg-on-tensors MLIR.

Conversion pipeline (primary — ONNX route):
  PyTorch model → ONNX (torch.onnx.export)
                → shape inference (onnxruntime symbolic_shape_infer)
                → torch MLIR (torch-mlir-import-onnx)
                → linalg MLIR (torch-mlir-opt lowering passes)

Fallback (when torch-mlir CLI tools are unavailable):
  PyTorch model → linalg MLIR via torch_mlir.fx.export_and_import (in-process)

Supported models: see data_utils.model_catalog.VISION_MODELS.
"""

from __future__ import annotations

import torch
import torchvision.models as tv_models
import argparse
import os
import re
import subprocess
import sys

from data_utils.model_catalog import VISION_MODELS, MODEL_REGISTRY

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


def _shape_infer_inprocess(base: str) -> str:
    import onnx
    try:
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
        model_proto = onnx.load(f"{base}.onnx")
        inferred = SymbolicShapeInference.infer_shapes(
            model_proto, auto_merge=True, guess_output_rank=True,
        )
        out = f"{base}_inferred.onnx"
        onnx.save(inferred, out)
        return out
    except ImportError:
        pass
    m = onnx.load(f"{base}.onnx")
    m2 = onnx.shape_inference.infer_shapes(m)
    out = f"{base}_inferred.onnx"
    onnx.save(m2, out)
    return out


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
    opset: int = 18,
    keep_onnx: bool = False,
    static_batch_size: int = 0,
    shape_infer: str = "inprocess",
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
    onnx_input = _shape_infer_inprocess(base)

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
# Model factory
# ---------------------------------------------------------------------------

_TV_WEIGHTS = {
    "resnet18":           tv_models.ResNet18_Weights,
    "resnet50":           tv_models.ResNet50_Weights,
    "resnext50_32x4d":    tv_models.ResNeXt50_32X4D_Weights,
    "efficientnet_b0":    tv_models.EfficientNet_B0_Weights,
    "mobilenet_v2":       tv_models.MobileNet_V2_Weights,
    "mobilenet_v3_small": tv_models.MobileNet_V3_Small_Weights,
    "densenet121":        tv_models.DenseNet121_Weights,
    "vit_b_16":           tv_models.ViT_B_16_Weights,
    "convnext_tiny":      tv_models.ConvNeXt_Tiny_Weights,
    "convnext_small":     tv_models.ConvNeXt_Small_Weights,
    "convnext_base":      tv_models.ConvNeXt_Base_Weights,
    "convnext_large":     tv_models.ConvNeXt_Large_Weights,
    "vgg11":              tv_models.VGG11_Weights,
    "vgg16":              tv_models.VGG16_Weights,
}


def _load_model(model_name: str) -> torch.nn.Module:
    entry = MODEL_REGISTRY.get(model_name)
    if entry is None or entry["category"] != "vision":
        raise ValueError(f"Unknown vision model: {model_name}. Choose from: {VISION_MODELS}")

    framework = entry["framework"]

    if framework == "torchvision":
        tv_name = entry["torchvision_name"]
        weights_cls = _TV_WEIGHTS.get(tv_name)
        loader_fn = getattr(tv_models, tv_name)
        return loader_fn(weights=weights_cls.DEFAULT).eval()

    if framework == "ultralytics":
        from ultralytics import YOLO
        yolo = YOLO(f"{model_name}.pt")
        return yolo.model.eval()

    raise ValueError(f"Unsupported framework '{framework}' for {model_name}")


def _get_dummy_input_size(model_name: str) -> int:
    entry = MODEL_REGISTRY.get(model_name, {})
    return entry.get("input_shape", (224, 224))[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert vision models to linalg MLIR."
    )
    parser.add_argument(
        "--model", type=str, default="resnet18", choices=VISION_MODELS,
        help="Vision model to convert.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated files.",
    )
    parser.add_argument(
        "--backend", choices=["onnx", "direct"], default="onnx",
        help="Conversion backend. 'onnx' (default) uses the ONNX route; "
             "'direct' uses the torch_mlir Python API directly.",
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
    model = _load_model(args.model)
    dummy_input_size = _get_dummy_input_size(args.model)
    dummy_input = torch.randn(1, 3, dummy_input_size, dummy_input_size)

    output_path = None
    if args.backend == "onnx":
        try:
            output_path = onnx_route(
                model, args.model, dummy_input, args.output_dir,
                input_names=["input"], output_names=["output"],
                opset=18, keep_onnx=args.keep_onnx,
                shape_infer="inprocess",
            )
        except Exception as e:
            print(f"  ONNX route failed: {e}")
            print("  Falling back to direct torch_mlir route...")
            output_path = direct_route(
                model, args.model, dummy_input, args.output_dir,
                func_name=args.model, try_compile=False,
            )
    else:
        output_path = direct_route(
            model, args.model, dummy_input, args.output_dir,
            func_name=args.model, try_compile=False,
        )

    backup_and_strip(output_path, args.strip_weights, args.verbose)
    print(f"\nDone: {output_path}")


if __name__ == "__main__":
    main()
