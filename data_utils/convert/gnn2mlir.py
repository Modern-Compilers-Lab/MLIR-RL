#!/usr/bin/env python3
"""
gnn2mlir.py
-----------
Convert Graph Neural Network models to linalg-on-tensors MLIR.

GNNs operate on sparse graphs, but torch-mlir only handles dense tensor
computations. The approach here is to represent each model as a fixed-size
dense graph (N nodes, F features) so that:
  - neighborhood aggregation → dense matrix multiply  (A @ X  or A @ X @ W)
  - node feature transform   → nn.Linear

This preserves all the meaningful operator shapes (matmul, generic activations)
that the RL scheduler will see in practice, while staying fully static.

Supported models: gcn, graphsage, gat, gin

Conversion goes through the ONNX route by default (same pipeline as
vision2mlir / transformers2mlir), with the direct torch_mlir route as fallback.

Usage:
    python data_utils.convert.gnn2mlir --model gcn
    python data_utils.convert.gnn2mlir --model gat --backend direct
    python data_utils.convert.gnn2mlir --model all   # run every supported model
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import re
import subprocess
import sys

from data_utils.model_catalog import GNN_MODELS

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NON_STRIPPED_DIR = os.path.join(PROJECT_ROOT, "data", "nn", "non_stripped_models")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "nn", "raw")

SUPPORTED_MODELS = GNN_MODELS

# Fixed graph: 128 nodes, 64 input features, 3 layers of 128 hidden, 32 output
N_NODES    = 128
IN_FEATS   = 64
HIDDEN     = 128
OUT_FEATS  = 32
N_HEADS    = 4   # GAT attention heads


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


# ── Weight stripping ─────────────────────────────────────────────────────────

def strip_only(output_path: str, strip_weights: bool,
               verbose: bool = False) -> None:
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


def _shape_infer_fallback_copy(base: str) -> str:
    import shutil
    shutil.copy(f"{base}.onnx", f"{base}_inferred.onnx")
    return f"{base}_inferred.onnx"


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


def _lower_to_linalg(base: str, method: str = "pipeline") -> None:
    torch_mlir_path = f"{base}_torch.mlir"
    linalg_path = f"{base}_linalg.mlir"

    torch_mlir_opt = os.path.join(os.path.dirname(sys.executable), "torch-mlir-opt")
    passes = (
        "torch-lower-to-backend-contract,"
        "torch-backend-to-linalg-on-tensors-backend-pipeline"
    )
    subprocess.run(
        [torch_mlir_opt, torch_mlir_path,
         f"--pass-pipeline=builtin.module({passes})",
         "-o", linalg_path],
        check=True,
    )


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
    static_batch_size: int = 1,
    shape_infer: str = "fallback_copy",
    lowering_method: str = "pipeline",
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
    onnx_input = _shape_infer_fallback_copy(base)

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
    static_batch_size: int = 1,
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
# Model definitions (dense, static, torch-mlir–compatible)
# ---------------------------------------------------------------------------

class GCN(nn.Module):
    """Graph Convolutional Network (Kipf & Welling, 2017).
    Aggregation: H' = ReLU(A_hat @ H @ W)  where A_hat = D^{-1/2} A D^{-1/2}
    Represented as two matmuls + activation — identical to what a sparse GCN
    produces after densification.
    """
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(IN_FEATS, HIDDEN, bias=False)
        self.W2 = nn.Linear(HIDDEN,   HIDDEN, bias=False)
        self.W3 = nn.Linear(HIDDEN,   OUT_FEATS, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(adj @ self.W1(x))
        h = F.relu(adj @ self.W2(h))
        return adj @ self.W3(h)


class GraphSAGE(nn.Module):
    """GraphSAGE (Hamilton et al., 2017) — mean aggregation.
    For each layer: h_v = ReLU(W @ concat(h_v, mean(h_N(v))))
    With fixed dense adj, mean aggregation = row-wise mean of adj @ X.
    """
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(IN_FEATS * 2, HIDDEN)
        self.W2 = nn.Linear(HIDDEN * 2,   HIDDEN)
        self.W3 = nn.Linear(HIDDEN * 2,   OUT_FEATS)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        neigh1 = adj @ x
        h1 = F.relu(self.W1(torch.cat([x, neigh1], dim=-1)))
        neigh2 = adj @ h1
        h2 = F.relu(self.W2(torch.cat([h1, neigh2], dim=-1)))
        neigh3 = adj @ h2
        return self.W3(torch.cat([h2, neigh3], dim=-1))


class GAT(nn.Module):
    """Graph Attention Network (Velickovic et al., 2018) — multi-head.
    Attention scores: e_ij = LeakyReLU(a^T [W h_i || W h_j])
    For a fixed dense graph this is a batched outer product + softmax + matmul.
    """
    def __init__(self):
        super().__init__()
        self.head_dim = HIDDEN // N_HEADS
        self.W   = nn.Linear(IN_FEATS, HIDDEN, bias=False)
        self.a   = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.W2  = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.a2  = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.out = nn.Linear(HIDDEN, OUT_FEATS)

    def _attn_layer(self, h: torch.Tensor, W: nn.Linear, a: nn.Linear) -> torch.Tensor:
        Wh = W(h).view(N_NODES, N_HEADS, self.head_dim)
        src = Wh.unsqueeze(1).expand(N_NODES, N_NODES, N_HEADS, self.head_dim)
        dst = Wh.unsqueeze(0).expand(N_NODES, N_NODES, N_HEADS, self.head_dim)
        e = F.leaky_relu(a(torch.cat([src, dst], dim=-1)).squeeze(-1))
        alpha = F.softmax(e, dim=1)
        out = (alpha.unsqueeze(-1) * Wh.unsqueeze(0)).sum(dim=1)
        return out.reshape(N_NODES, HIDDEN)

    def forward(self, x: torch.Tensor, _adj: torch.Tensor) -> torch.Tensor:
        h = F.elu(self._attn_layer(self.W(x), self.W2, self.a2))
        return self.out(h)


class GIN(nn.Module):
    """Graph Isomorphism Network (Xu et al., 2019).
    Aggregation: h_v = MLP((1+eps) h_v + sum_{u in N(v)} h_u)
    sum aggregation = adj @ X (dense); eps is a learnable scalar.
    """
    def __init__(self):
        super().__init__()
        self.eps1 = nn.Parameter(torch.zeros(1))
        self.eps2 = nn.Parameter(torch.zeros(1))
        self.mlp1 = nn.Sequential(nn.Linear(IN_FEATS, HIDDEN), nn.ReLU(),
                                   nn.Linear(HIDDEN, HIDDEN))
        self.mlp2 = nn.Sequential(nn.Linear(HIDDEN, HIDDEN),   nn.ReLU(),
                                   nn.Linear(HIDDEN, HIDDEN))
        self.mlp3 = nn.Sequential(nn.Linear(HIDDEN, HIDDEN),   nn.ReLU(),
                                   nn.Linear(HIDDEN, OUT_FEATS))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.mlp1((1 + self.eps1) * x   + adj @ x)
        h = self.mlp2((1 + self.eps2) * h   + adj @ h)
        return self.mlp3(h)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _load_model_and_inputs(model_name: str):
    """Return (model, example_inputs, input_names, output_names)."""
    name = model_name.lower()
    x   = torch.randn(N_NODES, IN_FEATS)
    raw = torch.rand(N_NODES, N_NODES)
    adj = (raw + raw.t()) / 2
    adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1e-6)

    models_map = {
        "gcn":       (GCN,       (x, adj), ["x", "adj"], ["out"]),
        "graphsage": (GraphSAGE, (x, adj), ["x", "adj"], ["out"]),
        "gat":       (GAT,       (x, adj), ["x", "adj"], ["out"]),
        "gin":       (GIN,       (x, adj), ["x", "adj"], ["out"]),
    }
    if name not in models_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {SUPPORTED_MODELS}")
    cls, inputs, in_names, out_names = models_map[name]
    return cls().eval(), inputs, in_names, out_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(model_name: str, output_dir: str, backend: str = "onnx",
            strip_weights: bool = False, keep_onnx: bool = False,
            verbose: bool = False,
            static_batch_size: int = 1) -> str:
    """Convert one GNN model. Returns path to produced *_linalg.mlir."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading {model_name}...")
    model, example_inputs, in_names, out_names = _load_model_and_inputs(model_name)
    output_path = None

    if backend == "onnx":
        try:
            output_path = onnx_route(
                model, model_name, example_inputs, output_dir,
                input_names=in_names, output_names=out_names,
                opset=18, keep_onnx=keep_onnx,
                static_batch_size=static_batch_size,
                shape_infer="fallback_copy",
                lowering_method="pipeline",
            )
        except Exception as e:
            print(f"  ONNX route failed: {e}")
            print("  Falling back to direct route...")
            output_path = direct_route(
                model, model_name, example_inputs, output_dir,
                static_batch_size=static_batch_size,
            )
    else:
        output_path = direct_route(
            model, model_name, example_inputs, output_dir,
            static_batch_size=static_batch_size,
        )

    strip_only(output_path, strip_weights, verbose)
    print(f"\nDone: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Graph Neural Network models to linalg MLIR."
    )
    parser.add_argument(
        "--model", type=str, default="gcn",
        choices=SUPPORTED_MODELS + ["all"],
        help="GNN model to convert. Use 'all' to run every supported model.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: data/nn/raw).",
    )
    parser.add_argument(
        "--backend", choices=["onnx", "direct"], default="onnx",
        help="Conversion backend.",
    )
    parser.add_argument(
        "--strip-weights", action="store_true", default=False,
    )
    parser.add_argument(
        "--keep-onnx", action="store_true", default=False,
        help="Keep ONNX intermediate files after success.",
    )
    parser.add_argument(
        "--static-batch-size", type=int, default=0,
        help="Static value used to replace dynamic '?' dimensions in final linalg MLIR. "
             "Set 0 to preserve dynamic shapes. Default: 0",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    targets = SUPPORTED_MODELS if args.model == "all" else [args.model]
    for m in targets:
        convert(m, args.output_dir, args.backend,
                args.strip_weights, args.keep_onnx, args.verbose,
                static_batch_size=args.static_batch_size)


if __name__ == "__main__":
    main()
