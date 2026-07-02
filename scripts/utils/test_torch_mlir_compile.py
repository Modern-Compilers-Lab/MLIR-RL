#!/usr/bin/env python3
# Minimal compile test for torch-mlir — runs a small model through compile
import sys, traceback
try:
    import torch
    import torch_mlir
    from torch import nn

    class SimpleNet(nn.Module):
        def forward(self, x):
            return torch.relu(x + 1.0)

    model = SimpleNet().eval()
    example_input = torch.randn(1, 8)

    print("Compiling model with torch-mlir (fx.export_and_import)...")
    # Use fx export_and_import when available
    try:
        from torch_mlir.fx import export_and_import
        module = export_and_import(model, example_input, output_type="linalg-on-tensors")
    except Exception:
        # Fallback for older APIs (if any)
        try:
            module = torch_mlir.fx.export_and_import(model, example_input, output_type="linalg-on-tensors")
        except Exception:
            raise
    print("Compile succeeded. IR preview:")
    ir = str(module)
    lines = ir.splitlines()
    for l in lines[:40]:
        print(l)
    if len(lines) > 40:
        print("... (truncated) ...")
    sys.exit(0)
except Exception:
    print("Compile test failed:")
    traceback.print_exc()
    sys.exit(2)
