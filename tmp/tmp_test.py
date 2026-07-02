import sys
import os
import mlir.ir as ir
from rl_autoschedular_paper.transforms import transform_tile, transform_vectorize

# Set up Context
ctx = ir.Context()

# Simple valid module with tagged operation (tag = "test_op")
code = """
module {
  func.func @main(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, i64) {
    %c0 = arith.constant 0 : i64
    %0 = linalg.matmul {tag = "test_op"} ins(%arg0, %arg0 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%arg0 : tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0, %c0 : tensor<8x8xf32>, i64
  }
}
"""
module = ir.Module.parse(code, ctx)

print("Applying valid tile transform...")
transform_tile(module, "test_op", [2, 2, 2])
print("Successfully tiled! Transformed code:")
print(str(module))

print("Applying vectorization to tiled code (this may fail if not structured correctly, testing error propagation)...")
try:
    transform_vectorize(module, "test_op")
    print("Vectorization completed.")
except Exception as e:
    print("Caught expected exception from invalid vectorization:", e)

print("Done! Exiting cleanly.")
