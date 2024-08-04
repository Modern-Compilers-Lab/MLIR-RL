from fusion_utils.transforms import (
    evaluate_code_2,
    transform_dialect_TP,
    transform_dialect_tile,
    transform_dialect_vectorise,
    transform_dialect_vectorise_img2col,
    transform_dialect_img2col,
    transform_dialect_fuse,
    get_raw_ast_info, get_ast,
    transform_dialect_vectorise_,
    apply_conv2d_decomposition,
    
)

from utils.transform_utils import evaluate_code_with_timeout
from utils.observation_utils import function_wrapper, lower_linalg_to_loops

import os


matmul = """

module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
func.func private @printFlops(f64)
func.func private @printI64(i64)
func.func private @printNewline()
func.func private @printMemrefF32(tensor<*xf32>)


func.func @matmul() -> tensor<512x512xf32>{
  
%val = arith.constant 2.00000e+00 : f32
%zero = arith.constant 0.00000e+00 : f32

%tmp_arg0 = bufferization.alloc_tensor() : tensor<512x32xf32>
%arg0 = linalg.fill ins(%val : f32) outs(%tmp_arg0 : tensor<512x32xf32>) -> tensor<512x32xf32>
%tmp_arg1 = bufferization.alloc_tensor() : tensor<32x512xf32>
%arg1 = linalg.fill ins(%val : f32) outs(%tmp_arg1 : tensor<32x512xf32>) -> tensor<32x512xf32>
%tmp_arg2 = bufferization.alloc_tensor() : tensor<512x512xf32>
%arg2 = linalg.fill ins(%val : f32) outs(%tmp_arg2 : tensor<512x512xf32>) -> tensor<512x512xf32>

%t0 = func.call @nanoTime() : () -> (i64)

%return_arg = linalg.matmul {tag = "operation_1"} ins(%arg0, %arg1 : tensor<512x32xf32>, tensor<32x512xf32>) outs(%arg2 : tensor<512x512xf32>) -> tensor<512x512xf32>
%t = func.call @nanoTime() : () -> (i64)
%delta = arith.subi %t, %t0 : i64
%fp = arith.uitofp %delta : i64 to f64
// func.call @printFlops(%fp) : (f64) -> ()
func.call @printI64(%delta) : (i64) -> ()
func.call @printNewline() : () -> ()

return %return_arg : tensor<512x512xf32> 
}

func.func @main(){
      %c1 = arith.constant 1: index
    %c0 = arith.constant 0 : index
    %n = arith.constant 2: index
    scf.for %i = %c0 to %n step %c1 {
      %outputmain = func.call @matmul() : () -> tensor<512x512xf32>
    }
    return
}
}

"""

conv2d = """

module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<4x8x61x29xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<4x64x128x512x128x64xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst : f32) outs(%0 : tensor<4x64x128x64xf32>) -> tensor<4x64x128x64xf32>
    %2 = bufferization.alloc_tensor() : tensor<8x64x3x3xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst : f32) outs(%2 : tensor<8x64x3x3xf32>) -> tensor<8x64x3x3xf32>
    %4 = bufferization.alloc_tensor() : tensor<4x8x61x29xf32>
    %5 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%4 : tensor<4x8x61x29xf32>) -> tensor<4x8x61x29xf32>
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_3"} ins(%1, %3 : tensor<4x64x128x64xf32>, tensor<8x64x3x3xf32>) outs(%5 : tensor<4x8x61x29xf32>) -> tensor<4x8x61x29xf32>
    %8 = call @nanoTime() : () -> i64
    %9 = arith.subi %8, %6 : i64
    %10 = arith.uitofp %9 : i64 to f64
    call @printI64(%9) : (i64) -> ()
    call @printNewline() : () -> ()
    return %7 : tensor<4x8x61x29xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<4x8x61x29xf32>
    }
    return
  }
}


"""

pool = """

module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<4x128x30x30xf32>{
    %val = arith.constant 2.00000e+00 : f32
    %zero = arith.constant 0.00000e+00 : f32
    %tmp_input = bufferization.alloc_tensor() : tensor<4x128x64x64xf32>
    %input = linalg.fill ins(%val : f32) outs(%tmp_input : tensor<4x128x64x64xf32>) -> tensor<4x128x64x64xf32>
    %tmp_filter = bufferization.alloc_tensor() : tensor<5x5xf32>
    %filter = linalg.fill ins(%val : f32) outs(%tmp_filter : tensor<5x5xf32>) -> tensor<5x5xf32>
    %tmp_init = bufferization.alloc_tensor() : tensor<4x128x30x30xf32>
    %init = linalg.fill ins(%val : f32) outs(%tmp_init : tensor<4x128x30x30xf32>) -> tensor<4x128x30x30xf32>
    %t0 = func.call @nanoTime() : () -> (i64)
    %return_arg = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_1"} ins (%input, %filter: tensor<4x128x64x64xf32>, tensor<5x5xf32>) outs (%init: tensor<4x128x30x30xf32>) -> tensor<4x128x30x30xf32>
    %t = func.call @nanoTime() : () -> (i64)
    %delta = arith.subi %t, %t0 : i64
    %fp = arith.uitofp %delta : i64 to f64
    // func.call @printFlops(%fp) : (f64) -> ()
    func.call @printI64(%delta) : (i64) -> ()
    func.call @printNewline() : () -> ()
    return %return_arg : tensor<4x128x30x30xf32> 
  }
  func.func @main(){
    %c1 = arith.constant 1: index
    %c0 = arith.constant 0 : index
    %n = arith.constant 2: index
    scf.for %i = %c0 to %n step %c1 {
    %outputmain = func.call @matmul() : () -> tensor<4x128x30x30xf32>
    }
    return
  }
}

"""

add = """

module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<64x128x512xf32>{
    %val = arith.constant 2.00000e+00 : f32
    %zero = arith.constant 0.00000e+00 : f32
    %tmp_arg0 = bufferization.alloc_tensor() : tensor<64x128x512xf32>
    %arg0 = linalg.fill ins(%val : f32) outs(%tmp_arg0 : tensor<64x128x512xf32>) -> tensor<64x128x512xf32>
    %tmp_arg1 = bufferization.alloc_tensor() : tensor<64x128x512xf32>
    %arg1 = linalg.fill ins(%val : f32) outs(%tmp_arg1 : tensor<64x128x512xf32>) -> tensor<64x128x512xf32>
    %tmp_arg2 = bufferization.alloc_tensor() : tensor<64x128x512xf32>
    %arg2 = linalg.fill ins(%val : f32) outs(%tmp_arg2 : tensor<64x128x512xf32>) -> tensor<64x128x512xf32>
    %t0 = func.call @nanoTime() : () -> (i64)
    %return_arg = linalg.add {tag = "operation_1"} ins(%arg0, %arg1: tensor<64x128x512xf32>, tensor<64x128x512xf32>) outs(%arg2: tensor<64x128x512xf32>) -> tensor<64x128x512xf32>
    %t = func.call @nanoTime() : () -> (i64)
    %delta = arith.subi %t, %t0 : i64
    %fp = arith.uitofp %delta : i64 to f64
    // func.call @printFlops(%fp) : (f64) -> ()
    func.call @printI64(%delta) : (i64) -> ()
    func.call @printNewline() : () -> ()
    return %return_arg : tensor<64x128x512xf32> 
  }
  func.func @main(){
    %c1 = arith.constant 1: index
    %c0 = arith.constant 0 : index
    %n = arith.constant 2: index
    scf.for %i = %c0 to %n step %c1 {
    %outputmain = func.call @matmul() : () -> tensor<64x128x512xf32>
    }
    return
  }
}

"""

# code = conv2d.strip()
# code = matmul.strip()
# code = pool.strip()
code = add.strip()

tmp_file = 'examples/temp_nn_mlir.mlir'

# code = transform_dialect_TP(code, "operation_1", [2, 0, 16], tmp_file)

# code = transform_dialect_TP(code, "operation_1", [4, 8, 0, 0, 0, 5], tmp_file)
# code = transform_dialect_tile(code, "operation_1", [0, 8, 5, 0, 5, 5], tmp_file)
# code = transform_dialect_tile(code, "operation_1", [0, 0, 1, 0, 1, 0], tmp_file)
# code = apply_conv2d_decomposition(code, "operation_1", tmp_file)
# code = transform_dialect_vectorise(code, "operation_1", tmp_file)

# code = transform_dialect_img2col(code, "operation_3")

# code = transform_dialect_TP(code, "operation_1", [ 4, 8, 32], tmp_file)
# code = transform_dialect_tile(code, "operation_1", [4, 8, 32], tmp_file)

# code = transform_dialect_vectorise_img2col(code, "operation_1")

code = transform_dialect_TP(code, "operation_1", [8], tmp_file)
code = transform_dialect_vectorise(code, "operation_1", tmp_file)


res = []
for _ in range(30):
  exec = evaluate_code_with_timeout(code, 600, tmp_file)
  print(exec*1e-9)
  res.append(exec)

print(sum(res)/len(res))