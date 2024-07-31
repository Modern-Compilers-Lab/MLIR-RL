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


conv2d = """

module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)


  func.func @matmul() -> tensor<32x112x67x64xf32>{
    
  %val = arith.constant 2.00000e+00 : f32
  %zero = arith.constant 0.00000e+00 : f32

  %tmp_input = bufferization.alloc_tensor() : tensor<32x3x230x140xf32>
  %input = linalg.fill ins(%val : f32) outs(%tmp_input : tensor<32x3x230x140xf32>) -> tensor<32x3x230x140xf32>
  %tmp_filter = bufferization.alloc_tensor() : tensor<64x3x7x7xf32>
  %filter = linalg.fill ins(%val : f32) outs(%tmp_filter : tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf32>
  %tmp_init = bufferization.alloc_tensor() : tensor<32x112x67x64xf32>
  %init = linalg.fill ins(%val : f32) outs(%tmp_init : tensor<32x112x67x64xf32>) -> tensor<32x112x67x64xf32>

  %t0 = func.call @nanoTime() : () -> (i64)

  %return_arg = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_1"} ins (%input, %filter: tensor<32x3x230x140xf32>, tensor<64x3x7x7xf32>) outs (%init: tensor<32x112x67x64xf32>) -> tensor<32x112x67x64xf32>
  
  %t = func.call @nanoTime() : () -> (i64)
  %delta = arith.subi %t, %t0 : i64
  %fp = arith.uitofp %delta : i64 to f64
  // func.call @printFlops(%fp) : (f64) -> ()
  func.call @printI64(%delta) : (i64) -> ()
  func.call @printNewline() : () -> ()

  return %return_arg : tensor<32x112x67x64xf32> 
  }

  func.func @main(){
      %c1 = arith.constant 1: index
      %c0 = arith.constant 0 : index
      %n = arith.constant 2: index
      scf.for %i = %c0 to %n step %c1 {
        %outputmain = func.call @matmul() : () -> tensor<32x112x67x64xf32>
      }
      return
  }
}


"""

conv2d = """

module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)


  func.func @matmul() -> tensor<32x112x67x64xf32>{
    
  %val = arith.constant 2.00000e+00 : f32
  %zero = arith.constant 0.00000e+00 : f32

  %tmp_input = bufferization.alloc_tensor() : tensor<32x230x140x3xf32>
  %input = linalg.fill ins(%val : f32) outs(%tmp_input : tensor<32x230x140x3xf32>) -> tensor<32x230x140x3xf32>
  %tmp_filter = bufferization.alloc_tensor() : tensor<7x7x3x64xf32>
  %filter = linalg.fill ins(%val : f32) outs(%tmp_filter : tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
  %tmp_init = bufferization.alloc_tensor() : tensor<32x112x67x64xf32>
  %init = linalg.fill ins(%val : f32) outs(%tmp_init : tensor<32x112x67x64xf32>) -> tensor<32x112x67x64xf32>

  %t0 = func.call @nanoTime() : () -> (i64)

  %return_arg = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_1"} ins (%input, %filter: tensor<32x230x140x3xf32>, tensor<7x7x3x64xf32>) outs (%init: tensor<32x112x67x64xf32>) -> tensor<32x112x67x64xf32>
  
  %t = func.call @nanoTime() : () -> (i64)
  %delta = arith.subi %t, %t0 : i64
  %fp = arith.uitofp %delta : i64 to f64
  // func.call @printFlops(%fp) : (f64) -> ()
  func.call @printI64(%delta) : (i64) -> ()
  func.call @printNewline() : () -> ()

  return %return_arg : tensor<32x112x67x64xf32> 
  }

  func.func @main(){
      %c1 = arith.constant 1: index
      %c0 = arith.constant 0 : index
      %n = arith.constant 2: index
      scf.for %i = %c0 to %n step %c1 {
        %outputmain = func.call @matmul() : () -> tensor<32x112x67x64xf32>
      }
      return
  }
}


"""

matmul = """

module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
func.func private @printFlops(f64)
func.func private @printI64(i64)
func.func private @printNewline()
func.func private @printMemrefF32(tensor<*xf32>)


func.func @matmul() -> tensor<1200x1000xf32>{
  
%val = arith.constant 2.00000e+00 : f32
%zero = arith.constant 0.00000e+00 : f32

%tmp_arg0 = bufferization.alloc_tensor() : tensor<1200x1500xf32>
%arg0 = linalg.fill ins(%val : f32) outs(%tmp_arg0 : tensor<1200x1500xf32>) -> tensor<1200x1500xf32>
%tmp_arg1 = bufferization.alloc_tensor() : tensor<1500x1000xf32>
%arg1 = linalg.fill ins(%val : f32) outs(%tmp_arg1 : tensor<1500x1000xf32>) -> tensor<1500x1000xf32>
%tmp_arg2 = bufferization.alloc_tensor() : tensor<1200x1000xf32>
%arg2 = linalg.fill ins(%val : f32) outs(%tmp_arg2 : tensor<1200x1000xf32>) -> tensor<1200x1000xf32>

%t0 = func.call @nanoTime() : () -> (i64)

%return_arg = linalg.matmul {tag = "operation_1"} ins(%arg0, %arg1 : tensor<1200x1500xf32>, tensor<1500x1000xf32>) outs(%arg2 : tensor<1200x1000xf32>) -> tensor<1200x1000xf32>
%t = func.call @nanoTime() : () -> (i64)
%delta = arith.subi %t, %t0 : i64
%fp = arith.uitofp %delta : i64 to f64
// func.call @printFlops(%fp) : (f64) -> ()
func.call @printI64(%delta) : (i64) -> ()
func.call @printNewline() : () -> ()

return %return_arg : tensor<1200x1000xf32> 
}

func.func @main(){
      %c1 = arith.constant 1: index
    %c0 = arith.constant 0 : index
    %n = arith.constant 2: index
    scf.for %i = %c0 to %n step %c1 {
      %outputmain = func.call @matmul() : () -> tensor<1200x1000xf32>
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
    %0 = bufferization.alloc_tensor() : tensor<4x64x128x64xf32>
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

code = conv2d.strip()
# code = matmul.strip()


# code = transform_dialect_TP(code, "operation_1", [12, 8])

# code = transform_dialect_TP(code, "operation_1", [2, 4, 0, 0, 0, 0, 0])
# code = transform_dialect_tile(code, "operation_1", [8, 4, 7, 28, 3, 7, 0])
# code = transform_dialect_tile(code, "operation_1", [8, 4, 1, 28, 3, 1, 0])
# code = transform_dialect_tile(code, "operation_1", [0, 0, 1, 0, 0, 1, 0])
# code = apply_conv2d_decomposition(code, "operation_1")

code = transform_dialect_img2col(code, "operation_3")

print(code)

exit()

code = transform_dialect_TP(code, "operation_1", [ 4, 8, 32])
# code = transform_dialect_tile(code, "operation_1", [4, 8, 32])

# code = transform_dialect_vectorise_img2col(code, "operation_1")



res = []
for _ in range(30):
  exec = evaluate_code_with_timeout(code, 600)
  print(exec*1e-9)
  res.append(exec)

print(sum(res)/len(res))








exit()

code = """

module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<32x27x27x128xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<32x56x56x3xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst : f32) outs(%0 : tensor<32x56x56x3xf32>) -> tensor<32x56x56x3xf32>
    %2 = bufferization.alloc_tensor() : tensor<3x3x3x128xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst : f32) outs(%2 : tensor<3x3x3x128xf32>) -> tensor<3x3x3x128xf32>
    %4 = bufferization.alloc_tensor() : tensor<32x27x27x128xf32>
    %5 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%4 : tensor<32x27x27x128xf32>) -> tensor<32x27x27x128xf32>
    
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_3"} ins(%1, %3 : tensor<32x56x56x3xf32>, tensor<3x3x3x128xf32>) outs(%5 : tensor<32x27x27x128xf32>) -> tensor<32x27x27x128xf32>
    %8 = call @nanoTime() : () -> i64
    %9 = arith.subi %8, %6 : i64
    %10 = arith.uitofp %9 : i64 to f64
    call @printI64(%9) : (i64) -> ()
    call @printNewline() : () -> ()
    return %7 : tensor<32x27x27x128xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<32x27x27x128xf32>
    }
    return
  }
}


""".strip()

# code = transform_dialect_img2col(code, "operation_3")


code = """

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0) -> (d0 floordiv 27)>
#map2 = affine_map<(d0) -> (d0 mod 27)>
#map3 = affine_map<(d0) -> (d0 floordiv 9)>
#map4 = affine_map<(d0) -> (d0 mod 9)>
#map5 = affine_map<(d0) -> ((d0 mod 9) floordiv 3)>
#map6 = affine_map<(d0) -> (d0 mod 3)>
#map7 = affine_map<(d0, d1) -> ((d0 floordiv 27) * 2 + d1 floordiv 9)>
#map8 = affine_map<(d0, d1) -> (d0 * 2 - (d0 floordiv 27) * 54 + (d1 mod 9) floordiv 3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
  module attributes {torch.debug_module_name = "Net"} {
    func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
    func.func private @printFlops(f64)
    func.func private @printI64(i64)
    func.func private @printNewline()
    func.func private @printMemrefF32(tensor<*xf32>)
    func.func @matmul() -> tensor<32x27x27x128xf32> {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.000000e+00 : f32
      %0 = bufferization.alloc_tensor() : tensor<32x56x56x3xf32>
      %1 = linalg.fill {tag = "operation_0"} ins(%cst : f32) outs(%0 : tensor<32x56x56x3xf32>) -> tensor<32x56x56x3xf32>
      %2 = bufferization.alloc_tensor() : tensor<3x3x3x128xf32>
      %3 = linalg.fill {tag = "operation_1"} ins(%cst : f32) outs(%2 : tensor<3x3x3x128xf32>) -> tensor<3x3x3x128xf32>
      %4 = bufferization.alloc_tensor() : tensor<32x27x27x128xf32>
      %5 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%4 : tensor<32x27x27x128xf32>) -> tensor<32x27x27x128xf32>
      %6 = call @nanoTime() : () -> i64
      %collapsed = tensor.collapse_shape %3 [[0, 1, 2], [3]] : tensor<3x3x3x128xf32> into tensor<27x128xf32>
      %collapsed_1 = tensor.collapse_shape %5 [[0], [1, 2], [3]] : tensor<32x27x27x128xf32> into tensor<32x729x128xf32>
      %7 = tensor.empty() : tensor<32x729x27xf32>
      %8 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%7 : tensor<32x729x27xf32>) attrs =  {tag = "operation_4"} {
      ^bb0(%out: f32):
        %13 = linalg.index 0 : index
        %14 = linalg.index 1 : index
        %15 = linalg.index 2 : index
        %c27 = arith.constant 27 : index
        %c27_2 = arith.constant 27 : index
        %16 = affine.apply #map1(%14)
        %17 = affine.apply #map2(%14)
        %c3 = arith.constant 3 : index
        %c3_3 = arith.constant 3 : index
        %c3_4 = arith.constant 3 : index
        %c9 = arith.constant 9 : index
        %18 = affine.apply #map3(%15)
        %19 = affine.apply #map4(%15)
        %20 = affine.apply #map5(%15)
        %21 = affine.apply #map6(%15)
        %22 = affine.apply #map7(%14, %15)
        %23 = affine.apply #map8(%14, %15)
        %extracted = tensor.extract %1[%13, %22, %23, %21] : tensor<32x56x56x3xf32>
        linalg.yield %extracted : f32
      } -> tensor<32x729x27xf32>
      %9 = linalg.generic {indexing_maps = [#map9, #map10, #map11], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%8, %collapsed : tensor<32x729x27xf32>, tensor<27x128xf32>) outs(%collapsed_1 : tensor<32x729x128xf32>) attrs =  {tag = "operation_3"} {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %13 = arith.mulf %in, %in_2 : f32
        %14 = arith.addf %13, %out : f32
        linalg.yield %14 : f32
      } -> tensor<32x729x128xf32>
      %expanded = tensor.expand_shape %9 [[0], [1, 2], [3]] : tensor<32x729x128xf32> into tensor<32x27x27x128xf32>
      %10 = call @nanoTime() : () -> i64
      %11 = arith.subi %10, %6 : i64
      %12 = arith.uitofp %11 : i64 to f64
      call @printI64(%11) : (i64) -> ()
      call @printNewline() : () -> ()
      return %expanded : tensor<32x27x27x128xf32>
    }
    func.func @main() {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %0 = func.call @matmul() : () -> tensor<32x27x27x128xf32>
      }
      return
    }
  }


""".strip()


code = transform_dialect_TP(code, "operation_3", [2,27])

code = transform_dialect_fuse(code, 'operation_3', 'operation_4')
code = transform_dialect_fuse(code, 'operation_3', 'operation_2')
# code = transform_dialect_fuse(code, 'operation_3', 'operation_1')
# code = transform_dialect_fuse(code, 'operation_3', 'operation_0')

# code = transform_dialect_vectorise(code, "operation_3")

print(code)

exec_time = evaluate_code_with_timeout(code, 600)




print(exec_time*1e-9)