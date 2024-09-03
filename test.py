from utils.transforms import (
    evaluate_code_2,
    transform_dialect_TP,
    transform_dialect_tile,
    transform_dialect_vectorise,
    transform_dialect_vectorise_with_backend,
    transform_dialect_vectorise_img2col,
    transform_dialect_img2col,
    transform_dialect_fuse,
    get_raw_ast_info, get_ast,
    transform_dialect_vectorise_,
    apply_conv2d_decomposition,
    
)

from utils.transform_utils import evaluate_code_with_timeout

import os

tmp_file = 'examples/temp_nn_mlir.mlir'

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
  func.func @matmul() -> tensor<256x256x14x14xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<256x256x16x16xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x256x16x16xf32>) -> tensor<256x256x16x16xf32>
    %2 = bufferization.alloc_tensor() : tensor<256x256x3x3xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %4 = bufferization.alloc_tensor() : tensor<256x256x14x14xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, tag = "operation_1"} ins(%1, %3 : tensor<256x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%5 : tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %8 = call @nanoTime() : () -> i64
    %9 = arith.subi %8, %6 : i64
    %10 = arith.uitofp %9 : i64 to f64
    call @printI64(%9) : (i64) -> ()
    call @printNewline() : () -> ()
    return %7 : tensor<256x256x14x14xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<256x256x14x14xf32>
    }
    return
  }
}


"""

conv2d_2 = """

module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<256x567x14x14xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<256x567x14x14xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x567x14x14xf32>) -> tensor<256x567x14x14xf32>
    %2 = bufferization.alloc_tensor() : tensor<567x567x1x1xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<567x567x1x1xf32>) -> tensor<567x567x1x1xf32>
    %4 = bufferization.alloc_tensor() : tensor<256x567x14x14xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<256x567x14x14xf32>) -> tensor<256x567x14x14xf32>
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, tag = "operation_1"} ins(%1, %3 : tensor<256x567x14x14xf32>, tensor<567x567x1x1xf32>) outs(%5 : tensor<256x567x14x14xf32>) -> tensor<256x567x14x14xf32>
    %8 = call @nanoTime() : () -> i64
    %9 = arith.subi %8, %6 : i64
    %10 = arith.uitofp %9 : i64 to f64
    call @printI64(%9) : (i64) -> ()
    call @printNewline() : () -> ()
    return %7 : tensor<256x567x14x14xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<256x567x14x14xf32>
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
  func.func @matmul() -> tensor<256x64x56x56xf32>{
    %val = arith.constant 2.00000e+00 : f32
    %zero = arith.constant 0.00000e+00 : f32
    %tmp_input = bufferization.alloc_tensor() : tensor<256x64x114x114xf32>
    %input = linalg.fill ins(%val : f32) outs(%tmp_input : tensor<256x64x114x114xf32>) -> tensor<256x64x114x114xf32>
    %tmp_filter = bufferization.alloc_tensor() : tensor<3x3xf32>
    %filter = linalg.fill ins(%val : f32) outs(%tmp_filter : tensor<3x3xf32>) -> tensor<3x3xf32>
    %tmp_init = bufferization.alloc_tensor() : tensor<256x64x56x56xf32>
    %init = linalg.fill ins(%val : f32) outs(%tmp_init : tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xf32>
    %t0 = func.call @nanoTime() : () -> (i64)
    %return_arg = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_1"} ins (%input, %filter: tensor<256x64x114x114xf32>, tensor<3x3xf32>) outs (%init: tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xf32>
    %t = func.call @nanoTime() : () -> (i64)
    %delta = arith.subi %t, %t0 : i64
    %fp = arith.uitofp %delta : i64 to f64
    // func.call @printFlops(%fp) : (f64) -> ()
    func.call @printI64(%delta) : (i64) -> ()
    func.call @printNewline() : () -> ()
    return %return_arg : tensor<256x64x56x56xf32> 
  }
  func.func @main(){
    %c1 = arith.constant 1: index
    %c0 = arith.constant 0 : index
    %n = arith.constant 2: index
    scf.for %i = %c0 to %n step %c1 {
    %outputmain = func.call @matmul() : () -> tensor<256x64x56x56xf32>
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
  func.func @matmul() -> XXXX{
    %val = arith.constant 2.00000e+00 : f32
    %zero = arith.constant 0.00000e+00 : f32
    %tmp_arg0 = bufferization.alloc_tensor() : XXXX
    %arg0 = linalg.fill ins(%val : f32) outs(%tmp_arg0 : XXXX) -> XXXX
    %tmp_arg1 = bufferization.alloc_tensor() : XXXX
    %arg1 = linalg.fill ins(%val : f32) outs(%tmp_arg1 : XXXX) -> XXXX
    %tmp_arg2 = bufferization.alloc_tensor() : XXXX
    %arg2 = linalg.fill ins(%val : f32) outs(%tmp_arg2 : XXXX) -> XXXX
    %t0 = func.call @nanoTime() : () -> (i64)
    %return_arg = linalg.add {tag = "operation_1"} ins(%arg0, %arg1: XXXX, XXXX) outs(%arg2: XXXX) -> XXXX
    %t = func.call @nanoTime() : () -> (i64)
    %delta = arith.subi %t, %t0 : i64
    %fp = arith.uitofp %delta : i64 to f64
    // func.call @printFlops(%fp) : (f64) -> ()
    func.call @printI64(%delta) : (i64) -> ()
    func.call @printNewline() : () -> ()
    return %return_arg : XXXX 
  }
  func.func @main(){
    %c1 = arith.constant 1: index
    %c0 = arith.constant 0 : index
    %n = arith.constant 2: index
    scf.for %i = %c0 to %n step %c1 {
    %outputmain = func.call @matmul() : () -> XXXX
    }
    return
  }
}

""".replace('XXXX', 'tensor<256x7x7x176xf32>')

nassim = """

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0) -> (d0 floordiv 14)>
#map2 = affine_map<(d0) -> (d0 mod 14)>
#map3 = affine_map<(d0) -> (d0 floordiv 768)>
#map4 = affine_map<(d0) -> (d0 mod 768)>
#map5 = affine_map<(d0) -> ((d0 mod 768) floordiv 256)>
#map6 = affine_map<(d0) -> (d0 mod 256)>
#map7 = affine_map<(d0, d1) -> (d0 floordiv 14 + d1 floordiv 768)>
#map8 = affine_map<(d0, d1) -> (d0 mod 14 + (d1 mod 768) floordiv 256)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func @conv() -> tensor<256x14x14x256xf32> {
    %0 = call @nanoTime() : () -> i64
    %cst = arith.constant 2.000000e+00 : f32
    %1 = bufferization.alloc_tensor() : tensor<256x16x16x256xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<256x16x16x256xf32>) -> tensor<256x16x16x256xf32>
    %3 = bufferization.alloc_tensor() : tensor<3x3x256x256xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<3x3x256x256xf32>) -> tensor<3x3x256x256xf32>
    %5 = bufferization.alloc_tensor() : tensor<256x14x14x256xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x14x14x256xf32>) -> tensor<256x14x14x256xf32>
    %collapsed = tensor.collapse_shape %4 [[0, 1, 2], [3]] : tensor<3x3x256x256xf32> into tensor<2304x256xf32>
    %collapsed_0 = tensor.collapse_shape %6 [[0], [1, 2], [3]] : tensor<256x14x14x256xf32> into tensor<256x196x256xf32>
    %7 = tensor.empty() : tensor<256x196x2304xf32>
    %8 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], tag = "img2col_producer"} outs(%7 : tensor<256x196x2304xf32>) {
    ^bb0(%out: f32):
      %13 = linalg.index 0 : index
      %14 = linalg.index 1 : index
      %15 = linalg.index 2 : index
      %c14 = arith.constant 14 : index
      %c14_1 = arith.constant 14 : index
      %16 = affine.apply #map1(%14)
      %17 = affine.apply #map2(%14)
      %c3 = arith.constant 3 : index
      %c3_2 = arith.constant 3 : index
      %c256 = arith.constant 256 : index
      %c768 = arith.constant 768 : index
      %18 = affine.apply #map3(%15)
      %19 = affine.apply #map4(%15)
      %20 = affine.apply #map5(%15)
      %21 = affine.apply #map6(%15)
      %22 = affine.apply #map7(%14, %15)
      %23 = affine.apply #map8(%14, %15)
      %extracted = tensor.extract %2[%13, %22, %23, %21] : tensor<256x16x16x256xf32>
      linalg.yield %extracted : f32
    } -> tensor<256x196x2304xf32>
    %9 = linalg.generic {indexing_maps = [#map9, #map10, #map11], iterator_types = ["parallel", "parallel", "parallel", "reduction"], tag = "operation_1"} ins(%8, %collapsed : tensor<256x196x2304xf32>, tensor<2304x256xf32>) outs(%collapsed_0 : tensor<256x196x256xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %13 = arith.mulf %in, %in_1 : f32
      %14 = arith.addf %13, %out : f32
      linalg.yield %14 : f32
    } -> tensor<256x196x256xf32>
    %expanded = tensor.expand_shape %9 [[0], [1, 2], [3]] : tensor<256x196x256xf32> into tensor<256x14x14x256xf32>
    %10 = call @nanoTime() : () -> i64
    %11 = arith.subi %10, %0 : i64
    %12 = arith.uitofp %11 : i64 to f64
    // call @printFlops(%12) : (f64) -> ()
    call @printI64(%11) : (i64) -> ()
    func.call @printNewline() : () -> ()
    return %expanded : tensor<256x14x14x256xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @conv() : () -> tensor<256x14x14x256xf32>
    }
    return
  }
}

""".strip()

nassim2 = """

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0) -> (d0 * 16)>
#map4 = affine_map<(d0) -> (d0 * 4)>
#map5 = affine_map<(d0) -> (d0 * 14)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func @conv() -> tensor<256x14x14x256xf32> {
    %cst = arith.constant dense<2.000000e+00> : vector<4x14x16xf32>
    %cst_0 = arith.constant dense<2.000000e+00> : vector<2304x16xf32>
    %cst_1 = arith.constant dense<2.000000e+00> : vector<4x14x2304xf32>
    %c0 = arith.constant 0 : index
    %0 = call @nanoTime() : () -> i64
    %1 = bufferization.alloc_tensor() : tensor<256x16x16x256xf32>
    %2 = bufferization.alloc_tensor() : tensor<3x3x256x256xf32>
    %3 = bufferization.alloc_tensor() : tensor<256x14x14x256xf32>
    %collapsed = tensor.collapse_shape %3 [[0], [1, 2], [3]] : tensor<256x14x14x256xf32> into tensor<256x196x256xf32>
    %4 = bufferization.alloc_tensor() : tensor<256x196x2304xf32>
    %5 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %cst_1, %cst_0, %cst : vector<4x14x2304xf32>, vector<2304x16xf32> into vector<4x14x16xf32>
    %6 = scf.forall (%arg0, %arg1, %arg2) in (64, 14, 16) shared_outs(%arg3 = %collapsed) -> (tensor<256x196x256xf32>) {
      %10 = affine.apply #map3(%arg2)
      %11 = affine.apply #map4(%arg0)
      %12 = affine.apply #map5(%arg1)
      %extracted_slice = tensor.extract_slice %arg3[%11, %12, %10] [4, 14, 16] [1, 1, 1] : tensor<256x196x256xf32> to tensor<4x14x16xf32>
      %13 = vector.transfer_write %5, %extracted_slice[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<4x14x16xf32>, tensor<4x14x16xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %13 into %arg3[%11, %12, %10] [4, 14, 16] [1, 1, 1] : tensor<4x14x16xf32> into tensor<256x196x256xf32>
      }
    } {consumer4}
    %expanded = tensor.expand_shape %6 [[0], [1, 2], [3]] : tensor<256x196x256xf32> into tensor<256x14x14x256xf32>
    %7 = call @nanoTime() : () -> i64
    %8 = arith.subi %7, %0 : i64
    %9 = arith.uitofp %8 : i64 to f64
    // call @printFlops(%9) : (f64) -> ()
    call @printI64(%8) : (i64) -> ()
    func.call @printNewline() : () -> ()
    return %expanded : tensor<256x14x14x256xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @conv() : () -> tensor<256x14x14x256xf32>
    }
    return
  }
}

""".strip()

relu = """

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>

module attributes {torch.debug_module_name = "Net"} {
func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
func.func private @printFlops(f64)
func.func private @printI64(i64)
func.func private @printNewline()
func.func private @printMemrefF32(tensor<*xf32>)


func.func @matmul() -> tensor<256x74x74x64xf32>{

%val = arith.constant 2.00000e+00 : f32
%zero = arith.constant 0.00000e+00 : f32

%tmp_28 = bufferization.alloc_tensor() : tensor<256x74x74x64xf32>
%28 = linalg.fill ins(%val : f32) outs(%tmp_28 : tensor<256x74x74x64xf32>) -> tensor<256x74x74x64xf32>
%tmp_25 = bufferization.alloc_tensor() : tensor<256x74x74x64xf32>
%25 = linalg.fill ins(%val : f32) outs(%tmp_25 : tensor<256x74x74x64xf32>) -> tensor<256x74x74x64xf32>

%t0 = func.call @nanoTime() : () -> (i64)

%return_arg = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], tag = "operation_1"} ins(%28 : tensor<256x74x74x64xf32>) outs(%25 : tensor<256x74x74x64xf32>) {
                    ^bb0(%in: f32, %out: f32):
                    %cst_1 = arith.constant 0.000000e+00 : f32
                    %90 = arith.cmpf ugt, %in, %cst_1 : f32
                    %91 = arith.select %90, %in, %cst_1 : f32
                    linalg.yield %91 : f32
                } -> tensor<256x74x74x64xf32>
%t = func.call @nanoTime() : () -> (i64)
%delta = arith.subi %t, %t0 : i64
%fp = arith.uitofp %delta : i64 to f64
// func.call @printFlops(%fp) : (f64) -> ()
func.call @printI64(%delta) : (i64) -> ()
func.call @printNewline() : () -> ()

return %return_arg : tensor<256x74x74x64xf32> 
}

func.func @main(){
    %c1 = arith.constant 1: index
    %c0 = arith.constant 0 : index
    %n = arith.constant 2: index
    scf.for %i = %c0 to %n step %c1 {
      %outputmain = func.call @matmul() : () -> tensor<256x74x74x64xf32>
    }
    return
  }
}
""".strip()

# code = nassim.strip()
# code = nassim2.strip()
# code = conv2d.strip()
# code = conv2d_2.strip()
# code = matmul.strip()
code = pool.strip()
# code = add.strip()
# code = relu.strip()

def get_conv(input_shape, kernel, stride):
  
    N, H, W, C = input_shape
    KH, KW, C, F = kernel

    dilation = 1
    padding = 0

    W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1
    H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1
  
  
    conv2d = """
    module attributes {torch.debug_module_name = "Net"} {
      func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
      func.func private @printFlops(f64)
      func.func private @printI64(i64)
      func.func private @printNewline()
      func.func private @printMemrefF32(tensor<*xf32>)
      func.func @matmul() -> tensor<BATCHSIZExFILTERxNEWHxNEWWxf32> {
        %cst = arith.constant 2.000000e+00 : f32
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = bufferization.alloc_tensor() : tensor<BATCHSIZExCHANNELxWIDTHxHEIGHTxf32>
        %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<BATCHSIZExCHANNELxWIDTHxHEIGHTxf32>) -> tensor<BATCHSIZExCHANNELxWIDTHxHEIGHTxf32>
        %2 = bufferization.alloc_tensor() : tensor<FILTERxCHANNELxKERNELxKERNELxf32>
        %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<FILTERxCHANNELxKERNELxKERNELxf32>) -> tensor<FILTERxCHANNELxKERNELxKERNELxf32>
        %4 = bufferization.alloc_tensor() : tensor<BATCHSIZExFILTERxNEWHxNEWWxf32>
        %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<BATCHSIZExFILTERxNEWHxNEWWxf32>) -> tensor<BATCHSIZExFILTERxNEWHxNEWWxf32>
        %6 = call @nanoTime() : () -> i64
        %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<STRIDE> : tensor<2xi64>, tag = "operation_1"} ins(%1, %3 : tensor<BATCHSIZExCHANNELxWIDTHxHEIGHTxf32>, tensor<FILTERxCHANNELxKERNELxKERNELxf32>) outs(%5 : tensor<BATCHSIZExFILTERxNEWHxNEWWxf32>) -> tensor<BATCHSIZExFILTERxNEWHxNEWWxf32>
        %8 = call @nanoTime() : () -> i64
        %9 = arith.subi %8, %6 : i64
        %10 = arith.uitofp %9 : i64 to f64
        call @printI64(%9) : (i64) -> ()
        call @printNewline() : () -> ()
        return %7 : tensor<BATCHSIZExFILTERxNEWHxNEWWxf32>
      }
      func.func @main() {
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        scf.for %arg0 = %c0 to %c2 step %c1 {
          %0 = func.call @matmul() : () -> tensor<BATCHSIZExFILTERxNEWHxNEWWxf32>
        }
        return
      }
    }
    """
    
    conv2d = conv2d.replace('BATCHSIZE', str(N))
    conv2d = conv2d.replace('CHANNEL', str(C))
    conv2d = conv2d.replace('WIDTH', str(W))
    conv2d = conv2d.replace('HEIGHT', str(H))
    conv2d = conv2d.replace('KERNEL', str(KH))
    conv2d = conv2d.replace('FILTER', str(F))
    conv2d = conv2d.replace('NEWH', str(H_))
    conv2d = conv2d.replace('NEWW', str(W_))
    conv2d = conv2d.replace('STRIDE', str(stride))
    
    return conv2d



def get_pool(input_shape, kernel):
    
    N, H, W, C = input_shape
    K, K = kernel

    dilation = 1
    stride = 2

    H_ = (H - dilation * (K - 1) - 1) // stride + 1
    W_ = (W - dilation * (K - 1) - 1) // stride + 1

    pool = """
    module attributes {torch.debug_module_name = "Net"} {
      func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
      func.func private @printFlops(f64)
      func.func private @printI64(i64)
      func.func private @printNewline()
      func.func private @printMemrefF32(tensor<*xf32>)
      func.func @matmul() -> tensor<BATCHSIZExCHANNELxNEWHxNEWWxf32>{
        %val = arith.constant 2.00000e+00 : f32
        %zero = arith.constant 0.00000e+00 : f32
        %tmp_input = bufferization.alloc_tensor() : tensor<BATCHSIZExCHANNELxHEIGHTxWIDTHxf32>
        %input = linalg.fill ins(%val : f32) outs(%tmp_input : tensor<BATCHSIZExCHANNELxHEIGHTxWIDTHxf32>) -> tensor<BATCHSIZExCHANNELxHEIGHTxWIDTHxf32>
        %tmp_filter = bufferization.alloc_tensor() : tensor<KERNELxKERNELxf32>
        %filter = linalg.fill ins(%val : f32) outs(%tmp_filter : tensor<KERNELxKERNELxf32>) -> tensor<KERNELxKERNELxf32>
        %tmp_init = bufferization.alloc_tensor() : tensor<BATCHSIZExCHANNELxNEWHxNEWWxf32>
        %init = linalg.fill ins(%val : f32) outs(%tmp_init : tensor<BATCHSIZExCHANNELxNEWHxNEWWxf32>) -> tensor<BATCHSIZExCHANNELxNEWHxNEWWxf32>
        %t0 = func.call @nanoTime() : () -> (i64)
        %return_arg = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_1"} ins (%input, %filter: tensor<BATCHSIZExCHANNELxHEIGHTxWIDTHxf32>, tensor<KERNELxKERNELxf32>) outs (%init: tensor<BATCHSIZExCHANNELxNEWHxNEWWxf32>) -> tensor<BATCHSIZExCHANNELxNEWHxNEWWxf32>
        %t = func.call @nanoTime() : () -> (i64)
        %delta = arith.subi %t, %t0 : i64
        %fp = arith.uitofp %delta : i64 to f64
        // func.call @printFlops(%fp) : (f64) -> ()
        func.call @printI64(%delta) : (i64) -> ()
        func.call @printNewline() : () -> ()
        return %return_arg : tensor<BATCHSIZExCHANNELxNEWHxNEWWxf32> 
      }
      func.func @main(){
        %c1 = arith.constant 1: index
        %c0 = arith.constant 0 : index
        %n = arith.constant 2: index
        scf.for %i = %c0 to %n step %c1 {
        %outputmain = func.call @matmul() : () -> tensor<BATCHSIZExCHANNELxNEWHxNEWWxf32>
        }
        return
      }
    }
    """.strip()
    
    pool = pool.replace('BATCHSIZE', str(N))
    pool = pool.replace('CHANNEL', str(C))
    pool = pool.replace('WIDTH', str(W))
    pool = pool.replace('HEIGHT', str(H))
    pool = pool.replace('KERNEL', str(K))
    pool = pool.replace('NEWH', str(H_))
    pool = pool.replace('NEWW', str(W_))
    
    return pool
    
    
    


# code = get_conv((256, 28, 28, 128), (3, 3, 128, 128), 1)

pools = [
    [  ((256, 114, 114, 64), (3, 3)), [2, 4, 8]  ],
    [  ((256, 147, 147, 64), (3, 3)), [2, 2]  ],
    [  ((256, 71, 71, 192), (3, 3)), [2, 4, 7, 7]  ],
    [  ((256, 167, 167, 42), (3, 3)), [4, 6]  ],
    [  ((256, 85, 85, 84), (3, 3)), [4, 3, 14]  ],
    [  ((256, 43, 43, 336), (3, 3)), [2, 7, 3, 7]  ],
    [  ((256, 23, 23, 672), (3, 3)), [2, 14, 11, 11]  ],
    [  ((256, 113, 113, 11), (3, 3)), [2, 11, 4, 7]  ],
    [  ((256, 57, 57, 22), (3, 3)), [16, 11, 2, 14]  ],
    [  ((256, 29, 29, 88), (3, 3)), [2, 4, 7]  ]
]

convs = [
    [  ((256, 14, 14, 256), (3, 3, 256, 256), 1) , [2, 0, 4]],
    [  ((256, 14, 14, 256), (1, 1, 256, 1024), 1) , [64, 4, 7]],
    [  ((256, 28, 28, 128), (3, 3, 128, 128), 1) , [32, 16, 13]],
    [  ((256, 28, 28, 128), (1, 1, 128, 512), 1) , [4, 16, 14]],
    [  ((256, 28, 28, 512), (1, 1, 512, 128), 1) , [2, 16, 16] ],
    [  ((256, 14, 14, 128), (3, 3, 128, 32), 1) , [2, 16, 36]],
    [  ((256, 7, 7, 128), (3, 3, 128, 32), 1) , [4, 32, 25]],
    [  ((256, 16, 16, 256), (3, 3, 256, 256), 1) , [4, 16, 14]],
    [  ((256, 14, 14, 576), (1, 1, 576, 576), 1) , [2, 36, 28]],
    [  ((256, 28, 28, 128), (3, 3, 128, 32), 1) , [8, 16, 4]],
    [  ((256, 14, 14, 336), (1, 1, 336, 336), 1) , [8, 3, 14]],
    [  ((256, 56, 56, 64), (3, 3, 64, 64), 1) , [2, 16, 36]],
    [  ((256, 28, 28, 448), (1, 1, 448, 448), 1) , [4, 32, 16]],
    [  ((256, 56, 56, 64), (1, 1, 64, 256), 1) , [2, 16, 32, 64]],
    [  ((256, 128, 128, 16), (7, 7, 16, 8), 2) , [32, 8, 61]],
    [  ((256, 64, 64, 64), (3, 3, 64, 16), 1) , [2, 0, 31]],
    [  ((256, 32, 32, 32), (7, 7, 32, 256), 2) , [4, 64, 13]],
    [  ((256, 230, 230, 3), (7, 7, 3, 64), 2) , [2, 16, 49]],
]


# for (input_shape, kernel), params in pools:

        
# code = get_conv(input_shape, kernel, stride)

relus = [
  (256, 57, 57, 64),
  (256, 74, 74, 64),
  (256, 36, 36, 192),
  (256, 85, 85, 42),
  (256, 43, 43, 84),
  (256, 23, 23, 336),
  (256, 14, 14, 672),
  (256, 29, 29, 22),
  (256, 14, 14, 88),
]


code = relu.strip()
# code = code.replace('tensor<256x74x74x64xf32>', f'tensor<256x{H}x{H}x{C}xf32>')

code = pool.strip()


# code = transform_dialect_img2col(code, "operation_1", tmp_file)
# code = transform_dialect_TP  (code, "operation_1", [32, 2], tmp_file)
code = transform_dialect_TP(code, "operation_1", [2], tmp_file)
# code = transform_dialect_vectorise(code, "operation_1", tmp_file)

res = []
get_out = False
for _ in range(5):
    exec = evaluate_code_with_timeout(code, 6000, tmp_file)
    if exec == None:
        get_out = True
        break
    # print(exec*1e-9)
    res.append(exec*1e-9)
    
print(sum(res)/len(res))
  
  # print('\n\n\n')