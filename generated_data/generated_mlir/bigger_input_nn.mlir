#map = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1, d0)>
#map4 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "Net"} {
  memref.global "private" @global_seed : memref<i64> = dense<0>
  
  func.func @forward() -> tensor<32x10xf32> {
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = bufferization.alloc_tensor() : tensor<32x3x230x230xf32>
    %cst_2 = bufferization.alloc_tensor() : tensor<64x3x7x7xf32>
    %cst_3 = bufferization.alloc_tensor() : tensor<64xf32>
    %cst_4 = bufferization.alloc_tensor() : tensor<16x64x5x5xf32>
    %cst_5 = bufferization.alloc_tensor() : tensor<16xf32>
    %cst_6 = bufferization.alloc_tensor() : tensor<120x10816xf32>
    %cst_7 = bufferization.alloc_tensor() : tensor<120xf32>
    %cst_8 = bufferization.alloc_tensor() : tensor<84x120xf32>
    %cst_9 = bufferization.alloc_tensor() : tensor<84xf32>
    %cst_10 = bufferization.alloc_tensor() : tensor<10x84xf32>
    %cst_11 = bufferization.alloc_tensor() : tensor<10xf32>


    %0 = tensor.empty() : tensor<32x64x112x112xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3 : tensor<64xf32>) outs(%0 : tensor<32x64x112x112xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x64x112x112xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%cst_1, %cst_2 : tensor<32x3x230x230xf32>, tensor<64x3x7x7xf32>) outs(%1 : tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<32x64x112x112xf32>) outs(%0 : tensor<32x64x112x112xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_00 = arith.constant 0.000000e+00 : f32
      %35 = arith.cmpf ugt, %in, %cst_00 : f32
      %36 = arith.select %35, %in, %cst_00 : f32
      linalg.yield %36 : f32
    } -> tensor<32x64x112x112xf32>
    %4 = tensor.empty() : tensor<32x64x56x56xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf32>
    %6 = tensor.empty() : tensor<2x2xf32>
    %7 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%3, %6 : tensor<32x64x112x112xf32>, tensor<2x2xf32>) outs(%5 : tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf32>
    %8 = tensor.empty() : tensor<32x16x52x52xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_5 : tensor<16xf32>) outs(%8 : tensor<32x16x52x52xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32x16x52x52xf32>
    %10 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%7, %cst_4 : tensor<32x64x56x56xf32>, tensor<16x64x5x5xf32>) outs(%9 : tensor<32x16x52x52xf32>) -> tensor<32x16x52x52xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10 : tensor<32x16x52x52xf32>) outs(%8 : tensor<32x16x52x52xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_00 = arith.constant 0.000000e+00 : f32
      %35 = arith.cmpf ugt, %in, %cst_00 : f32
      %36 = arith.select %35, %in, %cst_00 : f32
      linalg.yield %36 : f32
    } -> tensor<32x16x52x52xf32>
    %12 = tensor.empty() : tensor<32x16x26x26xf32>
    %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<32x16x26x26xf32>) -> tensor<32x16x26x26xf32>
    %14 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%11, %6 : tensor<32x16x52x52xf32>, tensor<2x2xf32>) outs(%13 : tensor<32x16x26x26xf32>) -> tensor<32x16x26x26xf32>
    %collapsed = tensor.collapse_shape %14 [[0], [1, 2, 3]] : tensor<32x16x26x26xf32> into tensor<32x10816xf32>
    %15 = tensor.empty() : tensor<10816x120xf32>
    %16 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%cst_6 : tensor<120x10816xf32>) outs(%15 : tensor<10816x120xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<10816x120xf32>
    %17 = tensor.empty() : tensor<32x120xf32>
    %18 = linalg.fill ins(%cst_0 : f32) outs(%17 : tensor<32x120xf32>) -> tensor<32x120xf32>
    %19 = linalg.matmul ins(%collapsed, %16 : tensor<32x10816xf32>, tensor<10816x120xf32>) outs(%18 : tensor<32x120xf32>) -> tensor<32x120xf32>
    %20 = linalg.generic {indexing_maps = [#map2, #map4, #map2], iterator_types = ["parallel", "parallel"]} ins(%19, %cst_7 : tensor<32x120xf32>, tensor<120xf32>) outs(%17 : tensor<32x120xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %35 = arith.addf %in, %in_12 : f32
      linalg.yield %35 : f32
    } -> tensor<32x120xf32>
    %21 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%20 : tensor<32x120xf32>) outs(%17 : tensor<32x120xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_00 = arith.constant 0.000000e+00 : f32
      %35 = arith.cmpf ugt, %in, %cst_00 : f32
      %36 = arith.select %35, %in, %cst_00 : f32
      linalg.yield %36 : f32
    } -> tensor<32x120xf32>
    %22 = tensor.empty() : tensor<120x84xf32>
    %23 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%cst_8 : tensor<84x120xf32>) outs(%22 : tensor<120x84xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<120x84xf32>
    %24 = tensor.empty() : tensor<32x84xf32>
    %25 = linalg.fill ins(%cst_0 : f32) outs(%24 : tensor<32x84xf32>) -> tensor<32x84xf32>
    %26 = linalg.matmul ins(%21, %23 : tensor<32x120xf32>, tensor<120x84xf32>) outs(%25 : tensor<32x84xf32>) -> tensor<32x84xf32>
    %27 = linalg.generic {indexing_maps = [#map2, #map4, #map2], iterator_types = ["parallel", "parallel"]} ins(%26, %cst_9 : tensor<32x84xf32>, tensor<84xf32>) outs(%24 : tensor<32x84xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %35 = arith.addf %in, %in_12 : f32
      linalg.yield %35 : f32
    } -> tensor<32x84xf32>
    %28 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%27 : tensor<32x84xf32>) outs(%24 : tensor<32x84xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_00 = arith.constant 0.000000e+00 : f32
      %35 = arith.cmpf ugt, %in, %cst_00 : f32
      %36 = arith.select %35, %in, %cst_00 : f32
      linalg.yield %36 : f32
    } -> tensor<32x84xf32>
    %29 = tensor.empty() : tensor<84x10xf32>
    %30 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%cst_10 : tensor<10x84xf32>) outs(%29 : tensor<84x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<84x10xf32>
    %31 = tensor.empty() : tensor<32x10xf32>
    %32 = linalg.fill ins(%cst_0 : f32) outs(%31 : tensor<32x10xf32>) -> tensor<32x10xf32>
    %33 = linalg.matmul ins(%28, %30 : tensor<32x84xf32>, tensor<84x10xf32>) outs(%32 : tensor<32x10xf32>) -> tensor<32x10xf32>
    %34 = linalg.generic {indexing_maps = [#map2, #map4, #map2], iterator_types = ["parallel", "parallel"]} ins(%33, %cst_11 : tensor<32x10xf32>, tensor<10xf32>) outs(%31 : tensor<32x10xf32>) {
    ^bb0(%in: f32, %in_12: f32, %out: f32):
      %35 = arith.addf %in, %in_12 : f32
      linalg.yield %35 : f32
    } -> tensor<32x10xf32>
    return %34 : tensor<32x10xf32>
  }
  
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()

  func.func @main() {
      %d1 = arith.constant 1: index
      %d0 = arith.constant 0 : index
      %n = arith.constant 1: index
  
    scf.for %i = %d0 to %n step %d1 {
      %0 = func.call @nanoTime() : () -> i64
      %1 = func.call @forward() : () -> tensor<32x10xf32>
      %2 = func.call @nanoTime() : () -> i64

      %3 = arith.subi %2, %0 : i64
      %4 = arith.uitofp %3 : i64 to f64
      // func.call @printFlops(%4) : (f64) -> ()
      func.call @printI64(%3) : (i64) -> ()
      func.call @printNewline() : () -> ()
      }
  
      return
  }
}




   