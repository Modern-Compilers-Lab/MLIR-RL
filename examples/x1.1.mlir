#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 28)>
#map2 = affine_map<(d0) -> (d0 * 8)>
#map3 = affine_map<(d0) -> (d0 * 14)>
  module attributes {torch.debug_module_name = "Net"} {
    func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
    func.func private @printFlops(f64)
    func.func private @printI64(i64)
    func.func private @printNewline()
    func.func private @printMemrefF32(tensor<*xf32>)
    func.func @matmul() -> tensor<32x64x112x112xf32> {
      %cst = arith.constant 2.000000e+00 : f32
      %0 = bufferization.alloc_tensor() : tensor<32x3x230x230xf32>
      %1 = linalg.fill {tag = "operation_0"} ins(%cst : f32) outs(%0 : tensor<32x3x230x230xf32>) -> tensor<32x3x230x230xf32>
      %2 = bufferization.alloc_tensor() : tensor<64x3x7x7xf32>
      %3 = linalg.fill {tag = "operation_1"} ins(%cst : f32) outs(%2 : tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf32>
      %4 = bufferization.alloc_tensor() : tensor<32x64x112x112xf32>
      %5 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%4 : tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xf32>
      %6 = call @nanoTime() : () -> i64
      %7 = scf.forall (%arg0, %arg1, %arg2, %arg3) in (16, 8, 8, 8) shared_outs(%arg4 = %5) -> (tensor<32x64x112x112xf32>) {
        %10 = affine.apply #map(%arg0)
        %11 = affine.apply #map1(%arg2)
        %12 = affine.apply #map1(%arg3)
        %13 = affine.apply #map2(%arg1)
        %14 = affine.apply #map(%arg0)
        %15 = affine.apply #map2(%arg1)
        %16 = affine.apply #map3(%arg2)
        %17 = affine.apply #map3(%arg3)
        %extracted_slice = tensor.extract_slice %1[%10, 0, %11, %12] [2, 3, 33, 33] [1, 1, 1, 1] : tensor<32x3x230x230xf32> to tensor<2x3x33x33xf32>
        %extracted_slice_0 = tensor.extract_slice %3[%13, 0, 0, 0] [8, 3, 7, 7] [1, 1, 1, 1] : tensor<64x3x7x7xf32> to tensor<8x3x7x7xf32>
        %extracted_slice_1 = tensor.extract_slice %arg4[%14, %15, %16, %17] [2, 8, 14, 14] [1, 1, 1, 1] : tensor<32x64x112x112xf32> to tensor<2x8x14x14xf32>
        %18 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_3"} ins(%extracted_slice, %extracted_slice_0 : tensor<2x3x33x33xf32>, tensor<8x3x7x7xf32>) outs(%extracted_slice_1 : tensor<2x8x14x14xf32>) -> tensor<2x8x14x14xf32>
        %19 = affine.apply #map(%arg0)
        %20 = affine.apply #map2(%arg1)
        %21 = affine.apply #map3(%arg2)
        %22 = affine.apply #map3(%arg3)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %18 into %arg4[%19, %20, %21, %22] [2, 8, 14, 14] [1, 1, 1, 1] : tensor<2x8x14x14xf32> into tensor<32x64x112x112xf32>
        }
      }
      %8 = call @nanoTime() : () -> i64
      %9 = arith.subi %8, %6 : i64
      call @printI64(%9) : (i64) -> ()
      call @printNewline() : () -> ()
      return %7 : tensor<32x64x112x112xf32>
    }
    func.func @main() {
      %0 = call @matmul() : () -> tensor<32x64x112x112xf32>
      return
    }
  }
