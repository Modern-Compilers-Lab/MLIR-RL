#map = affine_map<(d0) -> (d0 * 2)>
module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<1200x1000xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<1200x1500xf32>
    %c1200 = arith.constant 1200 : index
    %c750 = arith.constant 750 : index
    %1 = scf.forall (%arg0, %arg1) in (1200, 750) shared_outs(%arg2 = %0) -> (tensor<1200x1500xf32>) {
      %11 = affine.apply #map(%arg1)
      %12 = affine.apply #map(%arg1)
      %extracted_slice = tensor.extract_slice %arg2[%arg0, %12] [1, 2] [1, 1] : tensor<1200x1500xf32> to tensor<1x2xf32>
      %13 = linalg.fill ins(%cst : f32) outs(%extracted_slice : tensor<1x2xf32>) -> tensor<1x2xf32>
      %14 = affine.apply #map(%arg1)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %13 into %arg2[%arg0, %14] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<1200x1500xf32>
      }
    }
    %2 = bufferization.alloc_tensor() : tensor<1500x1000xf32>
    %c1500 = arith.constant 1500 : index
    %c500 = arith.constant 500 : index
    %3 = scf.forall (%arg0, %arg1) in (1500, 500) shared_outs(%arg2 = %2) -> (tensor<1500x1000xf32>) {
      %11 = affine.apply #map(%arg1)
      %12 = affine.apply #map(%arg1)
      %extracted_slice = tensor.extract_slice %arg2[%arg0, %12] [1, 2] [1, 1] : tensor<1500x1000xf32> to tensor<1x2xf32>
      %13 = linalg.fill ins(%cst : f32) outs(%extracted_slice : tensor<1x2xf32>) -> tensor<1x2xf32>
      %14 = affine.apply #map(%arg1)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %13 into %arg2[%arg0, %14] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<1500x1000xf32>
      }
    }
    %4 = bufferization.alloc_tensor() : tensor<1200x1000xf32>
    %c1200_1 = arith.constant 1200 : index
    %c500_2 = arith.constant 500 : index
    %5 = scf.forall (%arg0, %arg1) in (1200, 500) shared_outs(%arg2 = %4) -> (tensor<1200x1000xf32>) {
      %11 = affine.apply #map(%arg1)
      %12 = affine.apply #map(%arg1)
      %extracted_slice = tensor.extract_slice %arg2[%arg0, %12] [1, 2] [1, 1] : tensor<1200x1000xf32> to tensor<1x2xf32>
      %13 = linalg.fill ins(%cst_0 : f32) outs(%extracted_slice : tensor<1x2xf32>) -> tensor<1x2xf32>
      %14 = affine.apply #map(%arg1)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %13 into %arg2[%arg0, %14] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<1200x1000xf32>
      }
    }
    %6 = call @nanoTime() : () -> i64
    %c1200_3 = arith.constant 1200 : index
    %c500_4 = arith.constant 500 : index
    %7 = scf.forall (%arg0, %arg1) in (1200, 500) shared_outs(%arg2 = %5) -> (tensor<1200x1000xf32>) {
      %11 = affine.apply #map(%arg1)
      %12 = affine.apply #map(%arg1)
      %13 = affine.apply #map(%arg1)
      %extracted_slice = tensor.extract_slice %1[%arg0, 0] [1, 1500] [1, 1] : tensor<1200x1500xf32> to tensor<1x1500xf32>
      %extracted_slice_5 = tensor.extract_slice %3[0, %12] [1500, 2] [1, 1] : tensor<1500x1000xf32> to tensor<1500x2xf32>
      %extracted_slice_6 = tensor.extract_slice %arg2[%arg0, %13] [1, 2] [1, 1] : tensor<1200x1000xf32> to tensor<1x2xf32>
      %14 = linalg.matmul ins(%extracted_slice, %extracted_slice_5 : tensor<1x1500xf32>, tensor<1500x2xf32>) outs(%extracted_slice_6 : tensor<1x2xf32>) -> tensor<1x2xf32>
      %15 = affine.apply #map(%arg1)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %14 into %arg2[%arg0, %15] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<1200x1000xf32>
      }
    }
    %8 = call @nanoTime() : () -> i64
    %9 = arith.subi %8, %6 : i64
    %10 = arith.uitofp %9 : i64 to f64
    call @printFlops(%10) : (f64) -> ()
    call @printI64(%9) : (i64) -> ()
    return %7 : tensor<1200x1000xf32>
  }
  func.func @main() {
    %0 = call @matmul() : () -> tensor<1200x1000xf32>
    return
  }
}
