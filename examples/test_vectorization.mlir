#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 200)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d0)>
module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<1200x1000xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<1200x1500xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1200x1500xf32>) -> tensor<1200x1500xf32>
    %2 = bufferization.alloc_tensor() : tensor<1500x1000xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1500x1000xf32>) -> tensor<1500x1000xf32>
    %4 = bufferization.alloc_tensor() : tensor<1200x1000xf32>
    %5 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<1200x1000xf32>) -> tensor<1200x1000xf32>
    %6 = call @nanoTime() : () -> i64
    %c600 = arith.constant 600 : index
    %c5 = arith.constant 5 : index
    %7 = scf.forall (%arg0, %arg1) in (600, 5) shared_outs(%arg2 = %4) -> (tensor<1200x1000xf32>) {
      %11 = affine.apply #map(%arg0)
      %12 = affine.apply #map1(%arg1)
      %13 = affine.apply #map(%arg0)
      %14 = affine.apply #map1(%arg1)
      %15 = affine.apply #map(%arg0)
      %16 = affine.apply #map1(%arg1)
      %17 = affine.apply #map(%arg0)
      %extracted_slice = tensor.extract_slice %0[%17, 0] [2, 1500] [1, 1] : tensor<1200x1500xf32> to tensor<2x1500xf32>
      %18 = linalg.fill ins(%cst : f32) outs(%extracted_slice : tensor<2x1500xf32>) -> tensor<2x1500xf32>
      %19 = affine.apply #map1(%arg1)
      %extracted_slice_1 = tensor.extract_slice %2[0, %19] [1500, 200] [1, 1] : tensor<1500x1000xf32> to tensor<1500x200xf32>
      %20 = linalg.fill ins(%cst : f32) outs(%extracted_slice_1 : tensor<1500x200xf32>) -> tensor<1500x200xf32>
      %21 = affine.apply #map(%arg0)
      %22 = affine.apply #map1(%arg1)
      %extracted_slice_2 = tensor.extract_slice %arg2[%21, %22] [2, 200] [1, 1] : tensor<1200x1000xf32> to tensor<2x200xf32>
      %23 = linalg.fill ins(%cst_0 : f32) outs(%extracted_slice_2 : tensor<2x200xf32>) -> tensor<2x200xf32>
      %24 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "reduction", "parallel"]} ins(%18, %20 : tensor<2x1500xf32>, tensor<1500x200xf32>) outs(%23 : tensor<2x200xf32>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %27 = arith.mulf %in, %in_3 : f32
        %28 = arith.addf %out, %27 : f32
        linalg.yield %28 : f32
      } -> tensor<2x200xf32>
      %25 = affine.apply #map(%arg0)
      %26 = affine.apply #map1(%arg1)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %24 into %arg2[%25, %26] [2, 200] [1, 1] : tensor<2x200xf32> into tensor<1200x1000xf32>
      }
    }
    %8 = call @nanoTime() : () -> i64
    %9 = arith.subi %8, %6 : i64
    %10 = arith.uitofp %9 : i64 to f64
    call @printFlops(%10) : (f64) -> ()
    return %7 : tensor<1200x1000xf32>
  }
  func.func @main() {
    %0 = call @matmul() : () -> tensor<1200x1000xf32>
    return
  }
}

