#map = affine_map<(d0) -> (d0 * 12)>
#map1 = affine_map<(d0) -> (d0 * 8)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<1200x1000xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<2.000000e+00> : vector<1500x8xf32>
    %cst_1 = arith.constant dense<2.000000e+00> : vector<12x1500xf32>
    %c0 = arith.constant 0 : index
    %0 = bufferization.alloc_tensor() : tensor<1200x1500xf32>
    %1 = bufferization.alloc_tensor() : tensor<1500x1000xf32>
    %2 = bufferization.alloc_tensor() : tensor<1200x1000xf32>
    %3 = call @nanoTime() : () -> i64
    %4 = scf.forall (%arg0, %arg1) in (100, 125) shared_outs(%arg2 = %2) -> (tensor<1200x1000xf32>) {
      %8 = affine.apply #map(%arg0)
      %9 = affine.apply #map1(%arg1)
      %extracted_slice = tensor.extract_slice %arg2[%8, %9] [12, 8] [1, 1] : tensor<1200x1000xf32> to tensor<12x8xf32>
      %10 = affine.apply #map(%arg0)
      %11 = affine.apply #map1(%arg1)
      %12 = vector.transfer_read %arg2[%10, %11], %cst {in_bounds = [true, true]} : tensor<1200x1000xf32>, vector<12x8xf32>
      %13 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %cst_1, %cst_0, %12 : vector<12x1500xf32>, vector<1500x8xf32> into vector<12x8xf32>
      %14 = vector.transfer_write %13, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<12x8xf32>, tensor<12x8xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %14 into %arg2[%8, %9] [12, 8] [1, 1] : tensor<12x8xf32> into tensor<1200x1000xf32>
      }
    } {consumer0}
    %5 = call @nanoTime() : () -> i64
    %6 = arith.subi %5, %3 : i64
    %7 = arith.uitofp %6 : i64 to f64
    call @printFlops(%7) : (f64) -> ()
    return %4 : tensor<1200x1000xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<1200x1000xf32>
    }
    return
  }
}