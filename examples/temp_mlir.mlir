#map = affine_map<(d0) -> (d0 * 128)>
#map1 = affine_map<(d0) -> (d0 * 16)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
  module attributes {torch.debug_module_name = "Net"} {
    func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
    func.func private @printFlops(f64)
    func.func private @printI64(i64)
    func.func private @printNewline()
    func.func private @printMemrefF32(tensor<*xf32>)
    func.func @matmul() -> tensor<512x128xf32> {
      %cst = arith.constant dense<2.000000e+00> : vector<128x16xf32>
      %cst_0 = arith.constant dense<2.000000e+00> : vector<128x128xf32>
      %c0 = arith.constant 0 : index
      %0 = bufferization.alloc_tensor() : tensor<512x1024xf32>
      %1 = bufferization.alloc_tensor() : tensor<1024x128xf32>
      %2 = bufferization.alloc_tensor() : tensor<512x128xf32>
      %3 = call @nanoTime() : () -> i64
      %4 = scf.forall (%arg0, %arg1, %arg2) in (4, 8, 8) shared_outs(%arg3 = %2) -> (tensor<512x128xf32>) {
        %7 = affine.apply #map(%arg0)
        %8 = affine.apply #map1(%arg1)
        %extracted_slice = tensor.extract_slice %arg3[%7, %8] [128, 16] [1, 1] : tensor<512x128xf32> to tensor<128x16xf32>
        %9 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %cst_0, %cst, %cst : vector<128x128xf32>, vector<128x16xf32> into vector<128x16xf32>
        %10 = vector.transfer_write %9, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<128x16xf32>, tensor<128x16xf32>
        %11 = affine.apply #map(%arg0)
        %12 = affine.apply #map1(%arg1)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %10 into %arg3[%11, %12] [128, 16] [1, 1] : tensor<128x16xf32> into tensor<512x128xf32>
        }
      }
      %5 = call @nanoTime() : () -> i64
      %6 = arith.subi %5, %3 : i64
      call @printI64(%6) : (i64) -> ()
      call @printNewline() : () -> ()
      return %4 : tensor<512x128xf32>
    }
    func.func @main() {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %0 = func.call @matmul() : () -> tensor<512x128xf32>
      }
      return
    }
  }
  
