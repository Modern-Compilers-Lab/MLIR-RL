#map = affine_map<(d0) -> (d0 * 8)>
  module attributes {torch.debug_module_name = "Net"} {
    func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
    func.func private @printFlops(f64)
    func.func private @printI64(i64)
    func.func private @printNewline()
    func.func private @printMemrefF32(tensor<*xf32>)
    func.func @matmul() -> tensor<64x128x512xf32> {
      %cst = arith.constant dense<4.000000e+00> : vector<8x128x512xf32>
      %c0 = arith.constant 0 : index
      %0 = bufferization.alloc_tensor() : tensor<64x128x512xf32>
      %1 = bufferization.alloc_tensor() : tensor<64x128x512xf32>
      %2 = bufferization.alloc_tensor() : tensor<64x128x512xf32>
      %3 = call @nanoTime() : () -> i64
      %4 = scf.forall (%arg0) in (8) shared_outs(%arg1 = %2) -> (tensor<64x128x512xf32>) {
        %7 = affine.apply #map(%arg0)
        %extracted_slice = tensor.extract_slice %arg1[%7, 0, 0] [8, 128, 512] [1, 1, 1] : tensor<64x128x512xf32> to tensor<8x128x512xf32>
        %8 = vector.transfer_write %cst, %extracted_slice[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<8x128x512xf32>, tensor<8x128x512xf32>
        %9 = affine.apply #map(%arg0)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %8 into %arg1[%9, 0, 0] [8, 128, 512] [1, 1, 1] : tensor<8x128x512xf32> into tensor<64x128x512xf32>
        }
      }
      %5 = call @nanoTime() : () -> i64
      %6 = arith.subi %5, %3 : i64
      call @printI64(%6) : (i64) -> ()
      call @printNewline() : () -> ()
      return %4 : tensor<64x128x512xf32>
    }
    func.func @main() {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %0 = func.call @matmul() : () -> tensor<64x128x512xf32>
      }
      return
    }
  }
  
