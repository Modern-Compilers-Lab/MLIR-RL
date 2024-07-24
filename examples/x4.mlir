#map = affine_map<(d0) -> (d0 * 14)>
#map1 = affine_map<(d0) -> (d0 * 4)>
#map2 = affine_map<(d0) -> (d0 * 7)>
#map3 = affine_map<(d0) -> (d0 * 2)>
#map4 = affine_map<(d0, d1) -> (d0 * 2 + d1)>
  module attributes {torch.debug_module_name = "Net"} {
    func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
    func.func private @printFlops(f64)
    func.func private @printI64(i64)
    func.func private @printNewline()
    func.func private @printMemrefF32(tensor<*xf32>)
    func.func @matmul() -> tensor<32x64x112x112xf32> {
      %c1 = arith.constant 1 : index
      %c7 = arith.constant 7 : index
      %c28 = arith.constant 28 : index
      %c112 = arith.constant 112 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %0 = bufferization.alloc_tensor() : tensor<32x3x230x230xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x3x230x230xf32>) -> tensor<32x3x230x230xf32>
      %2 = bufferization.alloc_tensor() : tensor<64x3x7x7xf32>
      %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf32>
      %4 = bufferization.alloc_tensor() : tensor<32x64x112x112xf32>
      %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xf32>
      %6 = call @nanoTime() : () -> i64
      %7 = scf.forall (%arg0, %arg1) in (16, 16) shared_outs(%arg2 = %5) -> (tensor<32x64x112x112xf32>) {
        %10 = affine.apply #map(%arg1)
        %11 = affine.apply #map1(%arg0)
        %12 = affine.apply #map1(%arg0)
        %13 = affine.apply #map2(%arg1)
        %extracted_slice = tensor.extract_slice %1[0, 0, %10, 0] [32, 3, 19, 229] [1, 1, 1, 1] : tensor<32x3x230x230xf32> to tensor<32x3x19x229xf32>
        %extracted_slice_0 = tensor.extract_slice %3[%11, 0, 0, 0] [4, 3, 7, 7] [1, 1, 1, 1] : tensor<64x3x7x7xf32> to tensor<4x3x7x7xf32>
        %extracted_slice_1 = tensor.extract_slice %arg2[0, %12, %13, 0] [32, 4, 7, 112] [1, 1, 1, 1] : tensor<32x64x112x112xf32> to tensor<32x4x7x112xf32>
        %14 = scf.for %arg3 = %c0 to %c32 step %c8 iter_args(%arg4 = %extracted_slice_1) -> (tensor<32x4x7x112xf32>) {
          %17 = scf.for %arg5 = %c0 to %c112 step %c28 iter_args(%arg6 = %arg4) -> (tensor<32x4x7x112xf32>) {
            %18 = affine.apply #map3(%arg5)
            %extracted_slice_2 = tensor.extract_slice %extracted_slice[%arg3, 0, 0, %18] [8, 3, 19, 61] [1, 1, 1, 1] : tensor<32x3x19x229xf32> to tensor<8x3x19x61xf32>
            %extracted_slice_3 = tensor.extract_slice %arg6[%arg3, 0, 0, %arg5] [8, 4, 7, 28] [1, 1, 1, 1] : tensor<32x4x7x112xf32> to tensor<8x4x7x28xf32>
            %19 = scf.for %arg7 = %c0 to %c7 step %c1 iter_args(%arg8 = %extracted_slice_3) -> (tensor<8x4x7x28xf32>) {
              %20 = scf.for %arg9 = %c0 to %c7 step %c1 iter_args(%arg10 = %arg8) -> (tensor<8x4x7x28xf32>) {
                %21 = affine.apply #map4(%arg7, %arg9)
                %extracted_slice_4 = tensor.extract_slice %extracted_slice_2[0, 0, %21, 0] [8, 3, 1, 61] [1, 1, 1, 1] : tensor<8x3x19x61xf32> to tensor<8x3x1x61xf32>
                %extracted_slice_5 = tensor.extract_slice %extracted_slice_0[0, 0, %arg9, 0] [4, 3, 1, 7] [1, 1, 1, 1] : tensor<4x3x7x7xf32> to tensor<4x3x1x7xf32>
                %extracted_slice_6 = tensor.extract_slice %arg10[0, 0, %arg7, 0] [8, 4, 1, 28] [1, 1, 1, 1] : tensor<8x4x7x28xf32> to tensor<8x4x1x28xf32>
                %extracted_slice_7 = tensor.extract_slice %extracted_slice_4[0, 0, 0, 0] [8, 3, 1, 61] [1, 1, 1, 1] : tensor<8x3x1x61xf32> to tensor<8x3x61xf32>
                %extracted_slice_8 = tensor.extract_slice %extracted_slice_5[0, 0, 0, 0] [4, 3, 1, 7] [1, 1, 1, 1] : tensor<4x3x1x7xf32> to tensor<4x3x7xf32>
                %extracted_slice_9 = tensor.extract_slice %extracted_slice_6[0, 0, 0, 0] [8, 4, 1, 28] [1, 1, 1, 1] : tensor<8x4x1x28xf32> to tensor<8x4x28xf32>
                %22 = linalg.conv_1d_ncw_fcw {tag = "operation_1", dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%extracted_slice_7, %extracted_slice_8 : tensor<8x3x61xf32>, tensor<4x3x7xf32>) outs(%extracted_slice_9 : tensor<8x4x28xf32>) -> tensor<8x4x28xf32>
                %inserted_slice_10 = tensor.insert_slice %22 into %extracted_slice_6[0, 0, 0, 0] [8, 4, 1, 28] [1, 1, 1, 1] : tensor<8x4x28xf32> into tensor<8x4x1x28xf32>
                %inserted_slice_11 = tensor.insert_slice %inserted_slice_10 into %arg10[0, 0, %arg7, 0] [8, 4, 1, 28] [1, 1, 1, 1] : tensor<8x4x1x28xf32> into tensor<8x4x7x28xf32>
                scf.yield %inserted_slice_11 : tensor<8x4x7x28xf32>
              }
              scf.yield %20 : tensor<8x4x7x28xf32>
            }
            %inserted_slice = tensor.insert_slice %19 into %arg6[%arg3, 0, 0, %arg5] [8, 4, 7, 28] [1, 1, 1, 1] : tensor<8x4x7x28xf32> into tensor<32x4x7x112xf32>
            scf.yield %inserted_slice : tensor<32x4x7x112xf32>
          }
          scf.yield %17 : tensor<32x4x7x112xf32>
        }
        %15 = affine.apply #map1(%arg0)
        %16 = affine.apply #map2(%arg1)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %14 into %arg2[0, %15, %16, 0] [32, 4, 7, 112] [1, 1, 1, 1] : tensor<32x4x7x112xf32> into tensor<32x64x112x112xf32>
        }
      }
      %8 = call @nanoTime() : () -> i64
      %9 = arith.subi %8, %6 : i64
      call @printI64(%9) : (i64) -> ()
      call @printNewline() : () -> ()
      return %7 : tensor<32x64x112x112xf32>
    }
    func.func @main() {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %0 = func.call @matmul() : () -> tensor<32x64x112x112xf32>
      }
      return
    }
  }

module attributes {transform.with_named_sequence} {
transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) 
{

  %foralls = transform.structured.match ops{["scf.forall"]}  in %variant_op : (!transform.any_op) -> !transform.any_op

  transform.print %foralls {name = "foralls"}: !transform.any_op 


  transform.yield
}
}


// /scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt examples/x4.mlir -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule