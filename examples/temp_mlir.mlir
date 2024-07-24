#map = affine_map<(d0) -> (d0 * 8)>
#map1 = affine_map<(d0) -> (d0 * 4)>
#map2 = affine_map<(d0) -> (d0 * 448)>
#map3 = affine_map<(d0) -> (d0 * 21)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
  module attributes {torch.debug_module_name = "Net"} {
    func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
    func.func private @printFlops(f64)
    func.func private @printI64(i64)
    func.func private @printNewline()
    func.func private @printMemrefF32(tensor<*xf32>)
    func.func @matmul() -> tensor<32x64x112x112xf32> {
      %cst = arith.constant 0.000000e+00 : f32
      %cst_0 = arith.constant dense<2.000000e+00> : vector<32x147x12544xf32>
      %cst_1 = arith.constant dense<2.000000e+00> : vector<32x64x12544xf32>
      %cst_2 = arith.constant dense<2.000000e+00> : vector<64x147xf32>
      %c0 = arith.constant 0 : index
      %0 = bufferization.alloc_tensor() : tensor<32x3x230x230xf32>
      %1 = bufferization.alloc_tensor() : tensor<64x3x7x7xf32>
      %2 = bufferization.alloc_tensor() : tensor<32x64x112x112xf32>
      %3 = call @nanoTime() : () -> i64
      %collapsed = tensor.collapse_shape %1 [[0], [1, 2, 3]] : tensor<64x3x7x7xf32> into tensor<64x147xf32>
      %4 = vector.transfer_write %cst_2, %collapsed[%c0, %c0] {in_bounds = [true, true]} : vector<64x147xf32>, tensor<64x147xf32>
      %collapsed_3 = tensor.collapse_shape %2 [[0], [1], [2, 3]] : tensor<32x64x112x112xf32> into tensor<32x64x12544xf32>
      %5 = vector.transfer_write %cst_1, %collapsed_3[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<32x64x12544xf32>, tensor<32x64x12544xf32>
      %6 = tensor.empty() : tensor<32x147x12544xf32>
      %7 = vector.transfer_write %cst_0, %6[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<32x147x12544xf32>, tensor<32x147x12544xf32>
      %8 = scf.forall (%arg0, %arg1, %arg2, %arg3) in (4, 16, 28, 7) shared_outs(%arg4 = %5) -> (tensor<32x64x12544xf32>) {
        %11 = affine.apply #map(%arg0)
        %12 = affine.apply #map1(%arg1)
        %13 = affine.apply #map2(%arg2)
        %extracted_slice = tensor.extract_slice %arg4[%11, %12, %13] [8, 4, 448] [1, 1, 1] : tensor<32x64x12544xf32> to tensor<8x4x448xf32>
        %14 = affine.apply #map1(%arg1)
        %15 = affine.apply #map3(%arg3)
        %16 = vector.transfer_read %4[%14, %15], %cst {in_bounds = [true, true]} : tensor<64x147xf32>, vector<4x21xf32>
        %17 = affine.apply #map(%arg0)
        %18 = affine.apply #map3(%arg3)
        %19 = affine.apply #map2(%arg2)
        %20 = vector.transfer_read %7[%17, %18, %19], %cst {in_bounds = [true, true, true]} : tensor<32x147x12544xf32>, vector<8x21x448xf32>
        %21 = affine.apply #map(%arg0)
        %22 = affine.apply #map1(%arg1)
        %23 = affine.apply #map2(%arg2)
        %24 = vector.transfer_read %arg4[%21, %22, %23], %cst {in_bounds = [true, true, true]} : tensor<32x64x12544xf32>, vector<8x4x448xf32>
        %25 = vector.contract {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %16, %20, %24 : vector<4x21xf32>, vector<8x21x448xf32> into vector<8x4x448xf32>
        %26 = vector.transfer_write %25, %extracted_slice[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<8x4x448xf32>, tensor<8x4x448xf32>
        %27 = affine.apply #map(%arg0)
        %28 = affine.apply #map1(%arg1)
        %29 = affine.apply #map2(%arg2)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %26 into %arg4[%27, %28, %29] [8, 4, 448] [1, 1, 1] : tensor<8x4x448xf32> into tensor<32x64x12544xf32>
        }
      }
      %expanded = tensor.expand_shape %8 [[0], [1], [2, 3]] : tensor<32x64x12544xf32> into tensor<32x64x112x112xf32>
      %9 = call @nanoTime() : () -> i64
      %10 = arith.subi %9, %3 : i64
      call @printI64(%10) : (i64) -> ()
      call @printNewline() : () -> ()
      return %expanded : tensor<32x64x112x112xf32>
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
  
