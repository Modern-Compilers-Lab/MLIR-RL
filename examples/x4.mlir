#map = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0) -> (d0 * 2)>
#map3 = affine_map<(d0) -> (d0 * 8)>
#map4 = affine_map<(d0, d1) -> (d0 * 2 + d1)>
#map5 = affine_map<(d0, d1) -> (d0 + d1)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (d0 * 4)>
#map9 = affine_map<(d0, d1) -> (d1)>
  module attributes {torch.debug_module_name = "Net"} {
    memref.global "private" @global_seed : memref<i64> = dense<0>
    func.func @forward() -> tensor<32x10xf32> {
      %c7 = arith.constant 7 : index
      %c14 = arith.constant 14 : index
      %c112 = arith.constant 112 : index
      %c5 = arith.constant 5 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c52 = arith.constant 52 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0xFF800000 : f32
      %cst_0 = arith.constant 0.000000e+00 : f32
      %0 = bufferization.alloc_tensor() : tensor<32x3x230x230xf32>
      %1 = bufferization.alloc_tensor() : tensor<64x3x7x7xf32>
      %2 = bufferization.alloc_tensor() : tensor<64xf32>
      %3 = bufferization.alloc_tensor() : tensor<16x64x5x5xf32>
      %4 = bufferization.alloc_tensor() : tensor<16xf32>
      %5 = bufferization.alloc_tensor() : tensor<120x10816xf32>
      %6 = bufferization.alloc_tensor() : tensor<120xf32>
      %7 = bufferization.alloc_tensor() : tensor<84x120xf32>
      %8 = bufferization.alloc_tensor() : tensor<84xf32>
      %9 = bufferization.alloc_tensor() : tensor<10x84xf32>
      %10 = bufferization.alloc_tensor() : tensor<10xf32>
      %11 = tensor.empty() : tensor<32x64x112x112xf32>
      %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<64xf32>) outs(%11 : tensor<32x64x112x112xf32>) attrs =  {tag = "operation_0"} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<32x64x112x112xf32>
      %13 = scf.forall (%arg0, %arg1) in (16, 8) shared_outs(%arg2 = %12) -> (tensor<32x64x112x112xf32>) {
        %46 = affine.apply #map2(%arg0)
        %47 = affine.apply #map3(%arg1)
        %48 = affine.apply #map2(%arg0)
        %49 = affine.apply #map3(%arg1)
        %extracted_slice = tensor.extract_slice %0[%46, 0, 0, 0] [2, 3, 229, 229] [1, 1, 1, 1] : tensor<32x3x230x230xf32> to tensor<2x3x229x229xf32>
        %extracted_slice_1 = tensor.extract_slice %1[%47, 0, 0, 0] [8, 3, 7, 7] [1, 1, 1, 1] : tensor<64x3x7x7xf32> to tensor<8x3x7x7xf32>
        %extracted_slice_2 = tensor.extract_slice %arg2[%48, %49, 0, 0] [2, 8, 112, 112] [1, 1, 1, 1] : tensor<32x64x112x112xf32> to tensor<2x8x112x112xf32>
        %50 = scf.for %arg3 = %c0 to %c112 step %c14 iter_args(%arg4 = %extracted_slice_2) -> (tensor<2x8x112x112xf32>) {
          %53 = scf.for %arg5 = %c0 to %c112 step %c7 iter_args(%arg6 = %arg4) -> (tensor<2x8x112x112xf32>) {
            %54 = affine.apply #map2(%arg3)
            %55 = affine.apply #map2(%arg5)
            %extracted_slice_3 = tensor.extract_slice %extracted_slice[0, 0, %54, %55] [2, 3, 33, 19] [1, 1, 1, 1] : tensor<2x3x229x229xf32> to tensor<2x3x33x19xf32>
            %extracted_slice_4 = tensor.extract_slice %arg6[0, 0, %arg3, %arg5] [2, 8, 14, 7] [1, 1, 1, 1] : tensor<2x8x112x112xf32> to tensor<2x8x14x7xf32>
            %56 = scf.for %arg7 = %c0 to %c14 step %c1 iter_args(%arg8 = %extracted_slice_4) -> (tensor<2x8x14x7xf32>) {
              %57 = scf.for %arg9 = %c0 to %c7 step %c1 iter_args(%arg10 = %arg8) -> (tensor<2x8x14x7xf32>) {
                %58 = affine.apply #map4(%arg7, %arg9)
                %extracted_slice_5 = tensor.extract_slice %extracted_slice_3[0, 0, %58, 0] [2, 3, 1, 19] [1, 1, 1, 1] : tensor<2x3x33x19xf32> to tensor<2x3x1x19xf32>
                %extracted_slice_6 = tensor.extract_slice %extracted_slice_1[0, 0, %arg9, 0] [8, 3, 1, 7] [1, 1, 1, 1] : tensor<8x3x7x7xf32> to tensor<8x3x1x7xf32>
                %extracted_slice_7 = tensor.extract_slice %arg10[0, 0, %arg7, 0] [2, 8, 1, 7] [1, 1, 1, 1] : tensor<2x8x14x7xf32> to tensor<2x8x1x7xf32>
                %extracted_slice_8 = tensor.extract_slice %extracted_slice_5[0, 0, 0, 0] [2, 3, 1, 19] [1, 1, 1, 1] : tensor<2x3x1x19xf32> to tensor<2x3x19xf32>
                %extracted_slice_9 = tensor.extract_slice %extracted_slice_6[0, 0, 0, 0] [8, 3, 1, 7] [1, 1, 1, 1] : tensor<8x3x1x7xf32> to tensor<8x3x7xf32>
                %extracted_slice_10 = tensor.extract_slice %extracted_slice_7[0, 0, 0, 0] [2, 8, 1, 7] [1, 1, 1, 1] : tensor<2x8x1x7xf32> to tensor<2x8x7xf32>
                %59 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%extracted_slice_8, %extracted_slice_9 : tensor<2x3x19xf32>, tensor<8x3x7xf32>) outs(%extracted_slice_10 : tensor<2x8x7xf32>) -> tensor<2x8x7xf32>
                %inserted_slice_11 = tensor.insert_slice %59 into %extracted_slice_7[0, 0, 0, 0] [2, 8, 1, 7] [1, 1, 1, 1] : tensor<2x8x7xf32> into tensor<2x8x1x7xf32>
                %inserted_slice_12 = tensor.insert_slice %inserted_slice_11 into %arg10[0, 0, %arg7, 0] [2, 8, 1, 7] [1, 1, 1, 1] : tensor<2x8x1x7xf32> into tensor<2x8x14x7xf32>
                scf.yield %inserted_slice_12 : tensor<2x8x14x7xf32>
              }
              scf.yield %57 : tensor<2x8x14x7xf32>
            }
            %inserted_slice = tensor.insert_slice %56 into %arg6[0, 0, %arg3, %arg5] [2, 8, 14, 7] [1, 1, 1, 1] : tensor<2x8x14x7xf32> into tensor<2x8x112x112xf32>
            scf.yield %inserted_slice : tensor<2x8x112x112xf32>
          }
          scf.yield %53 : tensor<2x8x112x112xf32>
        }
        %51 = affine.apply #map2(%arg0)
        %52 = affine.apply #map3(%arg1)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %50 into %arg2[%51, %52, 0, 0] [2, 8, 112, 112] [1, 1, 1, 1] : tensor<2x8x112x112xf32> into tensor<32x64x112x112xf32>
        }
      }
      %14 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13 : tensor<32x64x112x112xf32>) outs(%11 : tensor<32x64x112x112xf32>) attrs =  {tag = "operation_2"} {
      ^bb0(%in: f32, %out: f32):
        %46 = arith.cmpf ugt, %in, %cst_0 : f32
        %47 = arith.select %46, %in, %cst_0 : f32
        linalg.yield %47 : f32
      } -> tensor<32x64x112x112xf32>
      %15 = tensor.empty() : tensor<32x64x56x56xf32>
      %16 = linalg.fill {tag = "operation_3"} ins(%cst : f32) outs(%15 : tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf32>
      %17 = tensor.empty() : tensor<2x2xf32>
      %18 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_4"} ins(%14, %17 : tensor<32x64x112x112xf32>, tensor<2x2xf32>) outs(%16 : tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xf32>
      %19 = tensor.empty() : tensor<32x16x52x52xf32>
      %20 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4 : tensor<16xf32>) outs(%19 : tensor<32x16x52x52xf32>) attrs =  {tag = "operation_5"} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<32x16x52x52xf32>
      %21 = scf.forall (%arg0, %arg1) in (16, 2) shared_outs(%arg2 = %20) -> (tensor<32x16x52x52xf32>) {
        %46 = affine.apply #map2(%arg0)
        %47 = affine.apply #map3(%arg1)
        %48 = affine.apply #map2(%arg0)
        %49 = affine.apply #map3(%arg1)
        %extracted_slice = tensor.extract_slice %18[%46, 0, 0, 0] [2, 64, 56, 56] [1, 1, 1, 1] : tensor<32x64x56x56xf32> to tensor<2x64x56x56xf32>
        %extracted_slice_1 = tensor.extract_slice %3[%47, 0, 0, 0] [8, 64, 5, 5] [1, 1, 1, 1] : tensor<16x64x5x5xf32> to tensor<8x64x5x5xf32>
        %extracted_slice_2 = tensor.extract_slice %arg2[%48, %49, 0, 0] [2, 8, 52, 52] [1, 1, 1, 1] : tensor<32x16x52x52xf32> to tensor<2x8x52x52xf32>
        %50 = scf.for %arg3 = %c0 to %c52 step %c4 iter_args(%arg4 = %extracted_slice_2) -> (tensor<2x8x52x52xf32>) {
          %53 = scf.for %arg5 = %c0 to %c52 step %c4 iter_args(%arg6 = %arg4) -> (tensor<2x8x52x52xf32>) {
            %54 = scf.for %arg7 = %c0 to %c64 step %c8 iter_args(%arg8 = %arg6) -> (tensor<2x8x52x52xf32>) {
              %extracted_slice_3 = tensor.extract_slice %extracted_slice[0, %arg7, %arg3, %arg5] [2, 8, 8, 8] [1, 1, 1, 1] : tensor<2x64x56x56xf32> to tensor<2x8x8x8xf32>
              %extracted_slice_4 = tensor.extract_slice %extracted_slice_1[0, %arg7, 0, 0] [8, 8, 5, 5] [1, 1, 1, 1] : tensor<8x64x5x5xf32> to tensor<8x8x5x5xf32>
              %extracted_slice_5 = tensor.extract_slice %arg8[0, 0, %arg3, %arg5] [2, 8, 4, 4] [1, 1, 1, 1] : tensor<2x8x52x52xf32> to tensor<2x8x4x4xf32>
              %55 = scf.for %arg9 = %c0 to %c4 step %c1 iter_args(%arg10 = %extracted_slice_5) -> (tensor<2x8x4x4xf32>) {
                %56 = scf.for %arg11 = %c0 to %c5 step %c1 iter_args(%arg12 = %arg10) -> (tensor<2x8x4x4xf32>) {
                  %57 = affine.apply #map5(%arg9, %arg11)
                  %extracted_slice_6 = tensor.extract_slice %extracted_slice_3[0, 0, %57, 0] [2, 8, 1, 8] [1, 1, 1, 1] : tensor<2x8x8x8xf32> to tensor<2x8x1x8xf32>
                  %extracted_slice_7 = tensor.extract_slice %extracted_slice_4[0, 0, %arg11, 0] [8, 8, 1, 5] [1, 1, 1, 1] : tensor<8x8x5x5xf32> to tensor<8x8x1x5xf32>
                  %extracted_slice_8 = tensor.extract_slice %arg12[0, 0, %arg9, 0] [2, 8, 1, 4] [1, 1, 1, 1] : tensor<2x8x4x4xf32> to tensor<2x8x1x4xf32>
                  %extracted_slice_9 = tensor.extract_slice %extracted_slice_6[0, 0, 0, 0] [2, 8, 1, 8] [1, 1, 1, 1] : tensor<2x8x1x8xf32> to tensor<2x8x8xf32>
                  %extracted_slice_10 = tensor.extract_slice %extracted_slice_7[0, 0, 0, 0] [8, 8, 1, 5] [1, 1, 1, 1] : tensor<8x8x1x5xf32> to tensor<8x8x5xf32>
                  %extracted_slice_11 = tensor.extract_slice %extracted_slice_8[0, 0, 0, 0] [2, 8, 1, 4] [1, 1, 1, 1] : tensor<2x8x1x4xf32> to tensor<2x8x4xf32>
                  %58 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%extracted_slice_9, %extracted_slice_10 : tensor<2x8x8xf32>, tensor<8x8x5xf32>) outs(%extracted_slice_11 : tensor<2x8x4xf32>) -> tensor<2x8x4xf32>
                  %inserted_slice_12 = tensor.insert_slice %58 into %extracted_slice_8[0, 0, 0, 0] [2, 8, 1, 4] [1, 1, 1, 1] : tensor<2x8x4xf32> into tensor<2x8x1x4xf32>
                  %inserted_slice_13 = tensor.insert_slice %inserted_slice_12 into %arg12[0, 0, %arg9, 0] [2, 8, 1, 4] [1, 1, 1, 1] : tensor<2x8x1x4xf32> into tensor<2x8x4x4xf32>
                  scf.yield %inserted_slice_13 : tensor<2x8x4x4xf32>
                }
                scf.yield %56 : tensor<2x8x4x4xf32>
              }
              %inserted_slice = tensor.insert_slice %55 into %arg8[0, 0, %arg3, %arg5] [2, 8, 4, 4] [1, 1, 1, 1] : tensor<2x8x4x4xf32> into tensor<2x8x52x52xf32>
              scf.yield %inserted_slice : tensor<2x8x52x52xf32>
            }
            scf.yield %54 : tensor<2x8x52x52xf32>
          }
          scf.yield %53 : tensor<2x8x52x52xf32>
        }
        %51 = affine.apply #map2(%arg0)
        %52 = affine.apply #map3(%arg1)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %50 into %arg2[%51, %52, 0, 0] [2, 8, 52, 52] [1, 1, 1, 1] : tensor<2x8x52x52xf32> into tensor<32x16x52x52xf32>
        }
      }
      %22 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21 : tensor<32x16x52x52xf32>) outs(%19 : tensor<32x16x52x52xf32>) attrs =  {tag = "operation_7"} {
      ^bb0(%in: f32, %out: f32):
        %46 = arith.cmpf ugt, %in, %cst_0 : f32
        %47 = arith.select %46, %in, %cst_0 : f32
        linalg.yield %47 : f32
      } -> tensor<32x16x52x52xf32>
      %23 = tensor.empty() : tensor<32x16x26x26xf32>
      %24 = linalg.fill {tag = "operation_8"} ins(%cst : f32) outs(%23 : tensor<32x16x26x26xf32>) -> tensor<32x16x26x26xf32>
      %25 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_9"} ins(%22, %17 : tensor<32x16x52x52xf32>, tensor<2x2xf32>) outs(%24 : tensor<32x16x26x26xf32>) -> tensor<32x16x26x26xf32>
      %collapsed = tensor.collapse_shape %25 [[0], [1, 2, 3]] : tensor<32x16x26x26xf32> into tensor<32x10816xf32>
      %26 = tensor.empty() : tensor<10816x120xf32>
      %27 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<120x10816xf32>) outs(%26 : tensor<10816x120xf32>) attrs =  {tag = "operation_10"} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<10816x120xf32>
      %28 = tensor.empty() : tensor<32x120xf32>
      %29 = linalg.fill {tag = "operation_11"} ins(%cst_0 : f32) outs(%28 : tensor<32x120xf32>) -> tensor<32x120xf32>
      %30 = scf.forall (%arg0, %arg1) in (8, 5408) shared_outs(%arg2 = %29) -> (tensor<32x120xf32>) {
        %46 = affine.apply #map8(%arg0)
        %47 = affine.apply #map2(%arg1)
        %48 = affine.apply #map2(%arg1)
        %49 = affine.apply #map8(%arg0)
        %extracted_slice = tensor.extract_slice %collapsed[%46, %47] [4, 2] [1, 1] : tensor<32x10816xf32> to tensor<4x2xf32>
        %extracted_slice_1 = tensor.extract_slice %27[%48, 0] [2, 120] [1, 1] : tensor<10816x120xf32> to tensor<2x120xf32>
        %extracted_slice_2 = tensor.extract_slice %arg2[%49, 0] [4, 120] [1, 1] : tensor<32x120xf32> to tensor<4x120xf32>
        %50 = linalg.matmul {tag = "operation_12"} ins(%extracted_slice, %extracted_slice_1 : tensor<4x2xf32>, tensor<2x120xf32>) outs(%extracted_slice_2 : tensor<4x120xf32>) -> tensor<4x120xf32>
        %51 = affine.apply #map8(%arg0)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %50 into %arg2[%51, 0] [4, 120] [1, 1] : tensor<4x120xf32> into tensor<32x120xf32>
        }
      }
      %31 = linalg.generic {indexing_maps = [#map6, #map9, #map6], iterator_types = ["parallel", "parallel"]} ins(%30, %6 : tensor<32x120xf32>, tensor<120xf32>) outs(%28 : tensor<32x120xf32>) attrs =  {tag = "operation_13"} {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %46 = arith.addf %in, %in_1 : f32
        linalg.yield %46 : f32
      } -> tensor<32x120xf32>
      %32 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%31 : tensor<32x120xf32>) outs(%28 : tensor<32x120xf32>) attrs =  {tag = "operation_14"} {
      ^bb0(%in: f32, %out: f32):
        %46 = arith.cmpf ugt, %in, %cst_0 : f32
        %47 = arith.select %46, %in, %cst_0 : f32
        linalg.yield %47 : f32
      } -> tensor<32x120xf32>
      %33 = tensor.empty() : tensor<120x84xf32>
      %34 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<84x120xf32>) outs(%33 : tensor<120x84xf32>) attrs =  {tag = "operation_15"} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<120x84xf32>
      %35 = tensor.empty() : tensor<32x84xf32>
      %36 = linalg.fill {tag = "operation_16"} ins(%cst_0 : f32) outs(%35 : tensor<32x84xf32>) -> tensor<32x84xf32>
      %37 = scf.forall (%arg0, %arg1) in (4, 60) shared_outs(%arg2 = %36) -> (tensor<32x84xf32>) {
        %46 = affine.apply #map3(%arg0)
        %47 = affine.apply #map2(%arg1)
        %48 = affine.apply #map2(%arg1)
        %49 = affine.apply #map3(%arg0)
        %extracted_slice = tensor.extract_slice %32[%46, %47] [8, 2] [1, 1] : tensor<32x120xf32> to tensor<8x2xf32>
        %extracted_slice_1 = tensor.extract_slice %34[%48, 0] [2, 84] [1, 1] : tensor<120x84xf32> to tensor<2x84xf32>
        %extracted_slice_2 = tensor.extract_slice %arg2[%49, 0] [8, 84] [1, 1] : tensor<32x84xf32> to tensor<8x84xf32>
        %50 = linalg.matmul {tag = "operation_17"} ins(%extracted_slice, %extracted_slice_1 : tensor<8x2xf32>, tensor<2x84xf32>) outs(%extracted_slice_2 : tensor<8x84xf32>) -> tensor<8x84xf32>
        %51 = affine.apply #map3(%arg0)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %50 into %arg2[%51, 0] [8, 84] [1, 1] : tensor<8x84xf32> into tensor<32x84xf32>
        }
      }
      %38 = linalg.generic {indexing_maps = [#map6, #map9, #map6], iterator_types = ["parallel", "parallel"]} ins(%37, %8 : tensor<32x84xf32>, tensor<84xf32>) outs(%35 : tensor<32x84xf32>) attrs =  {tag = "operation_18"} {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %46 = arith.addf %in, %in_1 : f32
        linalg.yield %46 : f32
      } -> tensor<32x84xf32>
      %39 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel", "parallel"]} ins(%38 : tensor<32x84xf32>) outs(%35 : tensor<32x84xf32>) attrs =  {tag = "operation_19"} {
      ^bb0(%in: f32, %out: f32):
        %46 = arith.cmpf ugt, %in, %cst_0 : f32
        %47 = arith.select %46, %in, %cst_0 : f32
        linalg.yield %47 : f32
      } -> tensor<32x84xf32>
      %40 = tensor.empty() : tensor<84x10xf32>
      %41 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<10x84xf32>) outs(%40 : tensor<84x10xf32>) attrs =  {tag = "operation_20"} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<84x10xf32>
      %42 = tensor.empty() : tensor<32x10xf32>
      %43 = linalg.fill {tag = "operation_21"} ins(%cst_0 : f32) outs(%42 : tensor<32x10xf32>) -> tensor<32x10xf32>
      %44 = linalg.matmul {tag = "operation_22"} ins(%39, %41 : tensor<32x84xf32>, tensor<84x10xf32>) outs(%43 : tensor<32x10xf32>) -> tensor<32x10xf32>
      %45 = linalg.generic {indexing_maps = [#map6, #map9, #map6], iterator_types = ["parallel", "parallel"]} ins(%44, %10 : tensor<32x10xf32>, tensor<10xf32>) outs(%42 : tensor<32x10xf32>) attrs =  {tag = "operation_23"} {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %46 = arith.addf %in, %in_1 : f32
        linalg.yield %46 : f32
      } -> tensor<32x10xf32>
      return %45 : tensor<32x10xf32>
    }
    func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
    func.func private @printFlops(f64)
    func.func private @printI64(i64)
    func.func private @printNewline()
    func.func @main() {
      %0 = call @nanoTime() : () -> i64
      %1 = call @forward() : () -> tensor<32x10xf32>
      %2 = call @nanoTime() : () -> i64
      %3 = arith.subi %2, %0 : i64
      call @printI64(%3) : (i64) -> ()
      call @printNewline() : () -> ()
      return
    }
  }
