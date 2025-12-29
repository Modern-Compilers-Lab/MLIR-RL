
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d4 + d2, d5 + d3, d0, d6, d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d4, d5, d0, d6, d7)>
#map3 = affine_map<(d0) -> (d0 * 4)>
#map4 = affine_map<(d0) -> (d0 * 2)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d6, d7, d1, d0, d3, d5, d2)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d7, d0, d5, d6, d1, d3, d2)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d2, d7, d0, d1, d6, d5, d3)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d7, d1, d5, d2, d0, d6, d3)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
#map10 = affine_map<(d0) -> (d0 * 8)>
#map11 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d2, d4, d6, d1, d3, d5, d7)>
#map12 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d2, d6, d7, d5, d1, d3, d0)>
#map13 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d6, d5, d3, d2, d7, d1, d0)>
#map14 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d6, d1, d2, d3, d5, d7, d0)>
#map15 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d5, d2, d4, d6, d3, d7, d0)>
#map16 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d5, d4, d1, d6, d3)>
#map17 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d2, d4, d6, d5)>
#map18 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d5, d6, d3, d0, d4)>
#map19 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d0, d6, d2, d5, d3)>
#map20 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d4, d5, d0, d6, d3)>
#map21 = affine_map<(d0, d1, d2, d3) -> (d3, d0, d2, d1)>
#map22 = affine_map<(d0, d1) -> (d0 + d1)>
#map23 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d1 * 2 + d2, d5 * 2 + d4)>
#map24 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d4)>
#map25 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d1, d5)>
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printI64(i64)
  func.func private @printF32(f32)
  func.func private @printNewline()
  func.func @main(%arg0: tensor<128x130x228x192xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<128x130x228x192xf32>) -> (tensor<128x128x55x47xf32>, i64) attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    %1 = tensor.empty() : tensor<64x130x228x48x4x2xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %arg0 low[0, 0, 0, 0] high[0, 0, 0, 0] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst : f32
    } : tensor<128x130x228x192xf32> to tensor<128x130x228x192xf32>
    %expanded = tensor.expand_shape %padded [[0, 1], [2], [3], [4, 5]] output_shape [64, 2, 130, 228, 48, 4] : tensor<128x130x228x192xf32> into tensor<64x2x130x228x48x4xf32>
    %transposed = linalg.transpose ins(%expanded : tensor<64x2x130x228x48x4xf32>) outs(%1 : tensor<64x130x228x48x4x2xf32>) permutation = [0, 2, 3, 4, 5, 1]  {tag = "operation_5"}
    %2 = tensor.empty() : tensor<64x130x228x48x4x2xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %padded_1 = tensor.pad %arg2 low[0, 0, 0, 0] high[0, 0, 0, 0] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst_0 : f32
    } : tensor<128x130x228x192xf32> to tensor<128x130x228x192xf32>
    %expanded_2 = tensor.expand_shape %padded_1 [[0, 1], [2], [3], [4, 5]] output_shape [64, 2, 130, 228, 48, 4] : tensor<128x130x228x192xf32> into tensor<64x2x130x228x48x4xf32>
    %transposed_3 = linalg.transpose ins(%expanded_2 : tensor<64x2x130x228x48x4xf32>) outs(%2 : tensor<64x130x228x48x4x2xf32>) permutation = [0, 2, 3, 4, 5, 1]  {tag = "operation_6"}
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "parallel", "parallel"]} ins(%transposed, %arg1 : tensor<64x130x228x48x4x2xf32>, tensor<1x1xf32>) outs(%transposed_3 : tensor<64x130x228x48x4x2xf32>) attrs =  {tag = "operation_0"} {
    ^bb0(%in: f32, %in_67: f32, %out: f32):
      %42 = arith.maximumf %out, %in : f32
      linalg.yield %42 : f32
    } -> tensor<64x130x228x48x4x2xf32>
    %4 = tensor.empty() : tensor<64x2x130x228x48x4xf32>
    %c16 = arith.constant 16 : index
    %c2 = arith.constant 2 : index
    %c65 = arith.constant 65 : index
    %c114 = arith.constant 114 : index
    %c48 = arith.constant 48 : index
    %5 = scf.forall (%arg3, %arg4, %arg5, %arg6, %arg7) in (16, 2, 65, 114, 48) shared_outs(%arg8 = %4) -> (tensor<64x2x130x228x48x4xf32>) {
      %42 = affine.apply #map3(%arg3)
      %43 = affine.apply #map4(%arg5)
      %44 = affine.apply #map4(%arg6)
      %45 = affine.apply #map3(%arg3)
      %46 = affine.apply #map4(%arg5)
      %47 = affine.apply #map4(%arg6)
      %48 = affine.apply #map3(%arg3)
      %49 = affine.apply #map4(%arg5)
      %50 = affine.apply #map4(%arg6)
      %51 = affine.apply #map3(%arg3)
      %52 = affine.apply #map4(%arg5)
      %53 = affine.apply #map4(%arg6)
      %54 = affine.apply #map3(%arg3)
      %55 = affine.apply #map4(%arg5)
      %56 = affine.apply #map4(%arg6)
      %extracted_slice_67 = tensor.extract_slice %transposed[%51, %52, %53, %arg7, 0, %arg4] [4, 2, 2, 1, 4, 1] [1, 1, 1, 1, 1, 1] : tensor<64x130x228x48x4x2xf32> to tensor<4x2x2x1x4x1xf32>
      %extracted_slice_68 = tensor.extract_slice %arg1[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
      %extracted_slice_69 = tensor.extract_slice %transposed_3[%54, %55, %56, %arg7, 0, %arg4] [4, 2, 2, 1, 4, 1] [1, 1, 1, 1, 1, 1] : tensor<64x130x228x48x4x2xf32> to tensor<4x2x2x1x4x1xf32>
      %57 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_67, %extracted_slice_68 : tensor<4x2x2x1x4x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_69 : tensor<4x2x2x1x4x1xf32>) attrs =  {tag = "operation_0"} {
      ^bb0(%in: f32, %in_72: f32, %out: f32):
        %61 = arith.maximumf %out, %in : f32
        linalg.yield %61 : f32
      } -> tensor<4x2x2x1x4x1xf32>
      %extracted_slice_70 = tensor.extract_slice %arg8[%48, %arg4, %49, %50, %arg7, 0] [4, 1, 2, 2, 1, 4] [1, 1, 1, 1, 1, 1] : tensor<64x2x130x228x48x4xf32> to tensor<4x1x2x2x1x4xf32>
      %transposed_71 = linalg.transpose ins(%57 : tensor<4x2x2x1x4x1xf32>) outs(%extracted_slice_70 : tensor<4x1x2x2x1x4xf32>) permutation = [0, 5, 1, 2, 3, 4]  {tag = "operation_7"}
      %58 = affine.apply #map3(%arg3)
      %59 = affine.apply #map4(%arg5)
      %60 = affine.apply #map4(%arg6)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %transposed_71 into %arg8[%58, %arg4, %59, %60, %arg7, 0] [4, 1, 2, 2, 1, 4] [1, 1, 1, 1, 1, 1] : tensor<4x1x2x2x1x4xf32> into tensor<64x2x130x228x48x4xf32>
      }
    }
    %collapsed = tensor.collapse_shape %5 [[0, 1], [2], [3], [4, 5]] : tensor<64x2x130x228x48x4xf32> into tensor<128x130x228x192xf32>
    %extracted_slice = tensor.extract_slice %collapsed[0, 0, 0, 0] [128, 130, 228, 192] [1, 1, 1, 1] : tensor<128x130x228x192xf32> to tensor<128x130x228x192xf32>
    %6 = tensor.empty() : tensor<4x65x114x3x32x2x2x64xf32>
    %cst_4 = arith.constant 0.000000e+00 : f32
    %padded_5 = tensor.pad %extracted_slice low[0, 0, 0, 0] high[0, 0, 0, 0] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst_4 : f32
    } : tensor<128x130x228x192xf32> to tensor<128x130x228x192xf32>
    %expanded_6 = tensor.expand_shape %padded_5 [[0, 1], [2, 3], [4, 5], [6, 7]] output_shape [4, 32, 65, 2, 114, 2, 3, 64] : tensor<128x130x228x192xf32> into tensor<4x32x65x2x114x2x3x64xf32>
    %7 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_6 : tensor<4x32x65x2x114x2x3x64xf32>) outs(%6 : tensor<4x65x114x3x32x2x2x64xf32>) attrs =  {tag = "operation_9"} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<4x65x114x3x32x2x2x64xf32>
    %8 = tensor.empty() : tensor<4x65x114x3x32x2x2x64xf32>
    %cst_7 = arith.constant 0.000000e+00 : f32
    %padded_8 = tensor.pad %arg2 low[0, 0, 0, 0] high[0, 0, 0, 0] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst_7 : f32
    } : tensor<128x130x228x192xf32> to tensor<128x130x228x192xf32>
    %expanded_9 = tensor.expand_shape %padded_8 [[0, 1], [2, 3], [4, 5], [6, 7]] output_shape [4, 32, 65, 2, 114, 2, 3, 64] : tensor<128x130x228x192xf32> into tensor<4x32x65x2x114x2x3x64xf32>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c2_10 = arith.constant 2 : index
    %9 = scf.for %arg3 = %c0 to %c4 step %c2_10 iter_args(%arg4 = %8) -> (tensor<4x65x114x3x32x2x2x64xf32>) {
      %c0_67 = arith.constant 0 : index
      %c114_68 = arith.constant 114 : index
      %c1_69 = arith.constant 1 : index
      %42 = scf.for %arg5 = %c0_67 to %c114_68 step %c1_69 iter_args(%arg6 = %arg4) -> (tensor<4x65x114x3x32x2x2x64xf32>) {
        %c0_70 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_71 = arith.constant 1 : index
        %43 = scf.for %arg7 = %c0_70 to %c3 step %c1_71 iter_args(%arg8 = %arg6) -> (tensor<4x65x114x3x32x2x2x64xf32>) {
          %c0_72 = arith.constant 0 : index
          %c64_73 = arith.constant 64 : index
          %c1_74 = arith.constant 1 : index
          %44 = scf.for %arg9 = %c0_72 to %c64_73 step %c1_74 iter_args(%arg10 = %arg8) -> (tensor<4x65x114x3x32x2x2x64xf32>) {
            %extracted_slice_75 = tensor.extract_slice %expanded_9[%arg3, 0, 0, 0, %arg5, 0, %arg7, %arg9] [2, 32, 65, 2, 1, 2, 1, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<4x32x65x2x114x2x3x64xf32> to tensor<2x32x65x2x1x2x1x1xf32>
            %extracted_slice_76 = tensor.extract_slice %arg10[%arg3, 0, %arg5, %arg7, 0, 0, 0, %arg9] [2, 65, 1, 1, 32, 2, 2, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<4x65x114x3x32x2x2x64xf32> to tensor<2x65x1x1x32x2x2x1xf32>
            %45 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_75 : tensor<2x32x65x2x1x2x1x1xf32>) outs(%extracted_slice_76 : tensor<2x65x1x1x32x2x2x1xf32>) attrs =  {tag = "operation_10"} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            } -> tensor<2x65x1x1x32x2x2x1xf32>
            %inserted_slice = tensor.insert_slice %45 into %arg10[%arg3, 0, %arg5, %arg7, 0, 0, 0, %arg9] [2, 65, 1, 1, 32, 2, 2, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x65x1x1x32x2x2x1xf32> into tensor<4x65x114x3x32x2x2x64xf32>
            scf.yield %inserted_slice : tensor<4x65x114x3x32x2x2x64xf32>
          }
          scf.yield %44 : tensor<4x65x114x3x32x2x2x64xf32>
        }
        scf.yield %43 : tensor<4x65x114x3x32x2x2x64xf32>
      }
      scf.yield %42 : tensor<4x65x114x3x32x2x2x64xf32>
    }
    %c0_11 = arith.constant 0 : index
    %c4_12 = arith.constant 4 : index
    %c2_13 = arith.constant 2 : index
    %10 = scf.for %arg3 = %c0_11 to %c4_12 step %c2_13 iter_args(%arg4 = %9) -> (tensor<4x65x114x3x32x2x2x64xf32>) {
      %c0_67 = arith.constant 0 : index
      %c114_68 = arith.constant 114 : index
      %c1_69 = arith.constant 1 : index
      %42 = scf.for %arg5 = %c0_67 to %c114_68 step %c1_69 iter_args(%arg6 = %arg4) -> (tensor<4x65x114x3x32x2x2x64xf32>) {
        %c0_70 = arith.constant 0 : index
        %c64_71 = arith.constant 64 : index
        %c2_72 = arith.constant 2 : index
        %43 = scf.for %arg7 = %c0_70 to %c64_71 step %c2_72 iter_args(%arg8 = %arg6) -> (tensor<4x65x114x3x32x2x2x64xf32>) {
          %extracted_slice_73 = tensor.extract_slice %7[%arg3, 0, %arg5, 0, 0, 0, 0, %arg7] [2, 65, 1, 3, 32, 2, 2, 2] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<4x65x114x3x32x2x2x64xf32> to tensor<2x65x1x3x32x2x2x2xf32>
          %extracted_slice_74 = tensor.extract_slice %arg8[%arg3, 0, %arg5, 0, 0, 0, 0, %arg7] [2, 65, 1, 3, 32, 2, 2, 2] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<4x65x114x3x32x2x2x64xf32> to tensor<2x65x1x3x32x2x2x2xf32>
          %c0_75 = arith.constant 0 : index
          %c2_76 = arith.constant 2 : index
          %c1_77 = arith.constant 1 : index
          %44 = scf.for %arg9 = %c0_75 to %c2_76 step %c1_77 iter_args(%arg10 = %extracted_slice_74) -> (tensor<2x65x1x3x32x2x2x2xf32>) {
            %c0_78 = arith.constant 0 : index
            %c3 = arith.constant 3 : index
            %c1_79 = arith.constant 1 : index
            %45 = scf.for %arg11 = %c0_78 to %c3 step %c1_79 iter_args(%arg12 = %arg10) -> (tensor<2x65x1x3x32x2x2x2xf32>) {
              %extracted_slice_80 = tensor.extract_slice %extracted_slice_73[%arg9, 0, 0, %arg11, 0, 0, 0, 0] [1, 65, 1, 1, 32, 2, 2, 2] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x65x1x3x32x2x2x2xf32> to tensor<1x65x1x1x32x2x2x2xf32>
              %extracted_slice_81 = tensor.extract_slice %arg12[%arg9, 0, 0, %arg11, 0, 0, 0, 0] [1, 65, 1, 1, 32, 2, 2, 2] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x65x1x3x32x2x2x2xf32> to tensor<1x65x1x1x32x2x2x2xf32>
              %c0_82 = arith.constant 0 : index
              %c2_83 = arith.constant 2 : index
              %c1_84 = arith.constant 1 : index
              %46 = scf.for %arg13 = %c0_82 to %c2_83 step %c1_84 iter_args(%arg14 = %extracted_slice_81) -> (tensor<1x65x1x1x32x2x2x2xf32>) {
                %extracted_slice_86 = tensor.extract_slice %extracted_slice_80[0, 0, 0, 0, 0, %arg13, 0, 0] [1, 65, 1, 1, 32, 1, 2, 2] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x65x1x1x32x2x2x2xf32> to tensor<1x65x1x1x32x1x2x2xf32>
                %extracted_slice_87 = tensor.extract_slice %arg14[0, 0, 0, 0, 0, %arg13, 0, 0] [1, 65, 1, 1, 32, 1, 2, 2] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x65x1x1x32x2x2x2xf32> to tensor<1x65x1x1x32x1x2x2xf32>
                %47 = linalg.generic {indexing_maps = [#map9, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_86 : tensor<1x65x1x1x32x1x2x2xf32>) outs(%extracted_slice_87 : tensor<1x65x1x1x32x1x2x2xf32>) attrs =  {tag = "operation_8"} {
                ^bb0(%in: f32, %out: f32):
                  linalg.yield %in : f32
                } -> tensor<1x65x1x1x32x1x2x2xf32>
                %inserted_slice_88 = tensor.insert_slice %47 into %arg14[0, 0, 0, 0, 0, %arg13, 0, 0] [1, 65, 1, 1, 32, 1, 2, 2] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x65x1x1x32x1x2x2xf32> into tensor<1x65x1x1x32x2x2x2xf32>
                scf.yield %inserted_slice_88 : tensor<1x65x1x1x32x2x2x2xf32>
              }
              %inserted_slice_85 = tensor.insert_slice %46 into %arg12[%arg9, 0, 0, %arg11, 0, 0, 0, 0] [1, 65, 1, 1, 32, 2, 2, 2] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x65x1x1x32x2x2x2xf32> into tensor<2x65x1x3x32x2x2x2xf32>
              scf.yield %inserted_slice_85 : tensor<2x65x1x3x32x2x2x2xf32>
            }
            scf.yield %45 : tensor<2x65x1x3x32x2x2x2xf32>
          }
          %inserted_slice = tensor.insert_slice %44 into %arg8[%arg3, 0, %arg5, 0, 0, 0, 0, %arg7] [2, 65, 1, 3, 32, 2, 2, 2] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x65x1x3x32x2x2x2xf32> into tensor<4x65x114x3x32x2x2x64xf32>
          scf.yield %inserted_slice : tensor<4x65x114x3x32x2x2x64xf32>
        }
        scf.yield %43 : tensor<4x65x114x3x32x2x2x64xf32>
      }
      scf.yield %42 : tensor<4x65x114x3x32x2x2x64xf32>
    }
    %11 = tensor.empty() : tensor<4x32x65x2x114x2x3x64xf32>
    %c0_14 = arith.constant 0 : index
    %c4_15 = arith.constant 4 : index
    %c2_16 = arith.constant 2 : index
    %12 = scf.for %arg3 = %c0_14 to %c4_15 step %c2_16 iter_args(%arg4 = %11) -> (tensor<4x32x65x2x114x2x3x64xf32>) {
      %c0_67 = arith.constant 0 : index
      %c64_68 = arith.constant 64 : index
      %c4_69 = arith.constant 4 : index
      %42 = scf.for %arg5 = %c0_67 to %c64_68 step %c4_69 iter_args(%arg6 = %arg4) -> (tensor<4x32x65x2x114x2x3x64xf32>) {
        %extracted_slice_70 = tensor.extract_slice %10[%arg3, 0, 0, 0, 0, 0, 0, %arg5] [2, 65, 114, 3, 32, 2, 2, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<4x65x114x3x32x2x2x64xf32> to tensor<2x65x114x3x32x2x2x4xf32>
        %extracted_slice_71 = tensor.extract_slice %arg6[%arg3, 0, 0, 0, 0, 0, 0, %arg5] [2, 32, 65, 2, 114, 2, 3, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<4x32x65x2x114x2x3x64xf32> to tensor<2x32x65x2x114x2x3x4xf32>
        %transposed_72 = linalg.transpose ins(%extracted_slice_70 : tensor<2x65x114x3x32x2x2x4xf32>) outs(%extracted_slice_71 : tensor<2x32x65x2x114x2x3x4xf32>) permutation = [0, 4, 1, 5, 2, 6, 3, 7]  {tag = "operation_11"}
        %inserted_slice = tensor.insert_slice %transposed_72 into %arg6[%arg3, 0, 0, 0, 0, 0, 0, %arg5] [2, 32, 65, 2, 114, 2, 3, 4] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x32x65x2x114x2x3x4xf32> into tensor<4x32x65x2x114x2x3x64xf32>
        scf.yield %inserted_slice : tensor<4x32x65x2x114x2x3x64xf32>
      }
      scf.yield %42 : tensor<4x32x65x2x114x2x3x64xf32>
    }
    %collapsed_17 = tensor.collapse_shape %12 [[0, 1], [2, 3], [4, 5], [6, 7]] : tensor<4x32x65x2x114x2x3x64xf32> into tensor<128x130x228x192xf32>
    %extracted_slice_18 = tensor.extract_slice %collapsed_17[0, 0, 0, 0] [128, 130, 228, 192] [1, 1, 1, 1] : tensor<128x130x228x192xf32> to tensor<128x130x228x192xf32>
    %13 = tensor.empty() : tensor<2x130x114x12x64x1x2x16xf32>
    %cst_19 = arith.constant 0.000000e+00 : f32
    %padded_20 = tensor.pad %extracted_slice_18 low[0, 0, 0, 0] high[0, 0, 0, 0] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst_19 : f32
    } : tensor<128x130x228x192xf32> to tensor<128x130x228x192xf32>
    %expanded_21 = tensor.expand_shape %padded_20 [[0, 1], [2, 3], [4, 5], [6, 7]] output_shape [2, 64, 130, 1, 114, 2, 12, 16] : tensor<128x130x228x192xf32> into tensor<2x64x130x1x114x2x12x16xf32>
    %c2_22 = arith.constant 2 : index
    %c114_23 = arith.constant 114 : index
    %c12 = arith.constant 12 : index
    %c16_24 = arith.constant 16 : index
    %c2_25 = arith.constant 2 : index
    %14 = scf.forall (%arg3, %arg4, %arg5, %arg6, %arg7) in (2, 114, 12, 16, 2) shared_outs(%arg8 = %13) -> (tensor<2x130x114x12x64x1x2x16xf32>) {
      %42 = affine.apply #map3(%arg6)
      %43 = affine.apply #map10(%arg7)
      %44 = affine.apply #map3(%arg6)
      %45 = affine.apply #map10(%arg7)
      %46 = affine.apply #map3(%arg6)
      %47 = affine.apply #map10(%arg7)
      %extracted_slice_67 = tensor.extract_slice %expanded_21[%arg3, %44, 0, 0, %arg4, 0, %arg5, %45] [1, 4, 130, 1, 1, 2, 1, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x64x130x1x114x2x12x16xf32> to tensor<1x4x130x1x1x2x1x8xf32>
      %extracted_slice_68 = tensor.extract_slice %arg8[%arg3, 0, %arg4, %arg5, %46, 0, 0, %47] [1, 130, 1, 1, 4, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x130x114x12x64x1x2x16xf32> to tensor<1x130x1x1x4x1x2x8xf32>
      %c0_69 = arith.constant 0 : index
      %c130_70 = arith.constant 130 : index
      %c1_71 = arith.constant 1 : index
      %48 = scf.for %arg9 = %c0_69 to %c130_70 step %c1_71 iter_args(%arg10 = %extracted_slice_68) -> (tensor<1x130x1x1x4x1x2x8xf32>) {
        %c0_72 = arith.constant 0 : index
        %c8_73 = arith.constant 8 : index
        %c2_74 = arith.constant 2 : index
        %51 = scf.for %arg11 = %c0_72 to %c8_73 step %c2_74 iter_args(%arg12 = %arg10) -> (tensor<1x130x1x1x4x1x2x8xf32>) {
          %extracted_slice_75 = tensor.extract_slice %extracted_slice_67[0, 0, %arg9, 0, 0, 0, 0, %arg11] [1, 4, 1, 1, 1, 2, 1, 2] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x4x130x1x1x2x1x8xf32> to tensor<1x4x1x1x1x2x1x2xf32>
          %extracted_slice_76 = tensor.extract_slice %arg12[0, %arg9, 0, 0, 0, 0, 0, %arg11] [1, 1, 1, 1, 4, 1, 2, 2] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x130x1x1x4x1x2x8xf32> to tensor<1x1x1x1x4x1x2x2xf32>
          %c0_77 = arith.constant 0 : index
          %c4_78 = arith.constant 4 : index
          %c1_79 = arith.constant 1 : index
          %52 = scf.for %arg13 = %c0_77 to %c4_78 step %c1_79 iter_args(%arg14 = %extracted_slice_76) -> (tensor<1x1x1x1x4x1x2x2xf32>) {
            %c0_80 = arith.constant 0 : index
            %c2_81 = arith.constant 2 : index
            %c1_82 = arith.constant 1 : index
            %53 = scf.for %arg15 = %c0_80 to %c2_81 step %c1_82 iter_args(%arg16 = %arg14) -> (tensor<1x1x1x1x4x1x2x2xf32>) {
              %extracted_slice_83 = tensor.extract_slice %extracted_slice_75[0, %arg13, 0, 0, 0, 0, 0, %arg15] [1, 1, 1, 1, 1, 2, 1, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x4x1x1x1x2x1x2xf32> to tensor<1x1x1x1x1x2x1x1xf32>
              %extracted_slice_84 = tensor.extract_slice %arg16[0, 0, 0, 0, %arg13, 0, 0, %arg15] [1, 1, 1, 1, 1, 1, 2, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x4x1x2x2xf32> to tensor<1x1x1x1x1x1x2x1xf32>
              %c1_85 = arith.constant 1 : index
              %c1_86 = arith.constant 1 : index
              %c1_87 = arith.constant 1 : index
              %c1_88 = arith.constant 1 : index
              %c1_89 = arith.constant 1 : index
              %c1_90 = arith.constant 1 : index
              %c2_91 = arith.constant 2 : index
              %c1_92 = arith.constant 1 : index
              %c0_93 = arith.constant 0 : index
              %cst_94 = arith.constant 0.000000e+00 : f32
              %54 = vector.transfer_read %extracted_slice_83[%c0_93, %c0_93, %c0_93, %c0_93, %c0_93, %c0_93, %c0_93, %c0_93], %cst_94 {permutation_map = #map11} : tensor<1x1x1x1x1x2x1x1xf32>, vector<1x1x1x1x1x1x2x1xf32>
              %cst_95 = arith.constant 0.000000e+00 : f32
              %55 = vector.transfer_read %extracted_slice_84[%c0_93, %c0_93, %c0_93, %c0_93, %c0_93, %c0_93, %c0_93, %c0_93], %cst_95 : tensor<1x1x1x1x1x1x2x1xf32>, vector<1x1x1x1x1x1x2x1xf32>
              %c0_96 = arith.constant 0 : index
              %56 = vector.transfer_write %54, %extracted_slice_84[%c0_96, %c0_96, %c0_96, %c0_96, %c0_96, %c0_96, %c0_96, %c0_96] : vector<1x1x1x1x1x1x2x1xf32>, tensor<1x1x1x1x1x1x2x1xf32>
              %inserted_slice_97 = tensor.insert_slice %56 into %arg16[0, 0, 0, 0, %arg13, 0, 0, %arg15] [1, 1, 1, 1, 1, 1, 2, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x1x1x2x1xf32> into tensor<1x1x1x1x4x1x2x2xf32>
              scf.yield %inserted_slice_97 : tensor<1x1x1x1x4x1x2x2xf32>
            }
            scf.yield %53 : tensor<1x1x1x1x4x1x2x2xf32>
          }
          %inserted_slice = tensor.insert_slice %52 into %arg12[0, %arg9, 0, 0, 0, 0, 0, %arg11] [1, 1, 1, 1, 4, 1, 2, 2] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x4x1x2x2xf32> into tensor<1x130x1x1x4x1x2x8xf32>
          scf.yield %inserted_slice : tensor<1x130x1x1x4x1x2x8xf32>
        }
        scf.yield %51 : tensor<1x130x1x1x4x1x2x8xf32>
      }
      %49 = affine.apply #map3(%arg6)
      %50 = affine.apply #map10(%arg7)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %48 into %arg8[%arg3, 0, %arg4, %arg5, %49, 0, 0, %50] [1, 130, 1, 1, 4, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x130x1x1x4x1x2x8xf32> into tensor<2x130x114x12x64x1x2x16xf32>
      }
    }
    %15 = tensor.empty() : tensor<2x130x114x12x64x1x2x16xf32>
    %cst_26 = arith.constant 0.000000e+00 : f32
    %padded_27 = tensor.pad %arg2 low[0, 0, 0, 0] high[0, 0, 0, 0] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst_26 : f32
    } : tensor<128x130x228x192xf32> to tensor<128x130x228x192xf32>
    %expanded_28 = tensor.expand_shape %padded_27 [[0, 1], [2, 3], [4, 5], [6, 7]] output_shape [2, 64, 130, 1, 114, 2, 12, 16] : tensor<128x130x228x192xf32> into tensor<2x64x130x1x114x2x12x16xf32>
    %c0_29 = arith.constant 0 : index
    %c16_30 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %16 = scf.for %arg3 = %c0_29 to %c16_30 step %c1 iter_args(%arg4 = %15) -> (tensor<2x130x114x12x64x1x2x16xf32>) {
      %c0_67 = arith.constant 0 : index
      %c2_68 = arith.constant 2 : index
      %c1_69 = arith.constant 1 : index
      %42 = scf.for %arg5 = %c0_67 to %c2_68 step %c1_69 iter_args(%arg6 = %arg4) -> (tensor<2x130x114x12x64x1x2x16xf32>) {
        %c0_70 = arith.constant 0 : index
        %c64_71 = arith.constant 64 : index
        %c2_72 = arith.constant 2 : index
        %43 = scf.for %arg7 = %c0_70 to %c64_71 step %c2_72 iter_args(%arg8 = %arg6) -> (tensor<2x130x114x12x64x1x2x16xf32>) {
          %c0_73 = arith.constant 0 : index
          %c12_74 = arith.constant 12 : index
          %c4_75 = arith.constant 4 : index
          %44 = scf.for %arg9 = %c0_73 to %c12_74 step %c4_75 iter_args(%arg10 = %arg8) -> (tensor<2x130x114x12x64x1x2x16xf32>) {
            %c0_76 = arith.constant 0 : index
            %c130_77 = arith.constant 130 : index
            %c2_78 = arith.constant 2 : index
            %45 = scf.for %arg11 = %c0_76 to %c130_77 step %c2_78 iter_args(%arg12 = %arg10) -> (tensor<2x130x114x12x64x1x2x16xf32>) {
              %extracted_slice_79 = tensor.extract_slice %expanded_28[0, %arg7, %arg11, 0, 0, %arg5, %arg9, %arg3] [2, 2, 2, 1, 114, 1, 4, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x64x130x1x114x2x12x16xf32> to tensor<2x2x2x1x114x1x4x1xf32>
              %extracted_slice_80 = tensor.extract_slice %arg12[0, %arg11, 0, %arg9, %arg7, 0, %arg5, %arg3] [2, 2, 114, 4, 2, 1, 1, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x130x114x12x64x1x2x16xf32> to tensor<2x2x114x4x2x1x1x1xf32>
              %46 = linalg.generic {indexing_maps = [#map12, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_79 : tensor<2x2x2x1x114x1x4x1xf32>) outs(%extracted_slice_80 : tensor<2x2x114x4x2x1x1x1xf32>) attrs =  {tag = "operation_14"} {
              ^bb0(%in: f32, %out: f32):
                linalg.yield %in : f32
              } -> tensor<2x2x114x4x2x1x1x1xf32>
              %inserted_slice = tensor.insert_slice %46 into %arg12[0, %arg11, 0, %arg9, %arg7, 0, %arg5, %arg3] [2, 2, 114, 4, 2, 1, 1, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x2x114x4x2x1x1x1xf32> into tensor<2x130x114x12x64x1x2x16xf32>
              scf.yield %inserted_slice : tensor<2x130x114x12x64x1x2x16xf32>
            }
            scf.yield %45 : tensor<2x130x114x12x64x1x2x16xf32>
          }
          scf.yield %44 : tensor<2x130x114x12x64x1x2x16xf32>
        }
        scf.yield %43 : tensor<2x130x114x12x64x1x2x16xf32>
      }
      scf.yield %42 : tensor<2x130x114x12x64x1x2x16xf32>
    }
    %c0_31 = arith.constant 0 : index
    %c130 = arith.constant 130 : index
    %c1_32 = arith.constant 1 : index
    %17 = scf.for %arg3 = %c0_31 to %c130 step %c1_32 iter_args(%arg4 = %16) -> (tensor<2x130x114x12x64x1x2x16xf32>) {
      %c0_67 = arith.constant 0 : index
      %c12_68 = arith.constant 12 : index
      %c2_69 = arith.constant 2 : index
      %42 = scf.for %arg5 = %c0_67 to %c12_68 step %c2_69 iter_args(%arg6 = %arg4) -> (tensor<2x130x114x12x64x1x2x16xf32>) {
        %extracted_slice_70 = tensor.extract_slice %14[0, %arg3, 0, %arg5, 0, 0, 0, 0] [2, 1, 114, 2, 64, 1, 2, 16] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x130x114x12x64x1x2x16xf32> to tensor<2x1x114x2x64x1x2x16xf32>
        %extracted_slice_71 = tensor.extract_slice %arg6[0, %arg3, 0, %arg5, 0, 0, 0, 0] [2, 1, 114, 2, 64, 1, 2, 16] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x130x114x12x64x1x2x16xf32> to tensor<2x1x114x2x64x1x2x16xf32>
        %c0_72 = arith.constant 0 : index
        %c2_73 = arith.constant 2 : index
        %c1_74 = arith.constant 1 : index
        %43 = scf.for %arg7 = %c0_72 to %c2_73 step %c1_74 iter_args(%arg8 = %extracted_slice_71) -> (tensor<2x1x114x2x64x1x2x16xf32>) {
          %c0_75 = arith.constant 0 : index
          %c64_76 = arith.constant 64 : index
          %c1_77 = arith.constant 1 : index
          %44 = scf.for %arg9 = %c0_75 to %c64_76 step %c1_77 iter_args(%arg10 = %arg8) -> (tensor<2x1x114x2x64x1x2x16xf32>) {
            %c0_78 = arith.constant 0 : index
            %c16_79 = arith.constant 16 : index
            %c1_80 = arith.constant 1 : index
            %45 = scf.for %arg11 = %c0_78 to %c16_79 step %c1_80 iter_args(%arg12 = %arg10) -> (tensor<2x1x114x2x64x1x2x16xf32>) {
              %extracted_slice_81 = tensor.extract_slice %extracted_slice_70[0, 0, 0, %arg7, %arg9, 0, 0, %arg11] [2, 1, 114, 1, 1, 1, 2, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x1x114x2x64x1x2x16xf32> to tensor<2x1x114x1x1x1x2x1xf32>
              %extracted_slice_82 = tensor.extract_slice %arg12[0, 0, 0, %arg7, %arg9, 0, 0, %arg11] [2, 1, 114, 1, 1, 1, 2, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x1x114x2x64x1x2x16xf32> to tensor<2x1x114x1x1x1x2x1xf32>
              %c2_83 = arith.constant 2 : index
              %c1_84 = arith.constant 1 : index
              %c114_85 = arith.constant 114 : index
              %c1_86 = arith.constant 1 : index
              %c1_87 = arith.constant 1 : index
              %c1_88 = arith.constant 1 : index
              %c2_89 = arith.constant 2 : index
              %c1_90 = arith.constant 1 : index
              %c0_91 = arith.constant 0 : index
              %cst_92 = arith.constant 0.000000e+00 : f32
              %46 = vector.transfer_read %extracted_slice_81[%c0_91, %c0_91, %c0_91, %c0_91, %c0_91, %c0_91, %c0_91, %c0_91], %cst_92 : tensor<2x1x114x1x1x1x2x1xf32>, vector<2x1x114x1x1x1x2x1xf32>
              %cst_93 = arith.constant 0.000000e+00 : f32
              %47 = vector.transfer_read %extracted_slice_82[%c0_91, %c0_91, %c0_91, %c0_91, %c0_91, %c0_91, %c0_91, %c0_91], %cst_93 : tensor<2x1x114x1x1x1x2x1xf32>, vector<2x1x114x1x1x1x2x1xf32>
              %c0_94 = arith.constant 0 : index
              %48 = vector.transfer_write %46, %extracted_slice_82[%c0_94, %c0_94, %c0_94, %c0_94, %c0_94, %c0_94, %c0_94, %c0_94] : vector<2x1x114x1x1x1x2x1xf32>, tensor<2x1x114x1x1x1x2x1xf32>
              %inserted_slice_95 = tensor.insert_slice %48 into %arg12[0, 0, 0, %arg7, %arg9, 0, 0, %arg11] [2, 1, 114, 1, 1, 1, 2, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x1x114x1x1x1x2x1xf32> into tensor<2x1x114x2x64x1x2x16xf32>
              scf.yield %inserted_slice_95 : tensor<2x1x114x2x64x1x2x16xf32>
            }
            scf.yield %45 : tensor<2x1x114x2x64x1x2x16xf32>
          }
          scf.yield %44 : tensor<2x1x114x2x64x1x2x16xf32>
        }
        %inserted_slice = tensor.insert_slice %43 into %arg6[0, %arg3, 0, %arg5, 0, 0, 0, 0] [2, 1, 114, 2, 64, 1, 2, 16] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x1x114x2x64x1x2x16xf32> into tensor<2x130x114x12x64x1x2x16xf32>
        scf.yield %inserted_slice : tensor<2x130x114x12x64x1x2x16xf32>
      }
      scf.yield %42 : tensor<2x130x114x12x64x1x2x16xf32>
    }
    %18 = tensor.empty() : tensor<2x64x130x1x114x2x12x16xf32>
    %c64 = arith.constant 64 : index
    %c114_33 = arith.constant 114 : index
    %c2_34 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %c16_35 = arith.constant 16 : index
    %19 = scf.forall (%arg3, %arg4, %arg5, %arg6, %arg7) in (64, 114, 2, 6, 16) shared_outs(%arg8 = %18) -> (tensor<2x64x130x1x114x2x12x16xf32>) {
      %42 = affine.apply #map4(%arg6)
      %43 = affine.apply #map4(%arg6)
      %44 = affine.apply #map4(%arg6)
      %extracted_slice_67 = tensor.extract_slice %17[0, 0, %arg4, %43, %arg3, 0, %arg5, %arg7] [2, 130, 1, 2, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x130x114x12x64x1x2x16xf32> to tensor<2x130x1x2x1x1x1x1xf32>
      %extracted_slice_68 = tensor.extract_slice %arg8[0, %arg3, 0, 0, %arg4, %arg5, %44, %arg7] [2, 1, 130, 1, 1, 1, 2, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x64x130x1x114x2x12x16xf32> to tensor<2x1x130x1x1x1x2x1xf32>
      %c0_69 = arith.constant 0 : index
      %c2_70 = arith.constant 2 : index
      %c1_71 = arith.constant 1 : index
      %45 = scf.for %arg9 = %c0_69 to %c2_70 step %c1_71 iter_args(%arg10 = %extracted_slice_68) -> (tensor<2x1x130x1x1x1x2x1xf32>) {
        %c0_72 = arith.constant 0 : index
        %c130_73 = arith.constant 130 : index
        %c1_74 = arith.constant 1 : index
        %47 = scf.for %arg11 = %c0_72 to %c130_73 step %c1_74 iter_args(%arg12 = %arg10) -> (tensor<2x1x130x1x1x1x2x1xf32>) {
          %extracted_slice_75 = tensor.extract_slice %extracted_slice_67[%arg9, %arg11, 0, 0, 0, 0, 0, 0] [1, 1, 1, 2, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x130x1x2x1x1x1x1xf32> to tensor<1x1x1x2x1x1x1x1xf32>
          %extracted_slice_76 = tensor.extract_slice %arg12[%arg9, 0, %arg11, 0, 0, 0, 0, 0] [1, 1, 1, 1, 1, 1, 2, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x1x130x1x1x1x2x1xf32> to tensor<1x1x1x1x1x1x2x1xf32>
          %c1_77 = arith.constant 1 : index
          %c1_78 = arith.constant 1 : index
          %c1_79 = arith.constant 1 : index
          %c1_80 = arith.constant 1 : index
          %c2_81 = arith.constant 2 : index
          %c1_82 = arith.constant 1 : index
          %c1_83 = arith.constant 1 : index
          %c1_84 = arith.constant 1 : index
          %c0_85 = arith.constant 0 : index
          %cst_86 = arith.constant 0.000000e+00 : f32
          %48 = vector.transfer_read %extracted_slice_75[%c0_85, %c0_85, %c0_85, %c0_85, %c0_85, %c0_85, %c0_85, %c0_85], %cst_86 {permutation_map = #map14} : tensor<1x1x1x2x1x1x1x1xf32>, vector<1x1x1x1x2x1x1x1xf32>
          %cst_87 = arith.constant 0.000000e+00 : f32
          %49 = vector.transfer_read %extracted_slice_76[%c0_85, %c0_85, %c0_85, %c0_85, %c0_85, %c0_85, %c0_85, %c0_85], %cst_87 {permutation_map = #map15} : tensor<1x1x1x1x1x1x2x1xf32>, vector<1x1x1x1x2x1x1x1xf32>
          %c0_88 = arith.constant 0 : index
          %50 = vector.transfer_write %48, %extracted_slice_76[%c0_88, %c0_88, %c0_88, %c0_88, %c0_88, %c0_88, %c0_88, %c0_88] {permutation_map = #map15} : vector<1x1x1x1x2x1x1x1xf32>, tensor<1x1x1x1x1x1x2x1xf32>
          %inserted_slice = tensor.insert_slice %50 into %arg12[%arg9, 0, %arg11, 0, 0, 0, 0, 0] [1, 1, 1, 1, 1, 1, 2, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x1x1x2x1xf32> into tensor<2x1x130x1x1x1x2x1xf32>
          scf.yield %inserted_slice : tensor<2x1x130x1x1x1x2x1xf32>
        }
        scf.yield %47 : tensor<2x1x130x1x1x1x2x1xf32>
      }
      %46 = affine.apply #map4(%arg6)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %45 into %arg8[0, %arg3, 0, 0, %arg4, %arg5, %46, %arg7] [2, 1, 130, 1, 1, 1, 2, 1] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<2x1x130x1x1x1x2x1xf32> into tensor<2x64x130x1x114x2x12x16xf32>
      }
    }
    %collapsed_36 = tensor.collapse_shape %19 [[0, 1], [2, 3], [4, 5], [6, 7]] : tensor<2x64x130x1x114x2x12x16xf32> into tensor<128x130x228x192xf32>
    %extracted_slice_37 = tensor.extract_slice %collapsed_36[0, 0, 0, 0] [128, 130, 228, 192] [1, 1, 1, 1] : tensor<128x130x228x192xf32> to tensor<128x130x228x192xf32>
    %20 = tensor.empty() : tensor<8x65x228x6x16x2x32xf32>
    %cst_38 = arith.constant 0.000000e+00 : f32
    %padded_39 = tensor.pad %extracted_slice_37 low[0, 0, 0, 0] high[0, 0, 0, 0] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst_38 : f32
    } : tensor<128x130x228x192xf32> to tensor<128x130x228x192xf32>
    %expanded_40 = tensor.expand_shape %padded_39 [[0, 1], [2, 3], [4], [5, 6]] output_shape [8, 16, 65, 2, 228, 6, 32] : tensor<128x130x228x192xf32> into tensor<8x16x65x2x228x6x32xf32>
    %c0_41 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1_42 = arith.constant 1 : index
    %21 = scf.for %arg3 = %c0_41 to %c8 step %c1_42 iter_args(%arg4 = %20) -> (tensor<8x65x228x6x16x2x32xf32>) {
      %c0_67 = arith.constant 0 : index
      %c65_68 = arith.constant 65 : index
      %c1_69 = arith.constant 1 : index
      %42 = scf.for %arg5 = %c0_67 to %c65_68 step %c1_69 iter_args(%arg6 = %arg4) -> (tensor<8x65x228x6x16x2x32xf32>) {
        %c0_70 = arith.constant 0 : index
        %c228 = arith.constant 228 : index
        %c1_71 = arith.constant 1 : index
        %43 = scf.for %arg7 = %c0_70 to %c228 step %c1_71 iter_args(%arg8 = %arg6) -> (tensor<8x65x228x6x16x2x32xf32>) {
          %c0_72 = arith.constant 0 : index
          %c6_73 = arith.constant 6 : index
          %c2_74 = arith.constant 2 : index
          %44 = scf.for %arg9 = %c0_72 to %c6_73 step %c2_74 iter_args(%arg10 = %arg8) -> (tensor<8x65x228x6x16x2x32xf32>) {
            %extracted_slice_75 = tensor.extract_slice %expanded_40[%arg3, 0, %arg5, 0, %arg7, %arg9, 0] [1, 16, 1, 2, 1, 2, 32] [1, 1, 1, 1, 1, 1, 1] : tensor<8x16x65x2x228x6x32xf32> to tensor<1x16x1x2x1x2x32xf32>
            %extracted_slice_76 = tensor.extract_slice %arg10[%arg3, %arg5, %arg7, %arg9, 0, 0, 0] [1, 1, 1, 2, 16, 2, 32] [1, 1, 1, 1, 1, 1, 1] : tensor<8x65x228x6x16x2x32xf32> to tensor<1x1x1x2x16x2x32xf32>
            %c0_77 = arith.constant 0 : index
            %c2_78 = arith.constant 2 : index
            %c1_79 = arith.constant 1 : index
            %45 = scf.for %arg11 = %c0_77 to %c2_78 step %c1_79 iter_args(%arg12 = %extracted_slice_76) -> (tensor<1x1x1x2x16x2x32xf32>) {
              %c0_80 = arith.constant 0 : index
              %c32 = arith.constant 32 : index
              %c1_81 = arith.constant 1 : index
              %46 = scf.for %arg13 = %c0_80 to %c32 step %c1_81 iter_args(%arg14 = %arg12) -> (tensor<1x1x1x2x16x2x32xf32>) {
                %extracted_slice_82 = tensor.extract_slice %extracted_slice_75[0, 0, 0, 0, 0, %arg11, %arg13] [1, 16, 1, 2, 1, 1, 1] [1, 1, 1, 1, 1, 1, 1] : tensor<1x16x1x2x1x2x32xf32> to tensor<1x16x1x2x1x1x1xf32>
                %extracted_slice_83 = tensor.extract_slice %arg14[0, 0, 0, %arg11, 0, 0, %arg13] [1, 1, 1, 1, 16, 2, 1] [1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x2x16x2x32xf32> to tensor<1x1x1x1x16x2x1xf32>
                %c1_84 = arith.constant 1 : index
                %c1_85 = arith.constant 1 : index
                %c1_86 = arith.constant 1 : index
                %c1_87 = arith.constant 1 : index
                %c16_88 = arith.constant 16 : index
                %c1_89 = arith.constant 1 : index
                %c2_90 = arith.constant 2 : index
                %c0_91 = arith.constant 0 : index
                %cst_92 = arith.constant 0.000000e+00 : f32
                %47 = vector.transfer_read %extracted_slice_82[%c0_91, %c0_91, %c0_91, %c0_91, %c0_91, %c0_91, %c0_91], %cst_92 {permutation_map = #map16} : tensor<1x16x1x2x1x1x1xf32>, vector<1x1x1x1x16x1x2xf32>
                %cst_93 = arith.constant 0.000000e+00 : f32
                %48 = vector.transfer_read %extracted_slice_83[%c0_91, %c0_91, %c0_91, %c0_91, %c0_91, %c0_91, %c0_91], %cst_93 {permutation_map = #map17} : tensor<1x1x1x1x16x2x1xf32>, vector<1x1x1x1x16x1x2xf32>
                %c0_94 = arith.constant 0 : index
                %49 = vector.transfer_write %47, %extracted_slice_83[%c0_94, %c0_94, %c0_94, %c0_94, %c0_94, %c0_94, %c0_94] {permutation_map = #map17} : vector<1x1x1x1x16x1x2xf32>, tensor<1x1x1x1x16x2x1xf32>
                %inserted_slice_95 = tensor.insert_slice %49 into %arg14[0, 0, 0, %arg11, 0, 0, %arg13] [1, 1, 1, 1, 16, 2, 1] [1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x16x2x1xf32> into tensor<1x1x1x2x16x2x32xf32>
                scf.yield %inserted_slice_95 : tensor<1x1x1x2x16x2x32xf32>
              }
              scf.yield %46 : tensor<1x1x1x2x16x2x32xf32>
            }
            %inserted_slice = tensor.insert_slice %45 into %arg10[%arg3, %arg5, %arg7, %arg9, 0, 0, 0] [1, 1, 1, 2, 16, 2, 32] [1, 1, 1, 1, 1, 1, 1] : tensor<1x1x1x2x16x2x32xf32> into tensor<8x65x228x6x16x2x32xf32>
            scf.yield %inserted_slice : tensor<8x65x228x6x16x2x32xf32>
          }
          scf.yield %44 : tensor<8x65x228x6x16x2x32xf32>
        }
        scf.yield %43 : tensor<8x65x228x6x16x2x32xf32>
      }
      scf.yield %42 : tensor<8x65x228x6x16x2x32xf32>
    }
    %22 = tensor.empty() : tensor<8x65x228x6x16x2x32xf32>
    %cst_43 = arith.constant 0.000000e+00 : f32
    %padded_44 = tensor.pad %arg2 low[0, 0, 0, 0] high[0, 0, 0, 0] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst_43 : f32
    } : tensor<128x130x228x192xf32> to tensor<128x130x228x192xf32>
    %expanded_45 = tensor.expand_shape %padded_44 [[0, 1], [2, 3], [4], [5, 6]] output_shape [8, 16, 65, 2, 228, 6, 32] : tensor<128x130x228x192xf32> into tensor<8x16x65x2x228x6x32xf32>
    %c8_46 = arith.constant 8 : index
    %c57 = arith.constant 57 : index
    %c6_47 = arith.constant 6 : index
    %c4_48 = arith.constant 4 : index
    %c2_49 = arith.constant 2 : index
    %23 = scf.forall (%arg3, %arg4, %arg5, %arg6, %arg7) in (8, 57, 6, 4, 2) shared_outs(%arg8 = %22) -> (tensor<8x65x228x6x16x2x32xf32>) {
      %42 = affine.apply #map3(%arg4)
      %43 = affine.apply #map3(%arg6)
      %44 = affine.apply #map3(%arg6)
      %45 = affine.apply #map3(%arg4)
      %46 = affine.apply #map3(%arg4)
      %47 = affine.apply #map3(%arg6)
      %extracted_slice_67 = tensor.extract_slice %expanded_45[%arg3, %44, 0, %arg7, %45, %arg5, 0] [1, 4, 65, 1, 4, 1, 32] [1, 1, 1, 1, 1, 1, 1] : tensor<8x16x65x2x228x6x32xf32> to tensor<1x4x65x1x4x1x32xf32>
      %extracted_slice_68 = tensor.extract_slice %arg8[%arg3, 0, %46, %arg5, %47, %arg7, 0] [1, 65, 4, 1, 4, 1, 32] [1, 1, 1, 1, 1, 1, 1] : tensor<8x65x228x6x16x2x32xf32> to tensor<1x65x4x1x4x1x32xf32>
      %transposed_69 = linalg.transpose ins(%extracted_slice_67 : tensor<1x4x65x1x4x1x32xf32>) outs(%extracted_slice_68 : tensor<1x65x4x1x4x1x32xf32>) permutation = [0, 2, 4, 5, 1, 3, 6]  {tag = "operation_18"}
      %48 = affine.apply #map3(%arg4)
      %49 = affine.apply #map3(%arg6)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %transposed_69 into %arg8[%arg3, 0, %48, %arg5, %49, %arg7, 0] [1, 65, 4, 1, 4, 1, 32] [1, 1, 1, 1, 1, 1, 1] : tensor<1x65x4x1x4x1x32xf32> into tensor<8x65x228x6x16x2x32xf32>
      }
    }
    %c0_50 = arith.constant 0 : index
    %c8_51 = arith.constant 8 : index
    %c2_52 = arith.constant 2 : index
    %24 = scf.for %arg3 = %c0_50 to %c8_51 step %c2_52 iter_args(%arg4 = %23) -> (tensor<8x65x228x6x16x2x32xf32>) {
      %c0_67 = arith.constant 0 : index
      %c228 = arith.constant 228 : index
      %c4_68 = arith.constant 4 : index
      %42 = scf.for %arg5 = %c0_67 to %c228 step %c4_68 iter_args(%arg6 = %arg4) -> (tensor<8x65x228x6x16x2x32xf32>) {
        %c0_69 = arith.constant 0 : index
        %c6_70 = arith.constant 6 : index
        %c2_71 = arith.constant 2 : index
        %43 = scf.for %arg7 = %c0_69 to %c6_70 step %c2_71 iter_args(%arg8 = %arg6) -> (tensor<8x65x228x6x16x2x32xf32>) {
          %c0_72 = arith.constant 0 : index
          %c2_73 = arith.constant 2 : index
          %c1_74 = arith.constant 1 : index
          %44 = scf.for %arg9 = %c0_72 to %c2_73 step %c1_74 iter_args(%arg10 = %arg8) -> (tensor<8x65x228x6x16x2x32xf32>) {
            %c0_75 = arith.constant 0 : index
            %c32 = arith.constant 32 : index
            %c2_76 = arith.constant 2 : index
            %45 = scf.for %arg11 = %c0_75 to %c32 step %c2_76 iter_args(%arg12 = %arg10) -> (tensor<8x65x228x6x16x2x32xf32>) {
              %extracted_slice_77 = tensor.extract_slice %21[%arg3, 0, %arg5, %arg7, 0, %arg9, %arg11] [2, 65, 4, 2, 16, 1, 2] [1, 1, 1, 1, 1, 1, 1] : tensor<8x65x228x6x16x2x32xf32> to tensor<2x65x4x2x16x1x2xf32>
              %extracted_slice_78 = tensor.extract_slice %arg12[%arg3, 0, %arg5, %arg7, 0, %arg9, %arg11] [2, 65, 4, 2, 16, 1, 2] [1, 1, 1, 1, 1, 1, 1] : tensor<8x65x228x6x16x2x32xf32> to tensor<2x65x4x2x16x1x2xf32>
              %c0_79 = arith.constant 0 : index
              %c65_80 = arith.constant 65 : index
              %c1_81 = arith.constant 1 : index
              %46 = scf.for %arg13 = %c0_79 to %c65_80 step %c1_81 iter_args(%arg14 = %extracted_slice_78) -> (tensor<2x65x4x2x16x1x2xf32>) {
                %c0_82 = arith.constant 0 : index
                %c2_83 = arith.constant 2 : index
                %c1_84 = arith.constant 1 : index
                %47 = scf.for %arg15 = %c0_82 to %c2_83 step %c1_84 iter_args(%arg16 = %arg14) -> (tensor<2x65x4x2x16x1x2xf32>) {
                  %c0_85 = arith.constant 0 : index
                  %c16_86 = arith.constant 16 : index
                  %c2_87 = arith.constant 2 : index
                  %48 = scf.for %arg17 = %c0_85 to %c16_86 step %c2_87 iter_args(%arg18 = %arg16) -> (tensor<2x65x4x2x16x1x2xf32>) {
                    %extracted_slice_88 = tensor.extract_slice %extracted_slice_77[0, %arg13, 0, 0, %arg17, 0, %arg15] [2, 1, 4, 2, 2, 1, 1] [1, 1, 1, 1, 1, 1, 1] : tensor<2x65x4x2x16x1x2xf32> to tensor<2x1x4x2x2x1x1xf32>
                    %extracted_slice_89 = tensor.extract_slice %arg18[0, %arg13, 0, 0, %arg17, 0, %arg15] [2, 1, 4, 2, 2, 1, 1] [1, 1, 1, 1, 1, 1, 1] : tensor<2x65x4x2x16x1x2xf32> to tensor<2x1x4x2x2x1x1xf32>
                    %c1_90 = arith.constant 1 : index
                    %c4_91 = arith.constant 4 : index
                    %c1_92 = arith.constant 1 : index
                    %c1_93 = arith.constant 1 : index
                    %c2_94 = arith.constant 2 : index
                    %c2_95 = arith.constant 2 : index
                    %c2_96 = arith.constant 2 : index
                    %c0_97 = arith.constant 0 : index
                    %cst_98 = arith.constant 0.000000e+00 : f32
                    %49 = vector.transfer_read %extracted_slice_88[%c0_97, %c0_97, %c0_97, %c0_97, %c0_97, %c0_97, %c0_97], %cst_98 {permutation_map = #map18} : tensor<2x1x4x2x2x1x1xf32>, vector<1x4x1x1x2x2x2xf32>
                    %cst_99 = arith.constant 0.000000e+00 : f32
                    %50 = vector.transfer_read %extracted_slice_89[%c0_97, %c0_97, %c0_97, %c0_97, %c0_97, %c0_97, %c0_97], %cst_99 {permutation_map = #map18} : tensor<2x1x4x2x2x1x1xf32>, vector<1x4x1x1x2x2x2xf32>
                    %c0_100 = arith.constant 0 : index
                    %51 = vector.transfer_write %49, %extracted_slice_89[%c0_100, %c0_100, %c0_100, %c0_100, %c0_100, %c0_100, %c0_100] {permutation_map = #map18} : vector<1x4x1x1x2x2x2xf32>, tensor<2x1x4x2x2x1x1xf32>
                    %inserted_slice_101 = tensor.insert_slice %51 into %arg18[0, %arg13, 0, 0, %arg17, 0, %arg15] [2, 1, 4, 2, 2, 1, 1] [1, 1, 1, 1, 1, 1, 1] : tensor<2x1x4x2x2x1x1xf32> into tensor<2x65x4x2x16x1x2xf32>
                    scf.yield %inserted_slice_101 : tensor<2x65x4x2x16x1x2xf32>
                  }
                  scf.yield %48 : tensor<2x65x4x2x16x1x2xf32>
                }
                scf.yield %47 : tensor<2x65x4x2x16x1x2xf32>
              }
              %inserted_slice = tensor.insert_slice %46 into %arg12[%arg3, 0, %arg5, %arg7, 0, %arg9, %arg11] [2, 65, 4, 2, 16, 1, 2] [1, 1, 1, 1, 1, 1, 1] : tensor<2x65x4x2x16x1x2xf32> into tensor<8x65x228x6x16x2x32xf32>
              scf.yield %inserted_slice : tensor<8x65x228x6x16x2x32xf32>
            }
            scf.yield %45 : tensor<8x65x228x6x16x2x32xf32>
          }
          scf.yield %44 : tensor<8x65x228x6x16x2x32xf32>
        }
        scf.yield %43 : tensor<8x65x228x6x16x2x32xf32>
      }
      scf.yield %42 : tensor<8x65x228x6x16x2x32xf32>
    }
    %25 = tensor.empty() : tensor<8x16x65x2x228x6x32xf32>
    %c2_53 = arith.constant 2 : index
    %c16_54 = arith.constant 16 : index
    %c6_55 = arith.constant 6 : index
    %c4_56 = arith.constant 4 : index
    %26 = scf.forall (%arg3, %arg4, %arg5, %arg6) in (2, 16, 6, 4) shared_outs(%arg7 = %25) -> (tensor<8x16x65x2x228x6x32xf32>) {
      %42 = affine.apply #map3(%arg3)
      %43 = affine.apply #map10(%arg6)
      %44 = affine.apply #map3(%arg3)
      %45 = affine.apply #map10(%arg6)
      %46 = affine.apply #map3(%arg3)
      %47 = affine.apply #map10(%arg6)
      %extracted_slice_67 = tensor.extract_slice %24[%44, 0, 0, %arg5, %arg4, 0, %45] [4, 65, 228, 1, 1, 2, 8] [1, 1, 1, 1, 1, 1, 1] : tensor<8x65x228x6x16x2x32xf32> to tensor<4x65x228x1x1x2x8xf32>
      %extracted_slice_68 = tensor.extract_slice %arg7[%46, %arg4, 0, 0, 0, %arg5, %47] [4, 1, 65, 2, 228, 1, 8] [1, 1, 1, 1, 1, 1, 1] : tensor<8x16x65x2x228x6x32xf32> to tensor<4x1x65x2x228x1x8xf32>
      %48 = linalg.generic {indexing_maps = [#map19, #map20], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_67 : tensor<4x65x228x1x1x2x8xf32>) outs(%extracted_slice_68 : tensor<4x1x65x2x228x1x8xf32>) attrs =  {tag = "operation_19"} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<4x1x65x2x228x1x8xf32>
      %49 = affine.apply #map3(%arg3)
      %50 = affine.apply #map10(%arg6)
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %48 into %arg7[%49, %arg4, 0, 0, 0, %arg5, %50] [4, 1, 65, 2, 228, 1, 8] [1, 1, 1, 1, 1, 1, 1] : tensor<4x1x65x2x228x1x8xf32> into tensor<8x16x65x2x228x6x32xf32>
      }
    }
    %collapsed_57 = tensor.collapse_shape %26 [[0, 1], [2, 3], [4], [5, 6]] : tensor<8x16x65x2x228x6x32xf32> into tensor<128x130x228x192xf32>
    %extracted_slice_58 = tensor.extract_slice %collapsed_57[0, 0, 0, 0] [128, 130, 228, 192] [1, 1, 1, 1] : tensor<128x130x228x192xf32> to tensor<128x130x228x192xf32>
    %27 = linalg.generic {indexing_maps = [#map21, #map21], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice_58 : tensor<128x130x228x192xf32>) outs(%arg2 : tensor<128x130x228x192xf32>) attrs =  {tag = "operation_20"} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<128x130x228x192xf32>
    %28 = bufferization.alloc_tensor() : tensor<3x3xf32>
    %29 = bufferization.alloc_tensor() : tensor<128x128x226x192xf32>
    %c0_59 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c2_60 = arith.constant 2 : index
    %30 = scf.for %arg3 = %c0_59 to %c128 step %c2_60 iter_args(%arg4 = %29) -> (tensor<128x128x226x192xf32>) {
      %c0_67 = arith.constant 0 : index
      %c128_68 = arith.constant 128 : index
      %c1_69 = arith.constant 1 : index
      %42 = scf.for %arg5 = %c0_67 to %c128_68 step %c1_69 iter_args(%arg6 = %arg4) -> (tensor<128x128x226x192xf32>) {
        %c0_70 = arith.constant 0 : index
        %c226 = arith.constant 226 : index
        %c2_71 = arith.constant 2 : index
        %43 = scf.for %arg7 = %c0_70 to %c226 step %c2_71 iter_args(%arg8 = %arg6) -> (tensor<128x128x226x192xf32>) {
          %c0_72 = arith.constant 0 : index
          %c192 = arith.constant 192 : index
          %c4_73 = arith.constant 4 : index
          %44 = scf.for %arg9 = %c0_72 to %c192 step %c4_73 iter_args(%arg10 = %arg8) -> (tensor<128x128x226x192xf32>) {
            %extracted_slice_74 = tensor.extract_slice %27[%arg3, %arg5, %arg7, %arg9] [2, 3, 4, 4] [1, 1, 1, 1] : tensor<128x130x228x192xf32> to tensor<2x3x4x4xf32>
            %extracted_slice_75 = tensor.extract_slice %28[0, 0] [3, 3] [1, 1] : tensor<3x3xf32> to tensor<3x3xf32>
            %extracted_slice_76 = tensor.extract_slice %arg10[%arg3, %arg5, %arg7, %arg9] [2, 1, 2, 4] [1, 1, 1, 1] : tensor<128x128x226x192xf32> to tensor<2x1x2x4xf32>
            %c0_77 = arith.constant 0 : index
            %c1_78 = arith.constant 1 : index
            %c1_79 = arith.constant 1 : index
            %45 = scf.for %arg11 = %c0_77 to %c1_78 step %c1_79 iter_args(%arg12 = %extracted_slice_76) -> (tensor<2x1x2x4xf32>) {
              %c0_80 = arith.constant 0 : index
              %c3 = arith.constant 3 : index
              %c1_81 = arith.constant 1 : index
              %46 = scf.for %arg13 = %c0_80 to %c3 step %c1_81 iter_args(%arg14 = %arg12) -> (tensor<2x1x2x4xf32>) {
                %47 = affine.apply #map22(%arg11, %arg13)
                %extracted_slice_82 = tensor.extract_slice %extracted_slice_74[0, %47, 0, 0] [2, 1, 4, 4] [1, 1, 1, 1] : tensor<2x3x4x4xf32> to tensor<2x1x4x4xf32>
                %extracted_slice_83 = tensor.extract_slice %extracted_slice_75[%arg13, 0] [1, 3] [1, 1] : tensor<3x3xf32> to tensor<1x3xf32>
                %extracted_slice_84 = tensor.extract_slice %arg14[0, %arg11, 0, 0] [2, 1, 2, 4] [1, 1, 1, 1] : tensor<2x1x2x4xf32> to tensor<2x1x2x4xf32>
                %extracted_slice_85 = tensor.extract_slice %extracted_slice_82[0, 0, 0, 0] [2, 1, 4, 4] [1, 1, 1, 1] : tensor<2x1x4x4xf32> to tensor<2x4x4xf32>
                %extracted_slice_86 = tensor.extract_slice %extracted_slice_83[0, 0] [1, 3] [1, 1] : tensor<1x3xf32> to tensor<3xf32>
                %extracted_slice_87 = tensor.extract_slice %extracted_slice_84[0, 0, 0, 0] [2, 1, 2, 4] [1, 1, 1, 1] : tensor<2x1x2x4xf32> to tensor<2x2x4xf32>
                %c2_88 = arith.constant 2 : index
                %c2_89 = arith.constant 2 : index
                %c4_90 = arith.constant 4 : index
                %c3_91 = arith.constant 3 : index
                %c0_92 = arith.constant 0 : index
                %cst_93 = arith.constant 0.000000e+00 : f32
                %48 = vector.transfer_read %extracted_slice_85[%c0_92, %c0_92, %c0_92], %cst_93 : tensor<2x4x4xf32>, vector<2x4x4xf32>
                %cst_94 = arith.constant 0.000000e+00 : f32
                %49 = vector.transfer_read %extracted_slice_87[%c0_92, %c0_92, %c0_92], %cst_94 : tensor<2x2x4xf32>, vector<2x2x4xf32>
                %50 = vector.extract_strided_slice %48 {offsets = [0, 0, 0], sizes = [2, 2, 4], strides = [1, 1, 1]} : vector<2x4x4xf32> to vector<2x2x4xf32>
                %51 = vector.extract_strided_slice %48 {offsets = [0, 1, 0], sizes = [2, 2, 4], strides = [1, 1, 1]} : vector<2x4x4xf32> to vector<2x2x4xf32>
                %52 = vector.extract_strided_slice %48 {offsets = [0, 2, 0], sizes = [2, 2, 4], strides = [1, 1, 1]} : vector<2x4x4xf32> to vector<2x2x4xf32>
                %53 = vector.extract_strided_slice %49 {offsets = [0, 0, 0], sizes = [2, 2, 4], strides = [1, 1, 1]} : vector<2x2x4xf32> to vector<2x2x4xf32>
                %54 = arith.addf %50, %53 : vector<2x2x4xf32>
                %55 = arith.addf %51, %54 : vector<2x2x4xf32>
                %56 = arith.addf %52, %55 : vector<2x2x4xf32>
                %57 = vector.insert_strided_slice %56, %49 {offsets = [0, 0, 0], strides = [1, 1, 1]} : vector<2x2x4xf32> into vector<2x2x4xf32>
                %58 = vector.transfer_write %57, %extracted_slice_87[%c0_92, %c0_92, %c0_92] : vector<2x2x4xf32>, tensor<2x2x4xf32>
                %inserted_slice_95 = tensor.insert_slice %58 into %extracted_slice_84[0, 0, 0, 0] [2, 1, 2, 4] [1, 1, 1, 1] : tensor<2x2x4xf32> into tensor<2x1x2x4xf32>
                %inserted_slice_96 = tensor.insert_slice %inserted_slice_95 into %arg14[0, %arg11, 0, 0] [2, 1, 2, 4] [1, 1, 1, 1] : tensor<2x1x2x4xf32> into tensor<2x1x2x4xf32>
                scf.yield %inserted_slice_96 : tensor<2x1x2x4xf32>
              }
              scf.yield %46 : tensor<2x1x2x4xf32>
            }
            %inserted_slice = tensor.insert_slice %45 into %arg10[%arg3, %arg5, %arg7, %arg9] [2, 1, 2, 4] [1, 1, 1, 1] : tensor<2x1x2x4xf32> into tensor<128x128x226x192xf32>
            scf.yield %inserted_slice : tensor<128x128x226x192xf32>
          }
          scf.yield %44 : tensor<128x128x226x192xf32>
        }
        scf.yield %43 : tensor<128x128x226x192xf32>
      }
      scf.yield %42 : tensor<128x128x226x192xf32>
    }
    %31 = bufferization.alloc_tensor() : tensor<3x3xf32>
    %32 = bufferization.alloc_tensor() : tensor<128x128x112x95xf32>
    %33 = linalg.generic {indexing_maps = [#map23, #map24, #map25], iterator_types = ["parallel", "parallel", "reduction", "parallel", "reduction", "parallel"]} ins(%30, %31 : tensor<128x128x226x192xf32>, tensor<3x3xf32>) outs(%32 : tensor<128x128x112x95xf32>) attrs =  {tag = "operation_2"} {
    ^bb0(%in: f32, %in_67: f32, %out: f32):
      %42 = arith.addf %out, %in : f32
      linalg.yield %42 : f32
    } -> tensor<128x128x112x95xf32>
    %34 = bufferization.alloc_tensor() : tensor<3x3xf32>
    %35 = bufferization.alloc_tensor() : tensor<128x128x55x47xf32>
    %c0_61 = arith.constant 0 : index
    %c128_62 = arith.constant 128 : index
    %c2_63 = arith.constant 2 : index
    %36 = scf.for %arg3 = %c0_61 to %c128_62 step %c2_63 iter_args(%arg4 = %35) -> (tensor<128x128x55x47xf32>) {
      %c0_67 = arith.constant 0 : index
      %c128_68 = arith.constant 128 : index
      %c4_69 = arith.constant 4 : index
      %42 = scf.for %arg5 = %c0_67 to %c128_68 step %c4_69 iter_args(%arg6 = %arg4) -> (tensor<128x128x55x47xf32>) {
        %c0_70 = arith.constant 0 : index
        %c47 = arith.constant 47 : index
        %c1_71 = arith.constant 1 : index
        %43 = scf.for %arg7 = %c0_70 to %c47 step %c1_71 iter_args(%arg8 = %arg6) -> (tensor<128x128x55x47xf32>) {
          %44 = affine.apply #map4(%arg7)
          %extracted_slice_72 = tensor.extract_slice %33[%arg3, %arg5, 0, %44] [2, 4, 111, 3] [1, 1, 1, 1] : tensor<128x128x112x95xf32> to tensor<2x4x111x3xf32>
          %extracted_slice_73 = tensor.extract_slice %34[0, 0] [3, 3] [1, 1] : tensor<3x3xf32> to tensor<3x3xf32>
          %extracted_slice_74 = tensor.extract_slice %arg8[%arg3, %arg5, 0, %arg7] [2, 4, 55, 1] [1, 1, 1, 1] : tensor<128x128x55x47xf32> to tensor<2x4x55x1xf32>
          %c0_75 = arith.constant 0 : index
          %c3 = arith.constant 3 : index
          %c1_76 = arith.constant 1 : index
          %45 = scf.for %arg9 = %c0_75 to %c3 step %c1_76 iter_args(%arg10 = %extracted_slice_74) -> (tensor<2x4x55x1xf32>) {
            %extracted_slice_77 = tensor.extract_slice %extracted_slice_72[0, 0, 0, %arg9] [2, 4, 111, 1] [1, 1, 1, 1] : tensor<2x4x111x3xf32> to tensor<2x4x111x1xf32>
            %extracted_slice_78 = tensor.extract_slice %extracted_slice_73[0, %arg9] [3, 1] [1, 1] : tensor<3x3xf32> to tensor<3x1xf32>
            %extracted_slice_79 = tensor.extract_slice %arg10[0, 0, 0, 0] [2, 4, 55, 1] [1, 1, 1, 1] : tensor<2x4x55x1xf32> to tensor<2x4x55x1xf32>
            %46 = linalg.pooling_nchw_sum {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_3"} ins(%extracted_slice_77, %extracted_slice_78 : tensor<2x4x111x1xf32>, tensor<3x1xf32>) outs(%extracted_slice_79 : tensor<2x4x55x1xf32>) -> tensor<2x4x55x1xf32>
            %inserted_slice_80 = tensor.insert_slice %46 into %arg10[0, 0, 0, 0] [2, 4, 55, 1] [1, 1, 1, 1] : tensor<2x4x55x1xf32> into tensor<2x4x55x1xf32>
            scf.yield %inserted_slice_80 : tensor<2x4x55x1xf32>
          }
          %inserted_slice = tensor.insert_slice %45 into %arg8[%arg3, %arg5, 0, %arg7] [2, 4, 55, 1] [1, 1, 1, 1] : tensor<2x4x55x1xf32> into tensor<128x128x55x47xf32>
          scf.yield %inserted_slice : tensor<128x128x55x47xf32>
        }
        scf.yield %43 : tensor<128x128x55x47xf32>
      }
      scf.yield %42 : tensor<128x128x55x47xf32>
    }
    %37 = bufferization.alloc_tensor() : tensor<1x1xf32>
    %38 = bufferization.alloc_tensor() : tensor<128x128x55x47xf32>
    %c0_64 = arith.constant 0 : index
    %c128_65 = arith.constant 128 : index
    %c1_66 = arith.constant 1 : index
    %39 = scf.for %arg3 = %c0_64 to %c128_65 step %c1_66 iter_args(%arg4 = %38) -> (tensor<128x128x55x47xf32>) {
      %c0_67 = arith.constant 0 : index
      %c128_68 = arith.constant 128 : index
      %c32 = arith.constant 32 : index
      %42 = scf.for %arg5 = %c0_67 to %c128_68 step %c32 iter_args(%arg6 = %arg4) -> (tensor<128x128x55x47xf32>) {
        %c0_69 = arith.constant 0 : index
        %c47 = arith.constant 47 : index
        %c1_70 = arith.constant 1 : index
        %43 = scf.for %arg7 = %c0_69 to %c47 step %c1_70 iter_args(%arg8 = %arg6) -> (tensor<128x128x55x47xf32>) {
          %extracted_slice_71 = tensor.extract_slice %36[%arg3, %arg5, 0, %arg7] [1, 32, 55, 1] [1, 1, 1, 1] : tensor<128x128x55x47xf32> to tensor<1x32x55x1xf32>
          %extracted_slice_72 = tensor.extract_slice %37[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> to tensor<1x1xf32>
          %extracted_slice_73 = tensor.extract_slice %arg8[%arg3, %arg5, 0, %arg7] [1, 32, 55, 1] [1, 1, 1, 1] : tensor<128x128x55x47xf32> to tensor<1x32x55x1xf32>
          %44 = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, tag = "operation_4"} ins(%extracted_slice_71, %extracted_slice_72 : tensor<1x32x55x1xf32>, tensor<1x1xf32>) outs(%extracted_slice_73 : tensor<1x32x55x1xf32>) -> tensor<1x32x55x1xf32>
          %inserted_slice = tensor.insert_slice %44 into %arg8[%arg3, %arg5, 0, %arg7] [1, 32, 55, 1] [1, 1, 1, 1] : tensor<1x32x55x1xf32> into tensor<128x128x55x47xf32>
          scf.yield %inserted_slice : tensor<128x128x55x47xf32>
        }
        scf.yield %43 : tensor<128x128x55x47xf32>
      }
      scf.yield %42 : tensor<128x128x55x47xf32>
    }
    %40 = call @nanoTime() : () -> i64
    %41 = arith.subi %40, %0 : i64
    return %39, %41 : tensor<128x128x55x47xf32>, i64
  }
}
