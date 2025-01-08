#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0) -> (d0 floordiv 9)>
#map5 = affine_map<(d0, d1) -> (d0 floordiv 56 + (d1 mod 9) floordiv 3)>
#map6 = affine_map<(d0, d1) -> (d0 + d1 - (d0 floordiv 56) * 56 - (d1 floordiv 3) * 3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map10 = affine_map<(d0, d1) -> ((d0 floordiv 28) * 2 + (d1 mod 9) floordiv 3)>
#map11 = affine_map<(d0, d1) -> (d0 * 2 + d1 - (d0 floordiv 28) * 56 - (d1 floordiv 3) * 3)>
#map12 = affine_map<(d0, d1) -> (d0 floordiv 28 + (d1 mod 9) floordiv 3)>
#map13 = affine_map<(d0, d1) -> (d0 + d1 - (d0 floordiv 28) * 28 - (d1 floordiv 3) * 3)>
#map14 = affine_map<(d0) -> ((d0 floordiv 28) * 2)>
#map15 = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 28) * 56)>
#map16 = affine_map<(d0, d1) -> (d0 floordiv 14 + (d1 mod 9) floordiv 3)>
#map17 = affine_map<(d0, d1) -> (d0 + d1 - (d0 floordiv 14) * 14 - (d1 floordiv 3) * 3)>
#map18 = affine_map<(d0) -> ((d0 floordiv 14) * 2)>
#map19 = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 14) * 28)>
#map20 = affine_map<(d0, d1) -> ((d0 floordiv 7) * 2 + (d1 mod 9) floordiv 3)>
#map21 = affine_map<(d0, d1) -> (d0 * 2 + d1 - (d0 floordiv 7) * 14 - (d1 floordiv 3) * 3)>
#map22 = affine_map<(d0, d1) -> (d0, d1)>
#map23 = affine_map<(d0, d1) -> (d1, d0)>
#map24 = affine_map<(d0, d1) -> (0, d1)>
#map25 = affine_map<(d0, d1) -> (d1)>
  module attributes {torch.debug_module_name = "Net"} {
    memref.global "private" @global_seed : memref<i64> = dense<0>
    func.func @forward(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
      %c1 = arith.constant 1 : index
      %c1000 = arith.constant 1000 : index
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %cst = arith.constant 4.900000e+01 : f32
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 1.000000e-05 : f64
      %cst_1 = arith.constant 0xFF800000 : f32
      %cst_2 = arith.constant 0.000000e+00 : f32
      %0 = bufferization.alloc_tensor() : tensor<1000xf32>
      %1 = bufferization.alloc_tensor() : tensor<1000x512xf32>
      %2 = bufferization.alloc_tensor() : tensor<512xf32>
      %3 = bufferization.alloc_tensor() : tensor<512xf32>
      %4 = bufferization.alloc_tensor() : tensor<512xf32>
      %5 = bufferization.alloc_tensor() : tensor<512xf32>
      %6 = bufferization.alloc_tensor() : tensor<512x512x3x3xf32>
      %7 = bufferization.alloc_tensor() : tensor<512xf32>
      %8 = bufferization.alloc_tensor() : tensor<512xf32>
      %9 = bufferization.alloc_tensor() : tensor<512xf32>
      %10 = bufferization.alloc_tensor() : tensor<512xf32>
      %11 = bufferization.alloc_tensor() : tensor<512x512x3x3xf32>
      %12 = bufferization.alloc_tensor() : tensor<512xf32>
      %13 = bufferization.alloc_tensor() : tensor<512xf32>
      %14 = bufferization.alloc_tensor() : tensor<512xf32>
      %15 = bufferization.alloc_tensor() : tensor<512x256x1x1xf32>
      %16 = bufferization.alloc_tensor() : tensor<512xf32>
      %17 = bufferization.alloc_tensor() : tensor<512xf32>
      %18 = bufferization.alloc_tensor() : tensor<512xf32>
      %19 = bufferization.alloc_tensor() : tensor<512xf32>
      %20 = bufferization.alloc_tensor() : tensor<512x512x3x3xf32>
      %21 = bufferization.alloc_tensor() : tensor<512xf32>
      %22 = bufferization.alloc_tensor() : tensor<512xf32>
      %23 = bufferization.alloc_tensor() : tensor<512xf32>
      %24 = bufferization.alloc_tensor() : tensor<512xf32>
      %25 = bufferization.alloc_tensor() : tensor<512x256x3x3xf32>
      %26 = bufferization.alloc_tensor() : tensor<256xf32>
      %27 = bufferization.alloc_tensor() : tensor<256xf32>
      %28 = bufferization.alloc_tensor() : tensor<256xf32>
      %29 = bufferization.alloc_tensor() : tensor<256xf32>
      %30 = bufferization.alloc_tensor() : tensor<256x256x3x3xf32>
      %31 = bufferization.alloc_tensor() : tensor<256xf32>
      %32 = bufferization.alloc_tensor() : tensor<256xf32>
      %33 = bufferization.alloc_tensor() : tensor<256xf32>
      %34 = bufferization.alloc_tensor() : tensor<256xf32>
      %35 = bufferization.alloc_tensor() : tensor<256x256x3x3xf32>
      %36 = bufferization.alloc_tensor() : tensor<256xf32>
      %37 = bufferization.alloc_tensor() : tensor<256xf32>
      %38 = bufferization.alloc_tensor() : tensor<256xf32>
      %39 = bufferization.alloc_tensor() : tensor<256x128x1x1xf32>
      %40 = bufferization.alloc_tensor() : tensor<256xf32>
      %41 = bufferization.alloc_tensor() : tensor<256xf32>
      %42 = bufferization.alloc_tensor() : tensor<256xf32>
      %43 = bufferization.alloc_tensor() : tensor<256xf32>
      %44 = bufferization.alloc_tensor() : tensor<256x256x3x3xf32>
      %45 = bufferization.alloc_tensor() : tensor<256xf32>
      %46 = bufferization.alloc_tensor() : tensor<256xf32>
      %47 = bufferization.alloc_tensor() : tensor<256xf32>
      %48 = bufferization.alloc_tensor() : tensor<256xf32>
      %49 = bufferization.alloc_tensor() : tensor<256x128x3x3xf32>
      %50 = bufferization.alloc_tensor() : tensor<128xf32>
      %51 = bufferization.alloc_tensor() : tensor<128xf32>
      %52 = bufferization.alloc_tensor() : tensor<128xf32>
      %53 = bufferization.alloc_tensor() : tensor<128xf32>
      %54 = bufferization.alloc_tensor() : tensor<128x128x3x3xf32>
      %55 = bufferization.alloc_tensor() : tensor<128xf32>
      %56 = bufferization.alloc_tensor() : tensor<128xf32>
      %57 = bufferization.alloc_tensor() : tensor<128xf32>
      %58 = bufferization.alloc_tensor() : tensor<128xf32>
      %59 = bufferization.alloc_tensor() : tensor<128x128x3x3xf32>
      %60 = bufferization.alloc_tensor() : tensor<128xf32>
      %61 = bufferization.alloc_tensor() : tensor<128xf32>
      %62 = bufferization.alloc_tensor() : tensor<128xf32>
      %63 = bufferization.alloc_tensor() : tensor<128x64x1x1xf32>
      %64 = bufferization.alloc_tensor() : tensor<128xf32>
      %65 = bufferization.alloc_tensor() : tensor<128xf32>
      %66 = bufferization.alloc_tensor() : tensor<128xf32>
      %67 = bufferization.alloc_tensor() : tensor<128xf32>
      %68 = bufferization.alloc_tensor() : tensor<128x128x3x3xf32>
      %69 = bufferization.alloc_tensor() : tensor<128xf32>
      %70 = bufferization.alloc_tensor() : tensor<128xf32>
      %71 = bufferization.alloc_tensor() : tensor<128xf32>
      %72 = bufferization.alloc_tensor() : tensor<128xf32>
      %73 = bufferization.alloc_tensor() : tensor<128x64x3x3xf32>
      %74 = bufferization.alloc_tensor() : tensor<64xf32>
      %75 = bufferization.alloc_tensor() : tensor<64xf32>
      %76 = bufferization.alloc_tensor() : tensor<64xf32>
      %77 = bufferization.alloc_tensor() : tensor<64xf32>
      %78 = bufferization.alloc_tensor() : tensor<64x64x3x3xf32>
      %79 = bufferization.alloc_tensor() : tensor<64xf32>
      %80 = bufferization.alloc_tensor() : tensor<64xf32>
      %81 = bufferization.alloc_tensor() : tensor<64xf32>
      %82 = bufferization.alloc_tensor() : tensor<64xf32>
      %83 = bufferization.alloc_tensor() : tensor<64x64x3x3xf32>
      %84 = bufferization.alloc_tensor() : tensor<64xf32>
      %85 = bufferization.alloc_tensor() : tensor<64xf32>
      %86 = bufferization.alloc_tensor() : tensor<64xf32>
      %87 = bufferization.alloc_tensor() : tensor<64xf32>
      %88 = bufferization.alloc_tensor() : tensor<64x64x3x3xf32>
      %89 = bufferization.alloc_tensor() : tensor<64xf32>
      %90 = bufferization.alloc_tensor() : tensor<64xf32>
      %91 = bufferization.alloc_tensor() : tensor<64xf32>
      %92 = bufferization.alloc_tensor() : tensor<64xf32>
      %93 = bufferization.alloc_tensor() : tensor<64x64x3x3xf32>
      %94 = bufferization.alloc_tensor() : tensor<64xf32>
      %95 = bufferization.alloc_tensor() : tensor<64xf32>
      %96 = bufferization.alloc_tensor() : tensor<64xf32>
      %97 = bufferization.alloc_tensor() : tensor<64xf32>
      %98 = bufferization.alloc_tensor() : tensor<64x3x7x7xf32>
      %padded = tensor.pad %arg0 low[0, 0, 3, 3] high[0, 0, 3, 3] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_2 : f32
      } : tensor<1x3x224x224xf32> to tensor<1x3x230x230xf32>
      %99 = tensor.empty() : tensor<1x64x112x112xf32>
      %100 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %99) -> (tensor<1x64x112x112xf32>) {
        %extracted_slice = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 112, 112] [1, 1, 1, 1] : tensor<1x64x112x112xf32> to tensor<1x2x112x112xf32>
        %204 = linalg.fill {tag = "operation_0"} ins(%cst_2 : f32) outs(%extracted_slice : tensor<1x2x112x112xf32>) -> tensor<1x2x112x112xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 112, 112] [1, 1, 1, 1] : tensor<1x2x112x112xf32> into tensor<1x64x112x112xf32>
        scf.yield %inserted_slice : tensor<1x64x112x112xf32>
      }
      %101 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %100) -> (tensor<1x64x112x112xf32>) {
        %extracted_slice = tensor.extract_slice %padded[0, 0, 0, 0] [1, 3, 229, 229] [1, 1, 1, 1] : tensor<1x3x230x230xf32> to tensor<1x3x229x229xf32>
        %extracted_slice_43 = tensor.extract_slice %98[%arg1, 0, 0, 0] [2, 3, 7, 7] [1, 1, 1, 1] : tensor<64x3x7x7xf32> to tensor<2x3x7x7xf32>
        %extracted_slice_44 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 112, 112] [1, 1, 1, 1] : tensor<1x64x112x112xf32> to tensor<1x2x112x112xf32>
        %204 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_1"} ins(%extracted_slice, %extracted_slice_43 : tensor<1x3x229x229xf32>, tensor<2x3x7x7xf32>) outs(%extracted_slice_44 : tensor<1x2x112x112xf32>) -> tensor<1x2x112x112xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 112, 112] [1, 1, 1, 1] : tensor<1x2x112x112xf32> into tensor<1x64x112x112xf32>
        scf.yield %inserted_slice : tensor<1x64x112x112xf32>
      }
      %102 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %101) -> (tensor<1x64x112x112xf32>) {
        %extracted_slice = tensor.extract_slice %101[0, %arg1, 0, 0] [1, 2, 112, 112] [1, 1, 1, 1] : tensor<1x64x112x112xf32> to tensor<1x2x112x112xf32>
        %extracted_slice_43 = tensor.extract_slice %95[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %94[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %97[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %96[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 112, 112] [1, 1, 1, 1] : tensor<1x64x112x112xf32> to tensor<1x2x112x112xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x112x112xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x112x112xf32>) attrs =  {tag = "operation_2"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x112x112xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 112, 112] [1, 1, 1, 1] : tensor<1x2x112x112xf32> into tensor<1x64x112x112xf32>
        scf.yield %inserted_slice : tensor<1x64x112x112xf32>
      }
      %103 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %99) -> (tensor<1x64x112x112xf32>) {
        %extracted_slice = tensor.extract_slice %102[0, %arg1, 0, 0] [1, 2, 112, 112] [1, 1, 1, 1] : tensor<1x64x112x112xf32> to tensor<1x2x112x112xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 112, 112] [1, 1, 1, 1] : tensor<1x64x112x112xf32> to tensor<1x2x112x112xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x112x112xf32>) outs(%extracted_slice_43 : tensor<1x2x112x112xf32>) attrs =  {tag = "operation_3"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.cmpf ugt, %in, %cst_2 : f32
          %206 = arith.select %205, %in, %cst_2 : f32
          linalg.yield %206 : f32
        } -> tensor<1x2x112x112xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 112, 112] [1, 1, 1, 1] : tensor<1x2x112x112xf32> into tensor<1x64x112x112xf32>
        scf.yield %inserted_slice : tensor<1x64x112x112xf32>
      }
      %padded_3 = tensor.pad %103 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_1 : f32
      } : tensor<1x64x112x112xf32> to tensor<1x64x114x114xf32>
      %104 = tensor.empty() : tensor<1x64x56x56xf32>
      %105 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %104) -> (tensor<1x64x56x56xf32>) {
        %extracted_slice = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %204 = linalg.fill {tag = "operation_4"} ins(%cst_1 : f32) outs(%extracted_slice : tensor<1x2x56x56xf32>) -> tensor<1x2x56x56xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x2x56x56xf32> into tensor<1x64x56x56xf32>
        scf.yield %inserted_slice : tensor<1x64x56x56xf32>
      }
      %106 = tensor.empty() : tensor<3x3xf32>
      %107 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %105) -> (tensor<1x64x56x56xf32>) {
        %extracted_slice = tensor.extract_slice %padded_3[0, %arg1, 0, 0] [1, 2, 113, 113] [1, 1, 1, 1] : tensor<1x64x114x114xf32> to tensor<1x2x113x113xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %204 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_5"} ins(%extracted_slice, %106 : tensor<1x2x113x113xf32>, tensor<3x3xf32>) outs(%extracted_slice_43 : tensor<1x2x56x56xf32>) -> tensor<1x2x56x56xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x2x56x56xf32> into tensor<1x64x56x56xf32>
        scf.yield %inserted_slice : tensor<1x64x56x56xf32>
      }
      %padded_4 = tensor.pad %107 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_2 : f32
      } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
      %108 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %104) -> (tensor<1x64x56x56xf32>) {
        %extracted_slice = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %204 = linalg.fill {tag = "operation_6"} ins(%cst_2 : f32) outs(%extracted_slice : tensor<1x2x56x56xf32>) -> tensor<1x2x56x56xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x2x56x56xf32> into tensor<1x64x56x56xf32>
        scf.yield %inserted_slice : tensor<1x64x56x56xf32>
      }
      %collapsed = tensor.collapse_shape %93 [[0], [1, 2, 3]] : tensor<64x64x3x3xf32> into tensor<64x576xf32>
      %collapsed_5 = tensor.collapse_shape %108 [[0], [1], [2, 3]] : tensor<1x64x56x56xf32> into tensor<1x64x3136xf32>
      %109 = tensor.empty() : tensor<1x576x3136xf32>
      %110 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel", "parallel"]} outs(%109 : tensor<1x576x3136xf32>) attrs =  {tag = "img2col_producer"} {
      ^bb0(%out: f32):
        %204 = linalg.index 0 : index
        %205 = linalg.index 1 : index
        %206 = linalg.index 2 : index
        %207 = affine.apply #map4(%205)
        %208 = affine.apply #map5(%206, %205)
        %209 = affine.apply #map6(%206, %205)
        %extracted = tensor.extract %padded_4[%204, %207, %208, %209] : tensor<1x64x58x58xf32>
        linalg.yield %extracted : f32
      } -> tensor<1x576x3136xf32>
      %111 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed, %110 : tensor<64x576xf32>, tensor<1x576x3136xf32>) outs(%collapsed_5 : tensor<1x64x3136xf32>) attrs =  {tag = "operation_7"} {
      ^bb0(%in: f32, %in_43: f32, %out: f32):
        %204 = arith.mulf %in, %in_43 : f32
        %205 = arith.addf %204, %out : f32
        linalg.yield %205 : f32
      } -> tensor<1x64x3136xf32>
      %expanded = tensor.expand_shape %111 [[0], [1], [2, 3]] : tensor<1x64x3136xf32> into tensor<1x64x56x56xf32>
      %112 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %expanded) -> (tensor<1x64x56x56xf32>) {
        %extracted_slice = tensor.extract_slice %expanded[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %extracted_slice_43 = tensor.extract_slice %90[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %89[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %92[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %91[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x56x56xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x56x56xf32>) attrs =  {tag = "operation_8"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x56x56xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x2x56x56xf32> into tensor<1x64x56x56xf32>
        scf.yield %inserted_slice : tensor<1x64x56x56xf32>
      }
      %113 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %104) -> (tensor<1x64x56x56xf32>) {
        %extracted_slice = tensor.extract_slice %112[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x56x56xf32>) outs(%extracted_slice_43 : tensor<1x2x56x56xf32>) attrs =  {tag = "operation_9"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.cmpf ugt, %in, %cst_2 : f32
          %206 = arith.select %205, %in, %cst_2 : f32
          linalg.yield %206 : f32
        } -> tensor<1x2x56x56xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x2x56x56xf32> into tensor<1x64x56x56xf32>
        scf.yield %inserted_slice : tensor<1x64x56x56xf32>
      }
      %padded_6 = tensor.pad %113 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_2 : f32
      } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
      %114 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %108) -> (tensor<1x64x56x56xf32>) {
        %extracted_slice = tensor.extract_slice %88[%arg1, 0, 0, 0] [2, 64, 3, 3] [1, 1, 1, 1] : tensor<64x64x3x3xf32> to tensor<2x64x3x3xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %204 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_10"} ins(%padded_6, %extracted_slice : tensor<1x64x58x58xf32>, tensor<2x64x3x3xf32>) outs(%extracted_slice_43 : tensor<1x2x56x56xf32>) -> tensor<1x2x56x56xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x2x56x56xf32> into tensor<1x64x56x56xf32>
        scf.yield %inserted_slice : tensor<1x64x56x56xf32>
      }
      %115 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %114) -> (tensor<1x64x56x56xf32>) {
        %extracted_slice = tensor.extract_slice %114[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %extracted_slice_43 = tensor.extract_slice %85[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %84[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %87[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %86[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x56x56xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x56x56xf32>) attrs =  {tag = "operation_11"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x56x56xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x2x56x56xf32> into tensor<1x64x56x56xf32>
        scf.yield %inserted_slice : tensor<1x64x56x56xf32>
      }
      %116 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %104) -> (tensor<1x64x56x56xf32>) {
        %extracted_slice = tensor.extract_slice %115[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %extracted_slice_43 = tensor.extract_slice %107[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %extracted_slice_44 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43 : tensor<1x2x56x56xf32>, tensor<1x2x56x56xf32>) outs(%extracted_slice_44 : tensor<1x2x56x56xf32>) attrs =  {tag = "operation_12"} {
        ^bb0(%in: f32, %in_45: f32, %out: f32):
          %205 = arith.addf %in, %in_45 : f32
          linalg.yield %205 : f32
        } -> tensor<1x2x56x56xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x2x56x56xf32> into tensor<1x64x56x56xf32>
        scf.yield %inserted_slice : tensor<1x64x56x56xf32>
      }
      %117 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %104) -> (tensor<1x64x56x56xf32>) {
        %extracted_slice = tensor.extract_slice %116[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x56x56xf32>) outs(%extracted_slice_43 : tensor<1x2x56x56xf32>) attrs =  {tag = "operation_13"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.cmpf ugt, %in, %cst_2 : f32
          %206 = arith.select %205, %in, %cst_2 : f32
          linalg.yield %206 : f32
        } -> tensor<1x2x56x56xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x2x56x56xf32> into tensor<1x64x56x56xf32>
        scf.yield %inserted_slice : tensor<1x64x56x56xf32>
      }
      %padded_7 = tensor.pad %117 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_2 : f32
      } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
      %118 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %108) -> (tensor<1x64x56x56xf32>) {
        %extracted_slice = tensor.extract_slice %83[%arg1, 0, 0, 0] [2, 64, 3, 3] [1, 1, 1, 1] : tensor<64x64x3x3xf32> to tensor<2x64x3x3xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %204 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_14"} ins(%padded_7, %extracted_slice : tensor<1x64x58x58xf32>, tensor<2x64x3x3xf32>) outs(%extracted_slice_43 : tensor<1x2x56x56xf32>) -> tensor<1x2x56x56xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x2x56x56xf32> into tensor<1x64x56x56xf32>
        scf.yield %inserted_slice : tensor<1x64x56x56xf32>
      }
      %119 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %118) -> (tensor<1x64x56x56xf32>) {
        %extracted_slice = tensor.extract_slice %118[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %extracted_slice_43 = tensor.extract_slice %80[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %79[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %82[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %81[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x56x56xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x56x56xf32>) attrs =  {tag = "operation_15"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x56x56xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x2x56x56xf32> into tensor<1x64x56x56xf32>
        scf.yield %inserted_slice : tensor<1x64x56x56xf32>
      }
      %120 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %104) -> (tensor<1x64x56x56xf32>) {
        %extracted_slice = tensor.extract_slice %119[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x56x56xf32>) outs(%extracted_slice_43 : tensor<1x2x56x56xf32>) attrs =  {tag = "operation_16"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.cmpf ugt, %in, %cst_2 : f32
          %206 = arith.select %205, %in, %cst_2 : f32
          linalg.yield %206 : f32
        } -> tensor<1x2x56x56xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x2x56x56xf32> into tensor<1x64x56x56xf32>
        scf.yield %inserted_slice : tensor<1x64x56x56xf32>
      }
      %padded_8 = tensor.pad %120 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_2 : f32
      } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
      %121 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %108) -> (tensor<1x64x56x56xf32>) {
        %extracted_slice = tensor.extract_slice %78[%arg1, 0, 0, 0] [2, 64, 3, 3] [1, 1, 1, 1] : tensor<64x64x3x3xf32> to tensor<2x64x3x3xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %204 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_17"} ins(%padded_8, %extracted_slice : tensor<1x64x58x58xf32>, tensor<2x64x3x3xf32>) outs(%extracted_slice_43 : tensor<1x2x56x56xf32>) -> tensor<1x2x56x56xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x2x56x56xf32> into tensor<1x64x56x56xf32>
        scf.yield %inserted_slice : tensor<1x64x56x56xf32>
      }
      %122 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %121) -> (tensor<1x64x56x56xf32>) {
        %extracted_slice = tensor.extract_slice %121[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %extracted_slice_43 = tensor.extract_slice %75[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %74[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %77[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %76[%arg1] [2] [1] : tensor<64xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x56x56xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x56x56xf32>) attrs =  {tag = "operation_18"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x56x56xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x2x56x56xf32> into tensor<1x64x56x56xf32>
        scf.yield %inserted_slice : tensor<1x64x56x56xf32>
      }
      %123 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %104) -> (tensor<1x64x56x56xf32>) {
        %extracted_slice = tensor.extract_slice %122[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %extracted_slice_43 = tensor.extract_slice %117[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %extracted_slice_44 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43 : tensor<1x2x56x56xf32>, tensor<1x2x56x56xf32>) outs(%extracted_slice_44 : tensor<1x2x56x56xf32>) attrs =  {tag = "operation_19"} {
        ^bb0(%in: f32, %in_45: f32, %out: f32):
          %205 = arith.addf %in, %in_45 : f32
          linalg.yield %205 : f32
        } -> tensor<1x2x56x56xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x2x56x56xf32> into tensor<1x64x56x56xf32>
        scf.yield %inserted_slice : tensor<1x64x56x56xf32>
      }
      %124 = scf.for %arg1 = %c0 to %c64 step %c2 iter_args(%arg2 = %104) -> (tensor<1x64x56x56xf32>) {
        %extracted_slice = tensor.extract_slice %123[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x64x56x56xf32> to tensor<1x2x56x56xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x56x56xf32>) outs(%extracted_slice_43 : tensor<1x2x56x56xf32>) attrs =  {tag = "operation_20"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.cmpf ugt, %in, %cst_2 : f32
          %206 = arith.select %205, %in, %cst_2 : f32
          linalg.yield %206 : f32
        } -> tensor<1x2x56x56xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 56, 56] [1, 1, 1, 1] : tensor<1x2x56x56xf32> into tensor<1x64x56x56xf32>
        scf.yield %inserted_slice : tensor<1x64x56x56xf32>
      }
      %padded_9 = tensor.pad %124 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_2 : f32
      } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
      %125 = tensor.empty() : tensor<1x128x28x28xf32>
      %126 = scf.for %arg1 = %c0 to %c128 step %c2 iter_args(%arg2 = %125) -> (tensor<1x128x28x28xf32>) {
        %extracted_slice = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %204 = linalg.fill {tag = "operation_21"} ins(%cst_2 : f32) outs(%extracted_slice : tensor<1x2x28x28xf32>) -> tensor<1x2x28x28xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x2x28x28xf32> into tensor<1x128x28x28xf32>
        scf.yield %inserted_slice : tensor<1x128x28x28xf32>
      }
      %collapsed_10 = tensor.collapse_shape %73 [[0], [1, 2, 3]] : tensor<128x64x3x3xf32> into tensor<128x576xf32>
      %collapsed_11 = tensor.collapse_shape %126 [[0], [1], [2, 3]] : tensor<1x128x28x28xf32> into tensor<1x128x784xf32>
      %127 = tensor.empty() : tensor<1x576x784xf32>
      %128 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel", "parallel"]} outs(%127 : tensor<1x576x784xf32>) attrs =  {AAAAA} {
      ^bb0(%out: f32):
        %204 = linalg.index 0 : index
        %205 = linalg.index 1 : index
        %206 = linalg.index 2 : index
        %207 = affine.apply #map4(%205)
        %208 = affine.apply #map10(%206, %205)
        %209 = affine.apply #map11(%206, %205)
        %extracted = tensor.extract %padded_9[%204, %207, %208, %209] : tensor<1x64x58x58xf32>
        linalg.yield %extracted : f32
      } -> tensor<1x576x784xf32>
      %129 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_10, %128 : tensor<128x576xf32>, tensor<1x576x784xf32>) outs(%collapsed_11 : tensor<1x128x784xf32>) attrs =  {tag = "operation_22"} {
      ^bb0(%in: f32, %in_43: f32, %out: f32):
        %204 = arith.mulf %in, %in_43 : f32
        %205 = arith.addf %204, %out : f32
        linalg.yield %205 : f32
      } -> tensor<1x128x784xf32>
      %expanded_12 = tensor.expand_shape %129 [[0], [1], [2, 3]] : tensor<1x128x784xf32> into tensor<1x128x28x28xf32>
      %130 = scf.for %arg1 = %c0 to %c128 step %c2 iter_args(%arg2 = %expanded_12) -> (tensor<1x128x28x28xf32>) {
        %extracted_slice = tensor.extract_slice %expanded_12[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %extracted_slice_43 = tensor.extract_slice %70[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %69[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %72[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %71[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x28x28xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x28x28xf32>) attrs =  {tag = "operation_23"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x28x28xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x2x28x28xf32> into tensor<1x128x28x28xf32>
        scf.yield %inserted_slice : tensor<1x128x28x28xf32>
      }
      %131 = scf.for %arg1 = %c0 to %c128 step %c2 iter_args(%arg2 = %125) -> (tensor<1x128x28x28xf32>) {
        %extracted_slice = tensor.extract_slice %130[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x28x28xf32>) outs(%extracted_slice_43 : tensor<1x2x28x28xf32>) attrs =  {tag = "operation_24"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.cmpf ugt, %in, %cst_2 : f32
          %206 = arith.select %205, %in, %cst_2 : f32
          linalg.yield %206 : f32
        } -> tensor<1x2x28x28xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x2x28x28xf32> into tensor<1x128x28x28xf32>
        scf.yield %inserted_slice : tensor<1x128x28x28xf32>
      }
      %padded_13 = tensor.pad %131 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_2 : f32
      } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
      %collapsed_14 = tensor.collapse_shape %68 [[0], [1, 2, 3]] : tensor<128x128x3x3xf32> into tensor<128x1152xf32>
      %collapsed_15 = tensor.collapse_shape %126 [[0], [1], [2, 3]] : tensor<1x128x28x28xf32> into tensor<1x128x784xf32>
      %132 = tensor.empty() : tensor<1x1152x784xf32>
      %133 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel", "parallel"]} outs(%132 : tensor<1x1152x784xf32>) attrs =  {AAAAA} {
      ^bb0(%out: f32):
        %204 = linalg.index 0 : index
        %205 = linalg.index 1 : index
        %206 = linalg.index 2 : index
        %207 = affine.apply #map4(%205)
        %208 = affine.apply #map12(%206, %205)
        %209 = affine.apply #map13(%206, %205)
        %extracted = tensor.extract %padded_13[%204, %207, %208, %209] : tensor<1x128x30x30xf32>
        linalg.yield %extracted : f32
      } -> tensor<1x1152x784xf32>
      %134 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_14, %133 : tensor<128x1152xf32>, tensor<1x1152x784xf32>) outs(%collapsed_15 : tensor<1x128x784xf32>) attrs =  {tag = "operation_25"} {
      ^bb0(%in: f32, %in_43: f32, %out: f32):
        %204 = arith.mulf %in, %in_43 : f32
        %205 = arith.addf %204, %out : f32
        linalg.yield %205 : f32
      } -> tensor<1x128x784xf32>
      %expanded_16 = tensor.expand_shape %134 [[0], [1], [2, 3]] : tensor<1x128x784xf32> into tensor<1x128x28x28xf32>
      %135 = scf.for %arg1 = %c0 to %c128 step %c2 iter_args(%arg2 = %expanded_16) -> (tensor<1x128x28x28xf32>) {
        %extracted_slice = tensor.extract_slice %expanded_16[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %extracted_slice_43 = tensor.extract_slice %65[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %64[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %67[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %66[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x28x28xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x28x28xf32>) attrs =  {tag = "operation_26"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x28x28xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x2x28x28xf32> into tensor<1x128x28x28xf32>
        scf.yield %inserted_slice : tensor<1x128x28x28xf32>
      }
      %collapsed_17 = tensor.collapse_shape %63 [[0], [1, 2, 3]] : tensor<128x64x1x1xf32> into tensor<128x64xf32>
      %collapsed_18 = tensor.collapse_shape %126 [[0], [1], [2, 3]] : tensor<1x128x28x28xf32> into tensor<1x128x784xf32>
      %136 = tensor.empty() : tensor<1x64x784xf32>
      %137 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel", "parallel"]} outs(%136 : tensor<1x64x784xf32>) attrs =  {AAAAA} {
      ^bb0(%out: f32):
        %204 = linalg.index 0 : index
        %205 = linalg.index 1 : index
        %206 = linalg.index 2 : index
        %207 = affine.apply #map14(%206)
        %208 = affine.apply #map15(%206)
        %extracted = tensor.extract %124[%204, %205, %207, %208] : tensor<1x64x56x56xf32>
        linalg.yield %extracted : f32
      } -> tensor<1x64x784xf32>
      %138 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_17, %137 : tensor<128x64xf32>, tensor<1x64x784xf32>) outs(%collapsed_18 : tensor<1x128x784xf32>) attrs =  {tag = "operation_27"} {
      ^bb0(%in: f32, %in_43: f32, %out: f32):
        %204 = arith.mulf %in, %in_43 : f32
        %205 = arith.addf %204, %out : f32
        linalg.yield %205 : f32
      } -> tensor<1x128x784xf32>
      %expanded_19 = tensor.expand_shape %138 [[0], [1], [2, 3]] : tensor<1x128x784xf32> into tensor<1x128x28x28xf32>
      %139 = scf.for %arg1 = %c0 to %c128 step %c2 iter_args(%arg2 = %expanded_19) -> (tensor<1x128x28x28xf32>) {
        %extracted_slice = tensor.extract_slice %expanded_19[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %extracted_slice_43 = tensor.extract_slice %60[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %64[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %62[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %61[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x28x28xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x28x28xf32>) attrs =  {tag = "operation_28"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x28x28xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x2x28x28xf32> into tensor<1x128x28x28xf32>
        scf.yield %inserted_slice : tensor<1x128x28x28xf32>
      }
      %140 = scf.for %arg1 = %c0 to %c128 step %c2 iter_args(%arg2 = %125) -> (tensor<1x128x28x28xf32>) {
        %extracted_slice = tensor.extract_slice %135[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %extracted_slice_43 = tensor.extract_slice %139[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %extracted_slice_44 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43 : tensor<1x2x28x28xf32>, tensor<1x2x28x28xf32>) outs(%extracted_slice_44 : tensor<1x2x28x28xf32>) attrs =  {tag = "operation_29"} {
        ^bb0(%in: f32, %in_45: f32, %out: f32):
          %205 = arith.addf %in, %in_45 : f32
          linalg.yield %205 : f32
        } -> tensor<1x2x28x28xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x2x28x28xf32> into tensor<1x128x28x28xf32>
        scf.yield %inserted_slice : tensor<1x128x28x28xf32>
      }
      %141 = scf.for %arg1 = %c0 to %c128 step %c2 iter_args(%arg2 = %125) -> (tensor<1x128x28x28xf32>) {
        %extracted_slice = tensor.extract_slice %140[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x28x28xf32>) outs(%extracted_slice_43 : tensor<1x2x28x28xf32>) attrs =  {tag = "operation_30"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.cmpf ugt, %in, %cst_2 : f32
          %206 = arith.select %205, %in, %cst_2 : f32
          linalg.yield %206 : f32
        } -> tensor<1x2x28x28xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x2x28x28xf32> into tensor<1x128x28x28xf32>
        scf.yield %inserted_slice : tensor<1x128x28x28xf32>
      }
      %padded_20 = tensor.pad %141 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_2 : f32
      } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
      %collapsed_21 = tensor.collapse_shape %59 [[0], [1, 2, 3]] : tensor<128x128x3x3xf32> into tensor<128x1152xf32>
      %collapsed_22 = tensor.collapse_shape %126 [[0], [1], [2, 3]] : tensor<1x128x28x28xf32> into tensor<1x128x784xf32>
      %142 = tensor.empty() : tensor<1x1152x784xf32>
      %143 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel", "parallel"]} outs(%142 : tensor<1x1152x784xf32>) attrs =  {AAAAA} {
      ^bb0(%out: f32):
        %204 = linalg.index 0 : index
        %205 = linalg.index 1 : index
        %206 = linalg.index 2 : index
        %207 = affine.apply #map4(%205)
        %208 = affine.apply #map12(%206, %205)
        %209 = affine.apply #map13(%206, %205)
        %extracted = tensor.extract %padded_20[%204, %207, %208, %209] : tensor<1x128x30x30xf32>
        linalg.yield %extracted : f32
      } -> tensor<1x1152x784xf32>
      %144 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_21, %143 : tensor<128x1152xf32>, tensor<1x1152x784xf32>) outs(%collapsed_22 : tensor<1x128x784xf32>) attrs =  {tag = "operation_31"} {
      ^bb0(%in: f32, %in_43: f32, %out: f32):
        %204 = arith.mulf %in, %in_43 : f32
        %205 = arith.addf %204, %out : f32
        linalg.yield %205 : f32
      } -> tensor<1x128x784xf32>
      %expanded_23 = tensor.expand_shape %144 [[0], [1], [2, 3]] : tensor<1x128x784xf32> into tensor<1x128x28x28xf32>
      %145 = scf.for %arg1 = %c0 to %c128 step %c2 iter_args(%arg2 = %expanded_23) -> (tensor<1x128x28x28xf32>) {
        %extracted_slice = tensor.extract_slice %expanded_23[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %extracted_slice_43 = tensor.extract_slice %56[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %55[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %58[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %57[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x28x28xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x28x28xf32>) attrs =  {tag = "operation_32"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x28x28xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x2x28x28xf32> into tensor<1x128x28x28xf32>
        scf.yield %inserted_slice : tensor<1x128x28x28xf32>
      }
      %146 = scf.for %arg1 = %c0 to %c128 step %c2 iter_args(%arg2 = %125) -> (tensor<1x128x28x28xf32>) {
        %extracted_slice = tensor.extract_slice %145[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x28x28xf32>) outs(%extracted_slice_43 : tensor<1x2x28x28xf32>) attrs =  {tag = "operation_33"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.cmpf ugt, %in, %cst_2 : f32
          %206 = arith.select %205, %in, %cst_2 : f32
          linalg.yield %206 : f32
        } -> tensor<1x2x28x28xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x2x28x28xf32> into tensor<1x128x28x28xf32>
        scf.yield %inserted_slice : tensor<1x128x28x28xf32>
      }
      %padded_24 = tensor.pad %146 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_2 : f32
      } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
      %147 = scf.for %arg1 = %c0 to %c128 step %c2 iter_args(%arg2 = %126) -> (tensor<1x128x28x28xf32>) {
        %extracted_slice = tensor.extract_slice %54[%arg1, 0, 0, 0] [2, 128, 3, 3] [1, 1, 1, 1] : tensor<128x128x3x3xf32> to tensor<2x128x3x3xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %204 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_34"} ins(%padded_24, %extracted_slice : tensor<1x128x30x30xf32>, tensor<2x128x3x3xf32>) outs(%extracted_slice_43 : tensor<1x2x28x28xf32>) -> tensor<1x2x28x28xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x2x28x28xf32> into tensor<1x128x28x28xf32>
        scf.yield %inserted_slice : tensor<1x128x28x28xf32>
      }
      %148 = scf.for %arg1 = %c0 to %c128 step %c2 iter_args(%arg2 = %147) -> (tensor<1x128x28x28xf32>) {
        %extracted_slice = tensor.extract_slice %147[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %extracted_slice_43 = tensor.extract_slice %51[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %50[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %53[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %52[%arg1] [2] [1] : tensor<128xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x28x28xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x28x28xf32>) attrs =  {tag = "operation_35"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x28x28xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x2x28x28xf32> into tensor<1x128x28x28xf32>
        scf.yield %inserted_slice : tensor<1x128x28x28xf32>
      }
      %149 = scf.for %arg1 = %c0 to %c128 step %c2 iter_args(%arg2 = %125) -> (tensor<1x128x28x28xf32>) {
        %extracted_slice = tensor.extract_slice %148[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %extracted_slice_43 = tensor.extract_slice %141[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %extracted_slice_44 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43 : tensor<1x2x28x28xf32>, tensor<1x2x28x28xf32>) outs(%extracted_slice_44 : tensor<1x2x28x28xf32>) attrs =  {tag = "operation_36"} {
        ^bb0(%in: f32, %in_45: f32, %out: f32):
          %205 = arith.addf %in, %in_45 : f32
          linalg.yield %205 : f32
        } -> tensor<1x2x28x28xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x2x28x28xf32> into tensor<1x128x28x28xf32>
        scf.yield %inserted_slice : tensor<1x128x28x28xf32>
      }
      %150 = scf.for %arg1 = %c0 to %c128 step %c2 iter_args(%arg2 = %125) -> (tensor<1x128x28x28xf32>) {
        %extracted_slice = tensor.extract_slice %149[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x128x28x28xf32> to tensor<1x2x28x28xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x28x28xf32>) outs(%extracted_slice_43 : tensor<1x2x28x28xf32>) attrs =  {tag = "operation_37"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.cmpf ugt, %in, %cst_2 : f32
          %206 = arith.select %205, %in, %cst_2 : f32
          linalg.yield %206 : f32
        } -> tensor<1x2x28x28xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 28, 28] [1, 1, 1, 1] : tensor<1x2x28x28xf32> into tensor<1x128x28x28xf32>
        scf.yield %inserted_slice : tensor<1x128x28x28xf32>
      }
      %padded_25 = tensor.pad %150 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_2 : f32
      } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
      %151 = tensor.empty() : tensor<1x256x14x14xf32>
      %152 = scf.for %arg1 = %c0 to %c256 step %c2 iter_args(%arg2 = %151) -> (tensor<1x256x14x14xf32>) {
        %extracted_slice = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %204 = linalg.fill {tag = "operation_38"} ins(%cst_2 : f32) outs(%extracted_slice : tensor<1x2x14x14xf32>) -> tensor<1x2x14x14xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x2x14x14xf32> into tensor<1x256x14x14xf32>
        scf.yield %inserted_slice : tensor<1x256x14x14xf32>
      }
      %153 = scf.for %arg1 = %c0 to %c256 step %c2 iter_args(%arg2 = %152) -> (tensor<1x256x14x14xf32>) {
        %extracted_slice = tensor.extract_slice %padded_25[0, 0, 0, 0] [1, 128, 29, 29] [1, 1, 1, 1] : tensor<1x128x30x30xf32> to tensor<1x128x29x29xf32>
        %extracted_slice_43 = tensor.extract_slice %49[%arg1, 0, 0, 0] [2, 128, 3, 3] [1, 1, 1, 1] : tensor<256x128x3x3xf32> to tensor<2x128x3x3xf32>
        %extracted_slice_44 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %204 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_39"} ins(%extracted_slice, %extracted_slice_43 : tensor<1x128x29x29xf32>, tensor<2x128x3x3xf32>) outs(%extracted_slice_44 : tensor<1x2x14x14xf32>) -> tensor<1x2x14x14xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x2x14x14xf32> into tensor<1x256x14x14xf32>
        scf.yield %inserted_slice : tensor<1x256x14x14xf32>
      }
      %154 = scf.for %arg1 = %c0 to %c256 step %c2 iter_args(%arg2 = %153) -> (tensor<1x256x14x14xf32>) {
        %extracted_slice = tensor.extract_slice %153[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %extracted_slice_43 = tensor.extract_slice %46[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %45[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %48[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %47[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x14x14xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x14x14xf32>) attrs =  {tag = "operation_40"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x14x14xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x2x14x14xf32> into tensor<1x256x14x14xf32>
        scf.yield %inserted_slice : tensor<1x256x14x14xf32>
      }
      %155 = scf.for %arg1 = %c0 to %c256 step %c2 iter_args(%arg2 = %151) -> (tensor<1x256x14x14xf32>) {
        %extracted_slice = tensor.extract_slice %154[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x14x14xf32>) outs(%extracted_slice_43 : tensor<1x2x14x14xf32>) attrs =  {tag = "operation_41"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.cmpf ugt, %in, %cst_2 : f32
          %206 = arith.select %205, %in, %cst_2 : f32
          linalg.yield %206 : f32
        } -> tensor<1x2x14x14xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x2x14x14xf32> into tensor<1x256x14x14xf32>
        scf.yield %inserted_slice : tensor<1x256x14x14xf32>
      }
      %padded_26 = tensor.pad %155 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_2 : f32
      } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
      %collapsed_27 = tensor.collapse_shape %44 [[0], [1, 2, 3]] : tensor<256x256x3x3xf32> into tensor<256x2304xf32>
      %collapsed_28 = tensor.collapse_shape %152 [[0], [1], [2, 3]] : tensor<1x256x14x14xf32> into tensor<1x256x196xf32>
      %156 = tensor.empty() : tensor<1x2304x196xf32>
      %157 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel", "parallel"]} outs(%156 : tensor<1x2304x196xf32>) attrs =  {AAAAA} {
      ^bb0(%out: f32):
        %204 = linalg.index 0 : index
        %205 = linalg.index 1 : index
        %206 = linalg.index 2 : index
        %207 = affine.apply #map4(%205)
        %208 = affine.apply #map16(%206, %205)
        %209 = affine.apply #map17(%206, %205)
        %extracted = tensor.extract %padded_26[%204, %207, %208, %209] : tensor<1x256x16x16xf32>
        linalg.yield %extracted : f32
      } -> tensor<1x2304x196xf32>
      %158 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_27, %157 : tensor<256x2304xf32>, tensor<1x2304x196xf32>) outs(%collapsed_28 : tensor<1x256x196xf32>) attrs =  {tag = "operation_42"} {
      ^bb0(%in: f32, %in_43: f32, %out: f32):
        %204 = arith.mulf %in, %in_43 : f32
        %205 = arith.addf %204, %out : f32
        linalg.yield %205 : f32
      } -> tensor<1x256x196xf32>
      %expanded_29 = tensor.expand_shape %158 [[0], [1], [2, 3]] : tensor<1x256x196xf32> into tensor<1x256x14x14xf32>
      %159 = scf.for %arg1 = %c0 to %c256 step %c2 iter_args(%arg2 = %expanded_29) -> (tensor<1x256x14x14xf32>) {
        %extracted_slice = tensor.extract_slice %expanded_29[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %extracted_slice_43 = tensor.extract_slice %41[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %40[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %43[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %42[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x14x14xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x14x14xf32>) attrs =  {tag = "operation_43"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x14x14xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x2x14x14xf32> into tensor<1x256x14x14xf32>
        scf.yield %inserted_slice : tensor<1x256x14x14xf32>
      }
      %collapsed_30 = tensor.collapse_shape %39 [[0], [1, 2, 3]] : tensor<256x128x1x1xf32> into tensor<256x128xf32>
      %collapsed_31 = tensor.collapse_shape %152 [[0], [1], [2, 3]] : tensor<1x256x14x14xf32> into tensor<1x256x196xf32>
      %160 = tensor.empty() : tensor<1x128x196xf32>
      %161 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel", "parallel"]} outs(%160 : tensor<1x128x196xf32>) attrs =  {AAAAA} {
      ^bb0(%out: f32):
        %204 = linalg.index 0 : index
        %205 = linalg.index 1 : index
        %206 = linalg.index 2 : index
        %207 = affine.apply #map18(%206)
        %208 = affine.apply #map19(%206)
        %extracted = tensor.extract %150[%204, %205, %207, %208] : tensor<1x128x28x28xf32>
        linalg.yield %extracted : f32
      } -> tensor<1x128x196xf32>
      %162 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_30, %161 : tensor<256x128xf32>, tensor<1x128x196xf32>) outs(%collapsed_31 : tensor<1x256x196xf32>) attrs =  {tag = "operation_44"} {
      ^bb0(%in: f32, %in_43: f32, %out: f32):
        %204 = arith.mulf %in, %in_43 : f32
        %205 = arith.addf %204, %out : f32
        linalg.yield %205 : f32
      } -> tensor<1x256x196xf32>
      %expanded_32 = tensor.expand_shape %162 [[0], [1], [2, 3]] : tensor<1x256x196xf32> into tensor<1x256x14x14xf32>
      %163 = scf.for %arg1 = %c0 to %c256 step %c2 iter_args(%arg2 = %expanded_32) -> (tensor<1x256x14x14xf32>) {
        %extracted_slice = tensor.extract_slice %expanded_32[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %extracted_slice_43 = tensor.extract_slice %36[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %40[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %38[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %37[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x14x14xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x14x14xf32>) attrs =  {tag = "operation_45"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x14x14xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x2x14x14xf32> into tensor<1x256x14x14xf32>
        scf.yield %inserted_slice : tensor<1x256x14x14xf32>
      }
      %164 = scf.for %arg1 = %c0 to %c256 step %c2 iter_args(%arg2 = %151) -> (tensor<1x256x14x14xf32>) {
        %extracted_slice = tensor.extract_slice %159[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %extracted_slice_43 = tensor.extract_slice %163[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %extracted_slice_44 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43 : tensor<1x2x14x14xf32>, tensor<1x2x14x14xf32>) outs(%extracted_slice_44 : tensor<1x2x14x14xf32>) attrs =  {tag = "operation_46"} {
        ^bb0(%in: f32, %in_45: f32, %out: f32):
          %205 = arith.addf %in, %in_45 : f32
          linalg.yield %205 : f32
        } -> tensor<1x2x14x14xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x2x14x14xf32> into tensor<1x256x14x14xf32>
        scf.yield %inserted_slice : tensor<1x256x14x14xf32>
      }
      %165 = scf.for %arg1 = %c0 to %c256 step %c2 iter_args(%arg2 = %151) -> (tensor<1x256x14x14xf32>) {
        %extracted_slice = tensor.extract_slice %164[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x14x14xf32>) outs(%extracted_slice_43 : tensor<1x2x14x14xf32>) attrs =  {tag = "operation_47"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.cmpf ugt, %in, %cst_2 : f32
          %206 = arith.select %205, %in, %cst_2 : f32
          linalg.yield %206 : f32
        } -> tensor<1x2x14x14xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x2x14x14xf32> into tensor<1x256x14x14xf32>
        scf.yield %inserted_slice : tensor<1x256x14x14xf32>
      }
      %padded_33 = tensor.pad %165 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_2 : f32
      } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
      %166 = scf.for %arg1 = %c0 to %c256 step %c2 iter_args(%arg2 = %152) -> (tensor<1x256x14x14xf32>) {
        %extracted_slice = tensor.extract_slice %35[%arg1, 0, 0, 0] [2, 256, 3, 3] [1, 1, 1, 1] : tensor<256x256x3x3xf32> to tensor<2x256x3x3xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %204 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_48"} ins(%padded_33, %extracted_slice : tensor<1x256x16x16xf32>, tensor<2x256x3x3xf32>) outs(%extracted_slice_43 : tensor<1x2x14x14xf32>) -> tensor<1x2x14x14xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x2x14x14xf32> into tensor<1x256x14x14xf32>
        scf.yield %inserted_slice : tensor<1x256x14x14xf32>
      }
      %167 = scf.for %arg1 = %c0 to %c256 step %c2 iter_args(%arg2 = %166) -> (tensor<1x256x14x14xf32>) {
        %extracted_slice = tensor.extract_slice %166[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %extracted_slice_43 = tensor.extract_slice %32[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %31[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %34[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %33[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x14x14xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x14x14xf32>) attrs =  {tag = "operation_49"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x14x14xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x2x14x14xf32> into tensor<1x256x14x14xf32>
        scf.yield %inserted_slice : tensor<1x256x14x14xf32>
      }
      %168 = scf.for %arg1 = %c0 to %c256 step %c2 iter_args(%arg2 = %151) -> (tensor<1x256x14x14xf32>) {
        %extracted_slice = tensor.extract_slice %167[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x14x14xf32>) outs(%extracted_slice_43 : tensor<1x2x14x14xf32>) attrs =  {tag = "operation_50"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.cmpf ugt, %in, %cst_2 : f32
          %206 = arith.select %205, %in, %cst_2 : f32
          linalg.yield %206 : f32
        } -> tensor<1x2x14x14xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x2x14x14xf32> into tensor<1x256x14x14xf32>
        scf.yield %inserted_slice : tensor<1x256x14x14xf32>
      }
      %padded_34 = tensor.pad %168 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_2 : f32
      } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
      %169 = scf.for %arg1 = %c0 to %c256 step %c2 iter_args(%arg2 = %152) -> (tensor<1x256x14x14xf32>) {
        %extracted_slice = tensor.extract_slice %30[%arg1, 0, 0, 0] [2, 256, 3, 3] [1, 1, 1, 1] : tensor<256x256x3x3xf32> to tensor<2x256x3x3xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %204 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_51"} ins(%padded_34, %extracted_slice : tensor<1x256x16x16xf32>, tensor<2x256x3x3xf32>) outs(%extracted_slice_43 : tensor<1x2x14x14xf32>) -> tensor<1x2x14x14xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x2x14x14xf32> into tensor<1x256x14x14xf32>
        scf.yield %inserted_slice : tensor<1x256x14x14xf32>
      }
      %170 = scf.for %arg1 = %c0 to %c256 step %c2 iter_args(%arg2 = %169) -> (tensor<1x256x14x14xf32>) {
        %extracted_slice = tensor.extract_slice %169[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %extracted_slice_43 = tensor.extract_slice %27[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %26[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %29[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %28[%arg1] [2] [1] : tensor<256xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x14x14xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x14x14xf32>) attrs =  {tag = "operation_52"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x14x14xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x2x14x14xf32> into tensor<1x256x14x14xf32>
        scf.yield %inserted_slice : tensor<1x256x14x14xf32>
      }
      %171 = scf.for %arg1 = %c0 to %c256 step %c2 iter_args(%arg2 = %151) -> (tensor<1x256x14x14xf32>) {
        %extracted_slice = tensor.extract_slice %170[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %extracted_slice_43 = tensor.extract_slice %165[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %extracted_slice_44 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43 : tensor<1x2x14x14xf32>, tensor<1x2x14x14xf32>) outs(%extracted_slice_44 : tensor<1x2x14x14xf32>) attrs =  {tag = "operation_53"} {
        ^bb0(%in: f32, %in_45: f32, %out: f32):
          %205 = arith.addf %in, %in_45 : f32
          linalg.yield %205 : f32
        } -> tensor<1x2x14x14xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x2x14x14xf32> into tensor<1x256x14x14xf32>
        scf.yield %inserted_slice : tensor<1x256x14x14xf32>
      }
      %172 = scf.for %arg1 = %c0 to %c256 step %c2 iter_args(%arg2 = %151) -> (tensor<1x256x14x14xf32>) {
        %extracted_slice = tensor.extract_slice %171[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x2x14x14xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x14x14xf32>) outs(%extracted_slice_43 : tensor<1x2x14x14xf32>) attrs =  {tag = "operation_54"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.cmpf ugt, %in, %cst_2 : f32
          %206 = arith.select %205, %in, %cst_2 : f32
          linalg.yield %206 : f32
        } -> tensor<1x2x14x14xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 14, 14] [1, 1, 1, 1] : tensor<1x2x14x14xf32> into tensor<1x256x14x14xf32>
        scf.yield %inserted_slice : tensor<1x256x14x14xf32>
      }
      %padded_35 = tensor.pad %172 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_2 : f32
      } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
      %173 = tensor.empty() : tensor<1x512x7x7xf32>
      %174 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %173) -> (tensor<1x512x7x7xf32>) {
        %extracted_slice = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %204 = linalg.fill {tag = "operation_55"} ins(%cst_2 : f32) outs(%extracted_slice : tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x2x7x7xf32> into tensor<1x512x7x7xf32>
        scf.yield %inserted_slice : tensor<1x512x7x7xf32>
      }
      %collapsed_36 = tensor.collapse_shape %25 [[0], [1, 2, 3]] : tensor<512x256x3x3xf32> into tensor<512x2304xf32>
      %collapsed_37 = tensor.collapse_shape %174 [[0], [1], [2, 3]] : tensor<1x512x7x7xf32> into tensor<1x512x49xf32>
      %175 = tensor.empty() : tensor<1x2304x49xf32>
      %176 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel", "parallel"]} outs(%175 : tensor<1x2304x49xf32>) attrs =  {AAAAA} {
      ^bb0(%out: f32):
        %204 = linalg.index 0 : index
        %205 = linalg.index 1 : index
        %206 = linalg.index 2 : index
        %207 = affine.apply #map4(%205)
        %208 = affine.apply #map20(%206, %205)
        %209 = affine.apply #map21(%206, %205)
        %extracted = tensor.extract %padded_35[%204, %207, %208, %209] : tensor<1x256x16x16xf32>
        linalg.yield %extracted : f32
      } -> tensor<1x2304x49xf32>
      %177 = linalg.generic {indexing_maps = [#map7, #map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_36, %176 : tensor<512x2304xf32>, tensor<1x2304x49xf32>) outs(%collapsed_37 : tensor<1x512x49xf32>) attrs =  {tag = "operation_56"} {
      ^bb0(%in: f32, %in_43: f32, %out: f32):
        %204 = arith.mulf %in, %in_43 : f32
        %205 = arith.addf %204, %out : f32
        linalg.yield %205 : f32
      } -> tensor<1x512x49xf32>
      %expanded_38 = tensor.expand_shape %177 [[0], [1], [2, 3]] : tensor<1x512x49xf32> into tensor<1x512x7x7xf32>
      %178 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %expanded_38) -> (tensor<1x512x7x7xf32>) {
        %extracted_slice = tensor.extract_slice %expanded_38[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %extracted_slice_43 = tensor.extract_slice %22[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %21[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %24[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %23[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x7x7xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x7x7xf32>) attrs =  {tag = "operation_57"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x7x7xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x2x7x7xf32> into tensor<1x512x7x7xf32>
        scf.yield %inserted_slice : tensor<1x512x7x7xf32>
      }
      %179 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %173) -> (tensor<1x512x7x7xf32>) {
        %extracted_slice = tensor.extract_slice %178[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x7x7xf32>) outs(%extracted_slice_43 : tensor<1x2x7x7xf32>) attrs =  {tag = "operation_58"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.cmpf ugt, %in, %cst_2 : f32
          %206 = arith.select %205, %in, %cst_2 : f32
          linalg.yield %206 : f32
        } -> tensor<1x2x7x7xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x2x7x7xf32> into tensor<1x512x7x7xf32>
        scf.yield %inserted_slice : tensor<1x512x7x7xf32>
      }
      %padded_39 = tensor.pad %179 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_2 : f32
      } : tensor<1x512x7x7xf32> to tensor<1x512x9x9xf32>
      %180 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %174) -> (tensor<1x512x7x7xf32>) {
        %extracted_slice = tensor.extract_slice %20[%arg1, 0, 0, 0] [2, 512, 3, 3] [1, 1, 1, 1] : tensor<512x512x3x3xf32> to tensor<2x512x3x3xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %204 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_59"} ins(%padded_39, %extracted_slice : tensor<1x512x9x9xf32>, tensor<2x512x3x3xf32>) outs(%extracted_slice_43 : tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x2x7x7xf32> into tensor<1x512x7x7xf32>
        scf.yield %inserted_slice : tensor<1x512x7x7xf32>
      }
      %181 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %180) -> (tensor<1x512x7x7xf32>) {
        %extracted_slice = tensor.extract_slice %180[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %extracted_slice_43 = tensor.extract_slice %17[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %16[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %19[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %18[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x7x7xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x7x7xf32>) attrs =  {tag = "operation_60"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x7x7xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x2x7x7xf32> into tensor<1x512x7x7xf32>
        scf.yield %inserted_slice : tensor<1x512x7x7xf32>
      }
      %182 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %174) -> (tensor<1x512x7x7xf32>) {
        %extracted_slice = tensor.extract_slice %172[0, 0, 0, 0] [1, 256, 13, 13] [1, 1, 1, 1] : tensor<1x256x14x14xf32> to tensor<1x256x13x13xf32>
        %extracted_slice_43 = tensor.extract_slice %15[%arg1, 0, 0, 0] [2, 256, 1, 1] [1, 1, 1, 1] : tensor<512x256x1x1xf32> to tensor<2x256x1x1xf32>
        %extracted_slice_44 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %204 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_61"} ins(%extracted_slice, %extracted_slice_43 : tensor<1x256x13x13xf32>, tensor<2x256x1x1xf32>) outs(%extracted_slice_44 : tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x2x7x7xf32> into tensor<1x512x7x7xf32>
        scf.yield %inserted_slice : tensor<1x512x7x7xf32>
      }
      %183 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %182) -> (tensor<1x512x7x7xf32>) {
        %extracted_slice = tensor.extract_slice %182[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %extracted_slice_43 = tensor.extract_slice %12[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %16[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %14[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %13[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x7x7xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x7x7xf32>) attrs =  {tag = "operation_62"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x7x7xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x2x7x7xf32> into tensor<1x512x7x7xf32>
        scf.yield %inserted_slice : tensor<1x512x7x7xf32>
      }
      %184 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %173) -> (tensor<1x512x7x7xf32>) {
        %extracted_slice = tensor.extract_slice %181[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %extracted_slice_43 = tensor.extract_slice %183[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %extracted_slice_44 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43 : tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) outs(%extracted_slice_44 : tensor<1x2x7x7xf32>) attrs =  {tag = "operation_63"} {
        ^bb0(%in: f32, %in_45: f32, %out: f32):
          %205 = arith.addf %in, %in_45 : f32
          linalg.yield %205 : f32
        } -> tensor<1x2x7x7xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x2x7x7xf32> into tensor<1x512x7x7xf32>
        scf.yield %inserted_slice : tensor<1x512x7x7xf32>
      }
      %185 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %173) -> (tensor<1x512x7x7xf32>) {
        %extracted_slice = tensor.extract_slice %184[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x7x7xf32>) outs(%extracted_slice_43 : tensor<1x2x7x7xf32>) attrs =  {tag = "operation_64"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.cmpf ugt, %in, %cst_2 : f32
          %206 = arith.select %205, %in, %cst_2 : f32
          linalg.yield %206 : f32
        } -> tensor<1x2x7x7xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x2x7x7xf32> into tensor<1x512x7x7xf32>
        scf.yield %inserted_slice : tensor<1x512x7x7xf32>
      }
      %padded_40 = tensor.pad %185 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_2 : f32
      } : tensor<1x512x7x7xf32> to tensor<1x512x9x9xf32>
      %186 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %174) -> (tensor<1x512x7x7xf32>) {
        %extracted_slice = tensor.extract_slice %11[%arg1, 0, 0, 0] [2, 512, 3, 3] [1, 1, 1, 1] : tensor<512x512x3x3xf32> to tensor<2x512x3x3xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %204 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_65"} ins(%padded_40, %extracted_slice : tensor<1x512x9x9xf32>, tensor<2x512x3x3xf32>) outs(%extracted_slice_43 : tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x2x7x7xf32> into tensor<1x512x7x7xf32>
        scf.yield %inserted_slice : tensor<1x512x7x7xf32>
      }
      %187 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %186) -> (tensor<1x512x7x7xf32>) {
        %extracted_slice = tensor.extract_slice %186[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %extracted_slice_43 = tensor.extract_slice %8[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %7[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %10[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %9[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x7x7xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x7x7xf32>) attrs =  {tag = "operation_66"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x7x7xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x2x7x7xf32> into tensor<1x512x7x7xf32>
        scf.yield %inserted_slice : tensor<1x512x7x7xf32>
      }
      %188 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %173) -> (tensor<1x512x7x7xf32>) {
        %extracted_slice = tensor.extract_slice %187[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x7x7xf32>) outs(%extracted_slice_43 : tensor<1x2x7x7xf32>) attrs =  {tag = "operation_67"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.cmpf ugt, %in, %cst_2 : f32
          %206 = arith.select %205, %in, %cst_2 : f32
          linalg.yield %206 : f32
        } -> tensor<1x2x7x7xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x2x7x7xf32> into tensor<1x512x7x7xf32>
        scf.yield %inserted_slice : tensor<1x512x7x7xf32>
      }
      %padded_41 = tensor.pad %188 low[0, 0, 1, 1] high[0, 0, 1, 1] {
      ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
        tensor.yield %cst_2 : f32
      } : tensor<1x512x7x7xf32> to tensor<1x512x9x9xf32>
      %189 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %174) -> (tensor<1x512x7x7xf32>) {
        %extracted_slice = tensor.extract_slice %6[%arg1, 0, 0, 0] [2, 512, 3, 3] [1, 1, 1, 1] : tensor<512x512x3x3xf32> to tensor<2x512x3x3xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %204 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_68"} ins(%padded_41, %extracted_slice : tensor<1x512x9x9xf32>, tensor<2x512x3x3xf32>) outs(%extracted_slice_43 : tensor<1x2x7x7xf32>) -> tensor<1x2x7x7xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x2x7x7xf32> into tensor<1x512x7x7xf32>
        scf.yield %inserted_slice : tensor<1x512x7x7xf32>
      }
      %190 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %189) -> (tensor<1x512x7x7xf32>) {
        %extracted_slice = tensor.extract_slice %189[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %extracted_slice_43 = tensor.extract_slice %3[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_44 = tensor.extract_slice %2[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_45 = tensor.extract_slice %5[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_46 = tensor.extract_slice %4[%arg1] [2] [1] : tensor<512xf32> to tensor<2xf32>
        %extracted_slice_47 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43, %extracted_slice_44, %extracted_slice_45, %extracted_slice_46 : tensor<1x2x7x7xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) outs(%extracted_slice_47 : tensor<1x2x7x7xf32>) attrs =  {tag = "operation_69"} {
        ^bb0(%in: f32, %in_48: f32, %in_49: f32, %in_50: f32, %in_51: f32, %out: f32):
          %205 = arith.truncf %cst_0 : f64 to f32
          %206 = arith.addf %in_51, %205 : f32
          %207 = math.rsqrt %206 : f32
          %208 = arith.subf %in, %in_50 : f32
          %209 = arith.mulf %208, %207 : f32
          %210 = arith.mulf %209, %in_48 : f32
          %211 = arith.addf %210, %in_49 : f32
          linalg.yield %211 : f32
        } -> tensor<1x2x7x7xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x2x7x7xf32> into tensor<1x512x7x7xf32>
        scf.yield %inserted_slice : tensor<1x512x7x7xf32>
      }
      %191 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %173) -> (tensor<1x512x7x7xf32>) {
        %extracted_slice = tensor.extract_slice %190[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %extracted_slice_43 = tensor.extract_slice %185[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %extracted_slice_44 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_43 : tensor<1x2x7x7xf32>, tensor<1x2x7x7xf32>) outs(%extracted_slice_44 : tensor<1x2x7x7xf32>) attrs =  {tag = "operation_70"} {
        ^bb0(%in: f32, %in_45: f32, %out: f32):
          %205 = arith.addf %in, %in_45 : f32
          linalg.yield %205 : f32
        } -> tensor<1x2x7x7xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x2x7x7xf32> into tensor<1x512x7x7xf32>
        scf.yield %inserted_slice : tensor<1x512x7x7xf32>
      }
      %192 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %173) -> (tensor<1x512x7x7xf32>) {
        %extracted_slice = tensor.extract_slice %191[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %204 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x7x7xf32>) outs(%extracted_slice_43 : tensor<1x2x7x7xf32>) attrs =  {tag = "operation_71"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.cmpf ugt, %in, %cst_2 : f32
          %206 = arith.select %205, %in, %cst_2 : f32
          linalg.yield %206 : f32
        } -> tensor<1x2x7x7xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x2x7x7xf32> into tensor<1x512x7x7xf32>
        scf.yield %inserted_slice : tensor<1x512x7x7xf32>
      }
      %193 = tensor.empty() : tensor<1x512x1x1xf32>
      %194 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %193) -> (tensor<1x512x1x1xf32>) {
        %extracted_slice = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 1, 1] [1, 1, 1, 1] : tensor<1x512x1x1xf32> to tensor<1x2x1x1xf32>
        %204 = linalg.fill {tag = "operation_72"} ins(%cst_2 : f32) outs(%extracted_slice : tensor<1x2x1x1xf32>) -> tensor<1x2x1x1xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 1, 1] [1, 1, 1, 1] : tensor<1x2x1x1xf32> into tensor<1x512x1x1xf32>
        scf.yield %inserted_slice : tensor<1x512x1x1xf32>
      }
      %195 = tensor.empty() : tensor<7x7xf32>
      %196 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %194) -> (tensor<1x512x1x1xf32>) {
        %extracted_slice = tensor.extract_slice %192[0, %arg1, 0, 0] [1, 2, 7, 7] [1, 1, 1, 1] : tensor<1x512x7x7xf32> to tensor<1x2x7x7xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 1, 1] [1, 1, 1, 1] : tensor<1x512x1x1xf32> to tensor<1x2x1x1xf32>
        %204 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_73"} ins(%extracted_slice, %195 : tensor<1x2x7x7xf32>, tensor<7x7xf32>) outs(%extracted_slice_43 : tensor<1x2x1x1xf32>) -> tensor<1x2x1x1xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 1, 1] [1, 1, 1, 1] : tensor<1x2x1x1xf32> into tensor<1x512x1x1xf32>
        scf.yield %inserted_slice : tensor<1x512x1x1xf32>
      }
      %197 = scf.for %arg1 = %c0 to %c512 step %c2 iter_args(%arg2 = %193) -> (tensor<1x512x1x1xf32>) {
        %extracted_slice = tensor.extract_slice %196[0, %arg1, 0, 0] [1, 2, 1, 1] [1, 1, 1, 1] : tensor<1x512x1x1xf32> to tensor<1x2x1x1xf32>
        %extracted_slice_43 = tensor.extract_slice %arg2[0, %arg1, 0, 0] [1, 2, 1, 1] [1, 1, 1, 1] : tensor<1x512x1x1xf32> to tensor<1x2x1x1xf32>
        %204 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x2x1x1xf32>) outs(%extracted_slice_43 : tensor<1x2x1x1xf32>) attrs =  {tag = "operation_74"} {
        ^bb0(%in: f32, %out: f32):
          %205 = arith.divf %in, %cst : f32
          linalg.yield %205 : f32
        } -> tensor<1x2x1x1xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1, 0, 0] [1, 2, 1, 1] [1, 1, 1, 1] : tensor<1x2x1x1xf32> into tensor<1x512x1x1xf32>
        scf.yield %inserted_slice : tensor<1x512x1x1xf32>
      }
      %collapsed_42 = tensor.collapse_shape %197 [[0], [1, 2, 3]] : tensor<1x512x1x1xf32> into tensor<1x512xf32>
      %198 = tensor.empty() : tensor<512x1000xf32>
      %199 = scf.for %arg1 = %c0 to %c1000 step %c1 iter_args(%arg2 = %198) -> (tensor<512x1000xf32>) {
        %204 = scf.for %arg3 = %c0 to %c512 step %c2 iter_args(%arg4 = %arg2) -> (tensor<512x1000xf32>) {
          %extracted_slice = tensor.extract_slice %1[%arg1, %arg3] [1, 2] [1, 1] : tensor<1000x512xf32> to tensor<1x2xf32>
          %extracted_slice_43 = tensor.extract_slice %arg4[%arg3, %arg1] [2, 1] [1, 1] : tensor<512x1000xf32> to tensor<2x1xf32>
          %205 = linalg.generic {indexing_maps = [#map22, #map23], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice : tensor<1x2xf32>) outs(%extracted_slice_43 : tensor<2x1xf32>) attrs =  {tag = "operation_75"} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<2x1xf32>
          %inserted_slice = tensor.insert_slice %205 into %arg4[%arg3, %arg1] [2, 1] [1, 1] : tensor<2x1xf32> into tensor<512x1000xf32>
          scf.yield %inserted_slice : tensor<512x1000xf32>
        }
        scf.yield %204 : tensor<512x1000xf32>
      }
      %200 = tensor.empty() : tensor<1x1000xf32>
      %201 = scf.for %arg1 = %c0 to %c1000 step %c2 iter_args(%arg2 = %200) -> (tensor<1x1000xf32>) {
        %extracted_slice = tensor.extract_slice %arg2[0, %arg1] [1, 2] [1, 1] : tensor<1x1000xf32> to tensor<1x2xf32>
        %204 = linalg.fill {tag = "operation_76"} ins(%cst_2 : f32) outs(%extracted_slice : tensor<1x2xf32>) -> tensor<1x2xf32>
        %inserted_slice = tensor.insert_slice %204 into %arg2[0, %arg1] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<1x1000xf32>
        scf.yield %inserted_slice : tensor<1x1000xf32>
      }
      %202 = linalg.matmul {tag = "operation_77"} ins(%collapsed_42, %199 : tensor<1x512xf32>, tensor<512x1000xf32>) outs(%201 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
      %203 = linalg.generic {indexing_maps = [#map24, #map25, #map22], iterator_types = ["parallel", "parallel"]} ins(%202, %0 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%200 : tensor<1x1000xf32>) attrs =  {tag = "operation_78"} {
      ^bb0(%in: f32, %in_43: f32, %out: f32):
        %204 = arith.addf %in, %in_43 : f32
        linalg.yield %204 : f32
      } -> tensor<1x1000xf32>
      return %203 : tensor<1x1000xf32>
    }
    func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
    func.func private @printFlops(f64)
    func.func private @printI64(i64)
    func.func private @printNewline()
    func.func @main() {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %0 = bufferization.alloc_tensor() : tensor<1x3x224x224xf32>
      %1 = linalg.fill {tag = "operation_79"} ins(%cst : f32) outs(%0 : tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32>
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %2 = func.call @nanoTime() : () -> i64
        %3 = func.call @forward(%1) : (tensor<1x3x224x224xf32>) -> tensor<1x1000xf32>
        %4 = func.call @nanoTime() : () -> i64
        %5 = arith.subi %4, %2 : i64
        func.call @printI64(%5) : (i64) -> ()
        func.call @printNewline() : () -> ()
      }
      return
    }
  }
  
