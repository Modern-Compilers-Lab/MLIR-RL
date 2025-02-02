#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1, d0)>
#map5 = affine_map<(d0, d1) -> (0, d1)>
#map6 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "Net"} {
  memref.global "private" @global_seed : memref<i64> = dense<0>
  func.func @forward(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %false = arith.constant false
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
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 1.000000e-05 : f64
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant 4.900000e+01 : f32
    %padded = tensor.pad %arg0 low[0, 0, 3, 3] high[0, 0, 3, 3] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x3x224x224xf32> to tensor<1x3x230x230xf32>
    %99 = tensor.empty() : tensor<1x64x112x112xf32>
    %100 = linalg.fill {tag = "operation_0"} ins(%cst : f32) outs(%99 : tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %101 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_1"} ins(%padded, %98 : tensor<1x3x230x230xf32>, tensor<64x3x7x7xf32>) outs(%100 : tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %102 = arith.cmpi eq, %false, %false : i1
    %103 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%101, %95, %94, %97, %96 : tensor<1x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%101 : tensor<1x64x112x112xf32>) attrs =  {tag = "operation_2"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x64x112x112xf32>
    %104 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%103 : tensor<1x64x112x112xf32>) outs(%99 : tensor<1x64x112x112xf32>) attrs =  {tag = "operation_3"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.cmpf ugt, %in, %cst : f32
      %190 = arith.select %189, %in, %cst : f32
      linalg.yield %190 : f32
    } -> tensor<1x64x112x112xf32>
    %padded_3 = tensor.pad %104 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x64x112x112xf32> to tensor<1x64x114x114xf32>
    %105 = tensor.empty() : tensor<1x64x56x56xf32>
    %106 = linalg.fill {tag = "operation_4"} ins(%cst_0 : f32) outs(%105 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %107 = tensor.empty() : tensor<3x3xf32>
    %108 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_5"} ins(%padded_3, %107 : tensor<1x64x114x114xf32>, tensor<3x3xf32>) outs(%106 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %padded_4 = tensor.pad %108 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %109 = linalg.fill {tag = "operation_6"} ins(%cst : f32) outs(%105 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %110 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_7"} ins(%padded_4, %93 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%109 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %111 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%110, %90, %89, %92, %91 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%110 : tensor<1x64x56x56xf32>) attrs =  {tag = "operation_8"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x64x56x56xf32>
    %112 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%111 : tensor<1x64x56x56xf32>) outs(%105 : tensor<1x64x56x56xf32>) attrs =  {tag = "operation_9"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.cmpf ugt, %in, %cst : f32
      %190 = arith.select %189, %in, %cst : f32
      linalg.yield %190 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_5 = tensor.pad %112 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %113 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_10"} ins(%padded_5, %88 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%109 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %114 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%113, %85, %84, %87, %86 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%113 : tensor<1x64x56x56xf32>) attrs =  {tag = "operation_11"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x64x56x56xf32>
    %115 = linalg.generic {indexing_maps = [#map2, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%114, %108 : tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) outs(%105 : tensor<1x64x56x56xf32>) attrs =  {tag = "operation_12"} {
    ^bb0(%in: f32, %in_20: f32, %out: f32):
      %189 = arith.addf %in, %in_20 : f32
      linalg.yield %189 : f32
    } -> tensor<1x64x56x56xf32>
    %116 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%115 : tensor<1x64x56x56xf32>) outs(%105 : tensor<1x64x56x56xf32>) attrs =  {tag = "operation_13"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.cmpf ugt, %in, %cst : f32
      %190 = arith.select %189, %in, %cst : f32
      linalg.yield %190 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_6 = tensor.pad %116 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %117 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_14"} ins(%padded_6, %83 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%109 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %118 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%117, %80, %79, %82, %81 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%117 : tensor<1x64x56x56xf32>) attrs =  {tag = "operation_15"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x64x56x56xf32>
    %119 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%118 : tensor<1x64x56x56xf32>) outs(%105 : tensor<1x64x56x56xf32>) attrs =  {tag = "operation_16"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.cmpf ugt, %in, %cst : f32
      %190 = arith.select %189, %in, %cst : f32
      linalg.yield %190 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_7 = tensor.pad %119 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %120 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_17"} ins(%padded_7, %78 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%109 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %121 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120, %75, %74, %77, %76 : tensor<1x64x56x56xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%120 : tensor<1x64x56x56xf32>) attrs =  {tag = "operation_18"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x64x56x56xf32>
    %122 = linalg.generic {indexing_maps = [#map2, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%121, %116 : tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) outs(%105 : tensor<1x64x56x56xf32>) attrs =  {tag = "operation_19"} {
    ^bb0(%in: f32, %in_20: f32, %out: f32):
      %189 = arith.addf %in, %in_20 : f32
      linalg.yield %189 : f32
    } -> tensor<1x64x56x56xf32>
    %123 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%122 : tensor<1x64x56x56xf32>) outs(%105 : tensor<1x64x56x56xf32>) attrs =  {tag = "operation_20"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.cmpf ugt, %in, %cst : f32
      %190 = arith.select %189, %in, %cst : f32
      linalg.yield %190 : f32
    } -> tensor<1x64x56x56xf32>
    %padded_8 = tensor.pad %123 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x64x56x56xf32> to tensor<1x64x58x58xf32>
    %124 = tensor.empty() : tensor<1x128x28x28xf32>
    %125 = linalg.fill {tag = "operation_21"} ins(%cst : f32) outs(%124 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %126 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_22"} ins(%padded_8, %73 : tensor<1x64x58x58xf32>, tensor<128x64x3x3xf32>) outs(%125 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %127 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%126, %70, %69, %72, %71 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%126 : tensor<1x128x28x28xf32>) attrs =  {tag = "operation_23"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x28x28xf32>
    %128 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%127 : tensor<1x128x28x28xf32>) outs(%124 : tensor<1x128x28x28xf32>) attrs =  {tag = "operation_24"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.cmpf ugt, %in, %cst : f32
      %190 = arith.select %189, %in, %cst : f32
      linalg.yield %190 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_9 = tensor.pad %128 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %129 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_25"} ins(%padded_9, %68 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%125 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %130 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%129, %65, %64, %67, %66 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%129 : tensor<1x128x28x28xf32>) attrs =  {tag = "operation_26"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x28x28xf32>
    %131 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_27"} ins(%123, %63 : tensor<1x64x56x56xf32>, tensor<128x64x1x1xf32>) outs(%125 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %132 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%131, %60, %64, %62, %61 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%131 : tensor<1x128x28x28xf32>) attrs =  {tag = "operation_28"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x28x28xf32>
    %133 = linalg.generic {indexing_maps = [#map2, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%130, %132 : tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) outs(%124 : tensor<1x128x28x28xf32>) attrs =  {tag = "operation_29"} {
    ^bb0(%in: f32, %in_20: f32, %out: f32):
      %189 = arith.addf %in, %in_20 : f32
      linalg.yield %189 : f32
    } -> tensor<1x128x28x28xf32>
    %134 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%133 : tensor<1x128x28x28xf32>) outs(%124 : tensor<1x128x28x28xf32>) attrs =  {tag = "operation_30"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.cmpf ugt, %in, %cst : f32
      %190 = arith.select %189, %in, %cst : f32
      linalg.yield %190 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_10 = tensor.pad %134 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %135 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_31"} ins(%padded_10, %59 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%125 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %136 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%135, %56, %55, %58, %57 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%135 : tensor<1x128x28x28xf32>) attrs =  {tag = "operation_32"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x28x28xf32>
    %137 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%136 : tensor<1x128x28x28xf32>) outs(%124 : tensor<1x128x28x28xf32>) attrs =  {tag = "operation_33"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.cmpf ugt, %in, %cst : f32
      %190 = arith.select %189, %in, %cst : f32
      linalg.yield %190 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_11 = tensor.pad %137 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %138 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_34"} ins(%padded_11, %54 : tensor<1x128x30x30xf32>, tensor<128x128x3x3xf32>) outs(%125 : tensor<1x128x28x28xf32>) -> tensor<1x128x28x28xf32>
    %139 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%138, %51, %50, %53, %52 : tensor<1x128x28x28xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) outs(%138 : tensor<1x128x28x28xf32>) attrs =  {tag = "operation_35"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x128x28x28xf32>
    %140 = linalg.generic {indexing_maps = [#map2, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%139, %134 : tensor<1x128x28x28xf32>, tensor<1x128x28x28xf32>) outs(%124 : tensor<1x128x28x28xf32>) attrs =  {tag = "operation_36"} {
    ^bb0(%in: f32, %in_20: f32, %out: f32):
      %189 = arith.addf %in, %in_20 : f32
      linalg.yield %189 : f32
    } -> tensor<1x128x28x28xf32>
    %141 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%140 : tensor<1x128x28x28xf32>) outs(%124 : tensor<1x128x28x28xf32>) attrs =  {tag = "operation_37"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.cmpf ugt, %in, %cst : f32
      %190 = arith.select %189, %in, %cst : f32
      linalg.yield %190 : f32
    } -> tensor<1x128x28x28xf32>
    %padded_12 = tensor.pad %141 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x128x28x28xf32> to tensor<1x128x30x30xf32>
    %142 = tensor.empty() : tensor<1x256x14x14xf32>
    %143 = linalg.fill {tag = "operation_38"} ins(%cst : f32) outs(%142 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %144 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_39"} ins(%padded_12, %49 : tensor<1x128x30x30xf32>, tensor<256x128x3x3xf32>) outs(%143 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %145 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%144, %46, %45, %48, %47 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%144 : tensor<1x256x14x14xf32>) attrs =  {tag = "operation_40"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x256x14x14xf32>
    %146 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%145 : tensor<1x256x14x14xf32>) outs(%142 : tensor<1x256x14x14xf32>) attrs =  {tag = "operation_41"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.cmpf ugt, %in, %cst : f32
      %190 = arith.select %189, %in, %cst : f32
      linalg.yield %190 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_13 = tensor.pad %146 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %147 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_42"} ins(%padded_13, %44 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%143 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %148 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%147, %41, %40, %43, %42 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%147 : tensor<1x256x14x14xf32>) attrs =  {tag = "operation_43"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x256x14x14xf32>
    %149 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_44"} ins(%141, %39 : tensor<1x128x28x28xf32>, tensor<256x128x1x1xf32>) outs(%143 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %150 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%149, %36, %40, %38, %37 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%149 : tensor<1x256x14x14xf32>) attrs =  {tag = "operation_45"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x256x14x14xf32>
    %151 = linalg.generic {indexing_maps = [#map2, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%148, %150 : tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) outs(%142 : tensor<1x256x14x14xf32>) attrs =  {tag = "operation_46"} {
    ^bb0(%in: f32, %in_20: f32, %out: f32):
      %189 = arith.addf %in, %in_20 : f32
      linalg.yield %189 : f32
    } -> tensor<1x256x14x14xf32>
    %152 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%151 : tensor<1x256x14x14xf32>) outs(%142 : tensor<1x256x14x14xf32>) attrs =  {tag = "operation_47"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.cmpf ugt, %in, %cst : f32
      %190 = arith.select %189, %in, %cst : f32
      linalg.yield %190 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_14 = tensor.pad %152 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %153 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_48"} ins(%padded_14, %35 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%143 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %154 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%153, %32, %31, %34, %33 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%153 : tensor<1x256x14x14xf32>) attrs =  {tag = "operation_49"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x256x14x14xf32>
    %155 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%154 : tensor<1x256x14x14xf32>) outs(%142 : tensor<1x256x14x14xf32>) attrs =  {tag = "operation_50"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.cmpf ugt, %in, %cst : f32
      %190 = arith.select %189, %in, %cst : f32
      linalg.yield %190 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_15 = tensor.pad %155 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %156 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_51"} ins(%padded_15, %30 : tensor<1x256x16x16xf32>, tensor<256x256x3x3xf32>) outs(%143 : tensor<1x256x14x14xf32>) -> tensor<1x256x14x14xf32>
    %157 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%156, %27, %26, %29, %28 : tensor<1x256x14x14xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>) outs(%156 : tensor<1x256x14x14xf32>) attrs =  {tag = "operation_52"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x256x14x14xf32>
    %158 = linalg.generic {indexing_maps = [#map2, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%157, %152 : tensor<1x256x14x14xf32>, tensor<1x256x14x14xf32>) outs(%142 : tensor<1x256x14x14xf32>) attrs =  {tag = "operation_53"} {
    ^bb0(%in: f32, %in_20: f32, %out: f32):
      %189 = arith.addf %in, %in_20 : f32
      linalg.yield %189 : f32
    } -> tensor<1x256x14x14xf32>
    %159 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%158 : tensor<1x256x14x14xf32>) outs(%142 : tensor<1x256x14x14xf32>) attrs =  {tag = "operation_54"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.cmpf ugt, %in, %cst : f32
      %190 = arith.select %189, %in, %cst : f32
      linalg.yield %190 : f32
    } -> tensor<1x256x14x14xf32>
    %padded_16 = tensor.pad %159 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x256x14x14xf32> to tensor<1x256x16x16xf32>
    %160 = tensor.empty() : tensor<1x512x7x7xf32>
    %161 = linalg.fill {tag = "operation_55"} ins(%cst : f32) outs(%160 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %162 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_56"} ins(%padded_16, %25 : tensor<1x256x16x16xf32>, tensor<512x256x3x3xf32>) outs(%161 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %163 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%162, %22, %21, %24, %23 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%162 : tensor<1x512x7x7xf32>) attrs =  {tag = "operation_57"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x512x7x7xf32>
    %164 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%163 : tensor<1x512x7x7xf32>) outs(%160 : tensor<1x512x7x7xf32>) attrs =  {tag = "operation_58"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.cmpf ugt, %in, %cst : f32
      %190 = arith.select %189, %in, %cst : f32
      linalg.yield %190 : f32
    } -> tensor<1x512x7x7xf32>
    %padded_17 = tensor.pad %164 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x512x7x7xf32> to tensor<1x512x9x9xf32>
    %165 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_59"} ins(%padded_17, %20 : tensor<1x512x9x9xf32>, tensor<512x512x3x3xf32>) outs(%161 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %166 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%165, %17, %16, %19, %18 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%165 : tensor<1x512x7x7xf32>) attrs =  {tag = "operation_60"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x512x7x7xf32>
    %167 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_61"} ins(%159, %15 : tensor<1x256x14x14xf32>, tensor<512x256x1x1xf32>) outs(%161 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %168 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%167, %12, %16, %14, %13 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%167 : tensor<1x512x7x7xf32>) attrs =  {tag = "operation_62"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x512x7x7xf32>
    %169 = linalg.generic {indexing_maps = [#map2, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%166, %168 : tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) outs(%160 : tensor<1x512x7x7xf32>) attrs =  {tag = "operation_63"} {
    ^bb0(%in: f32, %in_20: f32, %out: f32):
      %189 = arith.addf %in, %in_20 : f32
      linalg.yield %189 : f32
    } -> tensor<1x512x7x7xf32>
    %170 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%169 : tensor<1x512x7x7xf32>) outs(%160 : tensor<1x512x7x7xf32>) attrs =  {tag = "operation_64"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.cmpf ugt, %in, %cst : f32
      %190 = arith.select %189, %in, %cst : f32
      linalg.yield %190 : f32
    } -> tensor<1x512x7x7xf32>
    %padded_18 = tensor.pad %170 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x512x7x7xf32> to tensor<1x512x9x9xf32>
    %171 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_65"} ins(%padded_18, %11 : tensor<1x512x9x9xf32>, tensor<512x512x3x3xf32>) outs(%161 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %172 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%171, %8, %7, %10, %9 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%171 : tensor<1x512x7x7xf32>) attrs =  {tag = "operation_66"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x512x7x7xf32>
    %173 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%172 : tensor<1x512x7x7xf32>) outs(%160 : tensor<1x512x7x7xf32>) attrs =  {tag = "operation_67"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.cmpf ugt, %in, %cst : f32
      %190 = arith.select %189, %in, %cst : f32
      linalg.yield %190 : f32
    } -> tensor<1x512x7x7xf32>
    %padded_19 = tensor.pad %173 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst : f32
    } : tensor<1x512x7x7xf32> to tensor<1x512x9x9xf32>
    %174 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_68"} ins(%padded_19, %6 : tensor<1x512x9x9xf32>, tensor<512x512x3x3xf32>) outs(%161 : tensor<1x512x7x7xf32>) -> tensor<1x512x7x7xf32>
    %175 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%174, %3, %2, %5, %4 : tensor<1x512x7x7xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) outs(%174 : tensor<1x512x7x7xf32>) attrs =  {tag = "operation_69"} {
    ^bb0(%in: f32, %in_20: f32, %in_21: f32, %in_22: f32, %in_23: f32, %out: f32):
      %189 = arith.truncf %cst_1 : f64 to f32
      %190 = arith.addf %in_23, %189 : f32
      %191 = math.rsqrt %190 : f32
      %192 = arith.subf %in, %in_22 : f32
      %193 = arith.mulf %192, %191 : f32
      %194 = arith.mulf %193, %in_20 : f32
      %195 = arith.addf %194, %in_21 : f32
      linalg.yield %195 : f32
    } -> tensor<1x512x7x7xf32>
    %176 = linalg.generic {indexing_maps = [#map2, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%175, %170 : tensor<1x512x7x7xf32>, tensor<1x512x7x7xf32>) outs(%160 : tensor<1x512x7x7xf32>) attrs =  {tag = "operation_70"} {
    ^bb0(%in: f32, %in_20: f32, %out: f32):
      %189 = arith.addf %in, %in_20 : f32
      linalg.yield %189 : f32
    } -> tensor<1x512x7x7xf32>
    %177 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%176 : tensor<1x512x7x7xf32>) outs(%160 : tensor<1x512x7x7xf32>) attrs =  {tag = "operation_71"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.cmpf ugt, %in, %cst : f32
      %190 = arith.select %189, %in, %cst : f32
      linalg.yield %190 : f32
    } -> tensor<1x512x7x7xf32>
    %178 = tensor.empty() : tensor<1x512x1x1xf32>
    %179 = linalg.fill {tag = "operation_72"} ins(%cst : f32) outs(%178 : tensor<1x512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %180 = tensor.empty() : tensor<7x7xf32>
    %181 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_73"} ins(%177, %180 : tensor<1x512x7x7xf32>, tensor<7x7xf32>) outs(%179 : tensor<1x512x1x1xf32>) -> tensor<1x512x1x1xf32>
    %182 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%181 : tensor<1x512x1x1xf32>) outs(%178 : tensor<1x512x1x1xf32>) attrs =  {tag = "operation_74"} {
    ^bb0(%in: f32, %out: f32):
      %189 = arith.divf %in, %cst_2 : f32
      linalg.yield %189 : f32
    } -> tensor<1x512x1x1xf32>
    %collapsed = tensor.collapse_shape %182 [[0], [1, 2, 3]] : tensor<1x512x1x1xf32> into tensor<1x512xf32>
    %183 = tensor.empty() : tensor<512x1000xf32>
    %184 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<1000x512xf32>) outs(%183 : tensor<512x1000xf32>) attrs =  {tag = "operation_75"} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<512x1000xf32>
    %185 = tensor.empty() : tensor<1x1000xf32>
    %186 = linalg.fill {tag = "operation_76"} ins(%cst : f32) outs(%185 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %187 = linalg.matmul {tag = "operation_77"} ins(%collapsed, %184 : tensor<1x512xf32>, tensor<512x1000xf32>) outs(%186 : tensor<1x1000xf32>) -> tensor<1x1000xf32>
    %188 = linalg.generic {indexing_maps = [#map5, #map6, #map3], iterator_types = ["parallel", "parallel"]} ins(%187, %0 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%185 : tensor<1x1000xf32>) attrs =  {tag = "operation_78"} {
    ^bb0(%in: f32, %in_20: f32, %out: f32):
      %189 = arith.addf %in, %in_20 : f32
      linalg.yield %189 : f32
    } -> tensor<1x1000xf32>
    return %188 : tensor<1x1000xf32>
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
      %6 = arith.uitofp %5 : i64 to f64
      func.call @printI64(%5) : (i64) -> ()
      func.call @printNewline() : () -> ()
    }
    return
  }
}