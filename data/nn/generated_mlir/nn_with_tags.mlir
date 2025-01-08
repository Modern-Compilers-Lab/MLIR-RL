#map = affine_map<(d0, d1, d2, d3) -> (d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1, d0)>
#map4 = affine_map<(d0, d1) -> (d1)>
module attributes {torch.debug_module_name = "Net"} {
  memref.global "private" @global_seed : memref<i64> = dense<0>
  func.func @forward() -> tensor<32x10xf32> {
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
    %13 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_1"} ins(%0, %1 : tensor<32x3x230x230xf32>, tensor<64x3x7x7xf32>) outs(%12 : tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xf32>
    %14 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13 : tensor<32x64x112x112xf32>) outs(%11 : tensor<32x64x112x112xf32>) attrs =  {tag = "operation_2"} {
    ^bb0(%in: f32, %out: f32):
      %cst_1 = arith.constant 0.000000e+00 : f32
      %46 = arith.cmpf ugt, %in, %cst_1 : f32
      %47 = arith.select %46, %in, %cst_1 : f32
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
    %21 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, tag = "operation_6"} ins(%18, %3 : tensor<32x64x56x56xf32>, tensor<16x64x5x5xf32>) outs(%20 : tensor<32x16x52x52xf32>) -> tensor<32x16x52x52xf32>
    %22 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21 : tensor<32x16x52x52xf32>) outs(%19 : tensor<32x16x52x52xf32>) attrs =  {tag = "operation_7"} {
    ^bb0(%in: f32, %out: f32):
      %cst_1 = arith.constant 0.000000e+00 : f32
      %46 = arith.cmpf ugt, %in, %cst_1 : f32
      %47 = arith.select %46, %in, %cst_1 : f32
      linalg.yield %47 : f32
    } -> tensor<32x16x52x52xf32>
    %23 = tensor.empty() : tensor<32x16x26x26xf32>
    %24 = linalg.fill {tag = "operation_8"} ins(%cst : f32) outs(%23 : tensor<32x16x26x26xf32>) -> tensor<32x16x26x26xf32>
    %25 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_9"} ins(%22, %17 : tensor<32x16x52x52xf32>, tensor<2x2xf32>) outs(%24 : tensor<32x16x26x26xf32>) -> tensor<32x16x26x26xf32>
    %collapsed = tensor.collapse_shape %25 [[0], [1, 2, 3]] : tensor<32x16x26x26xf32> into tensor<32x10816xf32>
    %26 = tensor.empty() : tensor<10816x120xf32>
    %27 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<120x10816xf32>) outs(%26 : tensor<10816x120xf32>) attrs =  {tag = "operation_10"} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<10816x120xf32>
    %28 = tensor.empty() : tensor<32x120xf32>
    %29 = linalg.fill {tag = "operation_11"} ins(%cst_0 : f32) outs(%28 : tensor<32x120xf32>) -> tensor<32x120xf32>
    %30 = linalg.matmul {tag = "operation_12"} ins(%collapsed, %27 : tensor<32x10816xf32>, tensor<10816x120xf32>) outs(%29 : tensor<32x120xf32>) -> tensor<32x120xf32>
    %31 = linalg.generic {indexing_maps = [#map2, #map4, #map2], iterator_types = ["parallel", "parallel"]} ins(%30, %6 : tensor<32x120xf32>, tensor<120xf32>) outs(%28 : tensor<32x120xf32>) attrs =  {tag = "operation_13"} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %46 = arith.addf %in, %in_1 : f32
      linalg.yield %46 : f32
    } -> tensor<32x120xf32>
    %32 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%31 : tensor<32x120xf32>) outs(%28 : tensor<32x120xf32>) attrs =  {tag = "operation_14"} {
    ^bb0(%in: f32, %out: f32):
      %cst_1 = arith.constant 0.000000e+00 : f32
      %46 = arith.cmpf ugt, %in, %cst_1 : f32
      %47 = arith.select %46, %in, %cst_1 : f32
      linalg.yield %47 : f32
    } -> tensor<32x120xf32>
    %33 = tensor.empty() : tensor<120x84xf32>
    %34 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%7 : tensor<84x120xf32>) outs(%33 : tensor<120x84xf32>) attrs =  {tag = "operation_15"} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<120x84xf32>
    %35 = tensor.empty() : tensor<32x84xf32>
    %36 = linalg.fill {tag = "operation_16"} ins(%cst_0 : f32) outs(%35 : tensor<32x84xf32>) -> tensor<32x84xf32>
    %37 = linalg.matmul {tag = "operation_17"} ins(%32, %34 : tensor<32x120xf32>, tensor<120x84xf32>) outs(%36 : tensor<32x84xf32>) -> tensor<32x84xf32>
    %38 = linalg.generic {indexing_maps = [#map2, #map4, #map2], iterator_types = ["parallel", "parallel"]} ins(%37, %8 : tensor<32x84xf32>, tensor<84xf32>) outs(%35 : tensor<32x84xf32>) attrs =  {tag = "operation_18"} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %46 = arith.addf %in, %in_1 : f32
      linalg.yield %46 : f32
    } -> tensor<32x84xf32>
    %39 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%38 : tensor<32x84xf32>) outs(%35 : tensor<32x84xf32>) attrs =  {tag = "operation_19"} {
    ^bb0(%in: f32, %out: f32):
      %cst_1 = arith.constant 0.000000e+00 : f32
      %46 = arith.cmpf ugt, %in, %cst_1 : f32
      %47 = arith.select %46, %in, %cst_1 : f32
      linalg.yield %47 : f32
    } -> tensor<32x84xf32>
    %40 = tensor.empty() : tensor<84x10xf32>
    %41 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<10x84xf32>) outs(%40 : tensor<84x10xf32>) attrs =  {tag = "operation_20"} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<84x10xf32>
    %42 = tensor.empty() : tensor<32x10xf32>
    %43 = linalg.fill {tag = "operation_21"} ins(%cst_0 : f32) outs(%42 : tensor<32x10xf32>) -> tensor<32x10xf32>
    %44 = linalg.matmul {tag = "operation_22"} ins(%39, %41 : tensor<32x84xf32>, tensor<84x10xf32>) outs(%43 : tensor<32x10xf32>) -> tensor<32x10xf32>
    %45 = linalg.generic {indexing_maps = [#map2, #map4, #map2], iterator_types = ["parallel", "parallel"]} ins(%44, %10 : tensor<32x10xf32>, tensor<10xf32>) outs(%42 : tensor<32x10xf32>) attrs =  {tag = "operation_23"} {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %46 = arith.addf %in, %in_1 : f32
      linalg.yield %46 : f32
    } -> tensor<32x10xf32>
    return %45 : tensor<32x10xf32>
  }
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @nanoTime() : () -> i64
      %1 = func.call @forward() : () -> tensor<32x10xf32>
      %2 = func.call @nanoTime() : () -> i64
      %3 = arith.subi %2, %0 : i64
      %4 = arith.uitofp %3 : i64 to f64
      func.call @printFlops(%4) : (f64) -> ()
    }
    return
  }
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    // %op_operation_0 = transform.structured.match attributes{tag = "operation_1"} in %arg1 : (!transform.any_op) -> !transform.any_op
    // %op_operation_1 = transform.structured.match attributes{tag = "operation_1"} in %arg1 : (!transform.any_op) -> !transform.any_op
    %op_operation = transform.structured.match attributes{tag = "operation_22"} in %arg1 : (!transform.any_op) -> !transform.any_op
    
    // %tiled_op, %forall_op = transform.structured.tile_using_forall %op_operation  tile_sizes [2, 2, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // %forall_op_operation_2 = transform.get_parent_op %op_operation_2: (!transform.any_op) -> !transform.any_op

    // transform.structured.fuse_into_containing_op %op_operation_1 into %forall_op_consumer : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.structured.vectorize %op_operation: !transform.any_op
    // transform.structured.vectorize_children_and_apply_patterns %op_operation: (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %op_operation = transform.structured.match attributes{tag = "operation_22"} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %op_operation: !transform.any_op
    
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    
    transform.apply_patterns to %f {
        transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
        transform.apply_patterns.vector.transfer_permutation_patterns
        transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerparallel"
        transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "vector-transfer"
        transform.apply_patterns.vector.transfer_to_scf max_transfer_rank = 1 full_unroll = true
        transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1
        transform.apply_patterns.vector.lower_shape_cast
        transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d"
        transform.apply_patterns.canonicalization
    } : !transform.any_op

    transform.yield
  }
}