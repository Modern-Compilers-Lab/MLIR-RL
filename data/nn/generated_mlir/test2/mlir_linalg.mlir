#map = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1) -> (0, 0)>
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1482 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node21__model.layer-0.kernel", tf_saved_model.exported_names = [], type = tensor<2x4xf32>, value = dense<[[-0.366711378, 0.366582394, 0.264327288, -0.515119076], [-0.232851505, -0.044178009, -0.418082476, -0.119741678]]> : tensor<2x4xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node22__model.layer-0.bias", tf_saved_model.exported_names = [], type = tensor<4xf32>, value = dense<0.000000e+00> : tensor<4xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node29__model.layer-1.kernel", tf_saved_model.exported_names = [], type = tensor<4x8xf32>, value = dense<[[-0.518020332, -0.0076329708, -0.273415715, 0.446661174, -0.602498293, -0.121825993, 0.0328769088, -0.199197114], [-0.368024915, 0.63398689, -0.601587892, 3.564610e-01, 0.312985241, -0.690208554, -0.603075683, -0.648099839], [0.237009108, 0.482332766, 0.0121293664, -0.034699142, 0.0742671489, -0.331233412, -0.581755817, -0.152040184], [-0.084192872, -0.14643079, -0.460976422, -0.41481635, 0.611420094, -0.458459258, -0.137784958, -0.554757535]]> : tensor<4x8xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node30__model.layer-1.bias", tf_saved_model.exported_names = [], type = tensor<8xf32>, value = dense<0.000000e+00> : tensor<8xf32>} : () -> ()
  func.func @__inference_my_predict_1000(%arg0: tensor<1x2xf32> {tf._user_specified_name = "x", tf_saved_model.index_path = [0]}, %arg1: tensor<!tf_type.resource<tensor<2x4xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node21__model.layer-0.kernel"}, %arg2: tensor<!tf_type.resource<tensor<4xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node22__model.layer-0.bias"}, %arg3: tensor<!tf_type.resource<tensor<4x8xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node29__model.layer-1.kernel"}, %arg4: tensor<!tf_type.resource<tensor<8xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node30__model.layer-1.bias"}) -> (tensor<1x8xf32> {tf_saved_model.index_path = []}) attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x2>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful, tf_saved_model.exported_names = ["my_predict"]} {
    %0 = "tf.ReadVariableOp"(%arg4) : (tensor<!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
    %1 = "tf.ReadVariableOp"(%arg3) : (tensor<!tf_type.resource<tensor<4x8xf32>>>) -> tensor<4x8xf32>
    %2 = "tf.ReadVariableOp"(%arg2) : (tensor<!tf_type.resource<tensor<4xf32>>>) -> tensor<4xf32>
    %3 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf_type.resource<tensor<2x4xf32>>>) -> tensor<2x4xf32>
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2]] : tensor<1x2xf32> into tensor<1x1x2xf32>
    %expanded_0 = tensor.expand_shape %3 [[0, 1], [2]] : tensor<2x4xf32> into tensor<1x2x4xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %4 = tensor.empty() : tensor<1x1x4xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
    %6 = linalg.batch_matmul ins(%expanded, %expanded_0 : tensor<1x1x2xf32>, tensor<1x2x4xf32>) outs(%5 : tensor<1x1x4xf32>) -> tensor<1x1x4xf32>
    %collapsed = tensor.collapse_shape %6 [[0, 1], [2]] : tensor<1x1x4xf32> into tensor<1x4xf32>
    %expanded_1 = tensor.expand_shape %2 [[0, 1]] : tensor<4xf32> into tensor<1x4xf32>
    %7 = tensor.empty() : tensor<1x4xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel"]} ins(%collapsed, %expanded_1 : tensor<1x4xf32>, tensor<1x4xf32>) outs(%7 : tensor<1x4xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %30 = arith.addf %in, %in_11 : f32
      linalg.yield %30 : f32
    } -> tensor<1x4xf32>
    %9 = tensor.empty() : tensor<1x4xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%8 : tensor<1x4xf32>) outs(%9 : tensor<1x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_11 = arith.constant 0.000000e+00 : f32
      %cst_12 = arith.constant 3.40282347E+38 : f32
      %30 = arith.minf %in, %cst_12 : f32
      %31 = arith.maxf %30, %cst_11 : f32
      linalg.yield %31 : f32
    } -> tensor<1x4xf32>
    %expanded_2 = tensor.expand_shape %10 [[0, 1], [2]] : tensor<1x4xf32> into tensor<1x1x4xf32>
    %expanded_3 = tensor.expand_shape %1 [[0, 1], [2]] : tensor<4x8xf32> into tensor<1x4x8xf32>
    %cst_4 = arith.constant 0.000000e+00 : f32
    %11 = tensor.empty() : tensor<1x1x8xf32>
    %12 = linalg.fill ins(%cst_4 : f32) outs(%11 : tensor<1x1x8xf32>) -> tensor<1x1x8xf32>
    %13 = linalg.batch_matmul ins(%expanded_2, %expanded_3 : tensor<1x1x4xf32>, tensor<1x4x8xf32>) outs(%12 : tensor<1x1x8xf32>) -> tensor<1x1x8xf32>
    %collapsed_5 = tensor.collapse_shape %13 [[0, 1], [2]] : tensor<1x1x8xf32> into tensor<1x8xf32>
    %expanded_6 = tensor.expand_shape %0 [[0, 1]] : tensor<8xf32> into tensor<1x8xf32>
    %14 = tensor.empty() : tensor<1x8xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel"]} ins(%collapsed_5, %expanded_6 : tensor<1x8xf32>, tensor<1x8xf32>) outs(%14 : tensor<1x8xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %30 = arith.addf %in, %in_11 : f32
      linalg.yield %30 : f32
    } -> tensor<1x8xf32>
    %16 = tensor.empty() : tensor<1xf32>
    %cst_7 = arith.constant -3.40282347E+38 : f32
    %17 = linalg.fill ins(%cst_7 : f32) outs(%16 : tensor<1xf32>) -> tensor<1xf32>
    %18 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%15 : tensor<1x8xf32>) outs(%17 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = arith.maxf %in, %out : f32
      linalg.yield %30 : f32
    } -> tensor<1xf32>
    %expanded_8 = tensor.expand_shape %18 [[0, 1]] : tensor<1xf32> into tensor<1x1xf32>
    %19 = tensor.empty() : tensor<1x8xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%15, %expanded_8 : tensor<1x8xf32>, tensor<1x1xf32>) outs(%19 : tensor<1x8xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %30 = arith.subf %in, %in_11 : f32
      linalg.yield %30 : f32
    } -> tensor<1x8xf32>
    %21 = tensor.empty() : tensor<1x8xf32>
    %22 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%20 : tensor<1x8xf32>) outs(%21 : tensor<1x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = math.exp %in : f32
      linalg.yield %30 : f32
    } -> tensor<1x8xf32>
    %23 = tensor.empty() : tensor<1xf32>
    %cst_9 = arith.constant 0.000000e+00 : f32
    %24 = linalg.fill ins(%cst_9 : f32) outs(%23 : tensor<1xf32>) -> tensor<1xf32>
    %25 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%22 : tensor<1x8xf32>) outs(%24 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %30 = arith.addf %in, %out : f32
      linalg.yield %30 : f32
    } -> tensor<1xf32>
    %expanded_10 = tensor.expand_shape %25 [[0, 1]] : tensor<1xf32> into tensor<1x1xf32>
    %26 = tensor.empty() : tensor<1x1xf32>
    %27 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_10 : tensor<1x1xf32>) outs(%26 : tensor<1x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_11 = arith.constant 1.000000e+00 : f32
      %30 = arith.divf %cst_11, %in : f32
      linalg.yield %30 : f32
    } -> tensor<1x1xf32>
    %28 = tensor.empty() : tensor<1x8xf32>
    %29 = linalg.generic {indexing_maps = [#map, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%22, %27 : tensor<1x8xf32>, tensor<1x1xf32>) outs(%28 : tensor<1x8xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %30 = arith.mulf %in, %in_11 : f32
      linalg.yield %30 : f32
    } -> tensor<1x8xf32>
    return %29 : tensor<1x8xf32>
  }
}

