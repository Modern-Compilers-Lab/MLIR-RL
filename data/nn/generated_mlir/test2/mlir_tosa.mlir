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
    %4 = tosa.reshape %arg0 {new_shape = array<i64: 1, 1, 2>} : (tensor<1x2xf32>) -> tensor<1x1x2xf32>
    %5 = tosa.reshape %3 {new_shape = array<i64: 1, 2, 4>} : (tensor<2x4xf32>) -> tensor<1x2x4xf32>
    %6 = tosa.matmul %4, %5 : (tensor<1x1x2xf32>, tensor<1x2x4xf32>) -> tensor<1x1x4xf32>
    %7 = tosa.reshape %6 {new_shape = array<i64: 1, 4>} : (tensor<1x1x4xf32>) -> tensor<1x4xf32>
    %8 = tosa.reshape %2 {new_shape = array<i64: 1, 4>} : (tensor<4xf32>) -> tensor<1x4xf32>
    %9 = tosa.add %7, %8 : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %10 = tosa.clamp %9 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %11 = tosa.reshape %10 {new_shape = array<i64: 1, 1, 4>} : (tensor<1x4xf32>) -> tensor<1x1x4xf32>
    %12 = tosa.reshape %1 {new_shape = array<i64: 1, 4, 8>} : (tensor<4x8xf32>) -> tensor<1x4x8xf32>
    %13 = tosa.matmul %11, %12 : (tensor<1x1x4xf32>, tensor<1x4x8xf32>) -> tensor<1x1x8xf32>
    %14 = tosa.reshape %13 {new_shape = array<i64: 1, 8>} : (tensor<1x1x8xf32>) -> tensor<1x8xf32>
    %15 = tosa.reshape %0 {new_shape = array<i64: 1, 8>} : (tensor<8xf32>) -> tensor<1x8xf32>
    %16 = tosa.add %14, %15 : (tensor<1x8xf32>, tensor<1x8xf32>) -> tensor<1x8xf32>
    %17 = tosa.reduce_max %16 {axis = 1 : i32} : (tensor<1x8xf32>) -> tensor<1x1xf32>
    %18 = tosa.sub %16, %17 : (tensor<1x8xf32>, tensor<1x1xf32>) -> tensor<1x8xf32>
    %19 = tosa.exp %18 : (tensor<1x8xf32>) -> tensor<1x8xf32>
    %20 = tosa.reduce_sum %19 {axis = 1 : i32} : (tensor<1x8xf32>) -> tensor<1x1xf32>
    %21 = tosa.reciprocal %20 : (tensor<1x1xf32>) -> tensor<1x1xf32>
    %22 = tosa.mul %19, %21 {shift = 0 : i32} : (tensor<1x8xf32>, tensor<1x1xf32>) -> tensor<1x8xf32>
    return %22 : tensor<1x8xf32>
  }
}

