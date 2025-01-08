module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1482 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node21__model.layer-0.kernel", tf_saved_model.exported_names = [], type = tensor<2x4xf32>, value = dense<[[-0.0659065247, -0.71034503, -0.143171549, -0.260344028], [-0.69184041, -0.760239601, -0.711513996, -0.485834837]]> : tensor<2x4xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node22__model.layer-0.bias", tf_saved_model.exported_names = [], type = tensor<4xf32>, value = dense<0.000000e+00> : tensor<4xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node29__model.layer-1.kernel", tf_saved_model.exported_names = [], type = tensor<4x8xf32>, value = dense<[[-0.297534823, -0.0913531184, -0.409505337, 0.0201846361, 0.585041583, -0.560052037, 0.330370724, -0.0822428465], [-0.173333287, 0.462669194, 0.236135781, 0.55484134, 0.569796503, -0.187162876, 0.580581367, 0.25352478], [0.0387528539, -0.359654576, 0.0830879807, 0.244462192, 0.344339907, 0.616770446, -0.622811734, -0.163719416], [-0.704532623, -0.111978114, -0.384449035, 0.620973765, 0.159519911, -0.382416546, 0.422677457, 5.600330e-01]]> : tensor<4x8xf32>} : () -> ()
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

