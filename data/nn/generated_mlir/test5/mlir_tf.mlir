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
    %4 = "tf.MatMul"(%arg0, %3) {device = "", transpose_a = false, transpose_b = false} : (tensor<1x2xf32>, tensor<2x4xf32>) -> tensor<1x4xf32>
    %5 = "tf.BiasAdd"(%4, %2) {data_format = "NHWC", device = ""} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
    %6 = "tf.Relu"(%5) {device = ""} : (tensor<1x4xf32>) -> tensor<1x4xf32>
    %7 = "tf.MatMul"(%6, %1) {device = "", transpose_a = false, transpose_b = false} : (tensor<1x4xf32>, tensor<4x8xf32>) -> tensor<1x8xf32>
    %8 = "tf.BiasAdd"(%7, %0) {data_format = "NHWC", device = ""} : (tensor<1x8xf32>, tensor<8xf32>) -> tensor<1x8xf32>
    %9 = "tf.Softmax"(%8) {device = ""} : (tensor<1x8xf32>) -> tensor<1x8xf32>
    %10 = "tf.Identity"(%9) {device = ""} : (tensor<1x8xf32>) -> tensor<1x8xf32>
    return %10 : tensor<1x8xf32>
  }
  func.func private @__inference__wrapped_model_1500(%arg0: tensor<?x2xf32> {tf._user_specified_name = "dense_input"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}, %arg3: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}, %arg4: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}) -> tensor<?x8xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x2>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = "tf.ReadVariableOp"(%arg4) {device = ""} : (tensor<!tf_type.resource>) -> tensor<8xf32>
    %1 = "tf.ReadVariableOp"(%arg3) {device = ""} : (tensor<!tf_type.resource>) -> tensor<4x8xf32>
    %2 = "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<!tf_type.resource>) -> tensor<4xf32>
    %3 = "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<!tf_type.resource>) -> tensor<2x4xf32>
    %4 = "tf.MatMul"(%arg0, %3) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x2xf32>, tensor<2x4xf32>) -> tensor<?x4xf32>
    %5 = "tf.BiasAdd"(%4, %2) {data_format = "NHWC", device = ""} : (tensor<?x4xf32>, tensor<4xf32>) -> tensor<?x4xf32>
    %6 = "tf.Relu"(%5) {device = ""} : (tensor<?x4xf32>) -> tensor<?x4xf32>
    %7 = "tf.MatMul"(%6, %1) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x4xf32>, tensor<4x8xf32>) -> tensor<?x8xf32>
    %8 = "tf.BiasAdd"(%7, %0) {data_format = "NHWC", device = ""} : (tensor<?x8xf32>, tensor<8xf32>) -> tensor<?x8xf32>
    %9 = "tf.Softmax"(%8) {device = ""} : (tensor<?x8xf32>) -> tensor<?x8xf32>
    %10 = "tf.Identity"(%9) {device = ""} : (tensor<?x8xf32>) -> tensor<?x8xf32>
    return %10 : tensor<?x8xf32>
  }
  func.func private @__inference_dense_1_layer_call_and_return_conditional_losses_1790(%arg0: tensor<?x4xf32> {tf._user_specified_name = "inputs"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}) -> tensor<?x8xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x4>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<!tf_type.resource>) -> tensor<8xf32>
    %1 = "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<!tf_type.resource>) -> tensor<4x8xf32>
    %2 = "tf.MatMul"(%arg0, %1) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x4xf32>, tensor<4x8xf32>) -> tensor<?x8xf32>
    %3 = "tf.BiasAdd"(%2, %0) {data_format = "NHWC", device = ""} : (tensor<?x8xf32>, tensor<8xf32>) -> tensor<?x8xf32>
    %4 = "tf.Softmax"(%3) {device = ""} : (tensor<?x8xf32>) -> tensor<?x8xf32>
    %5 = "tf.Identity"(%4) {device = ""} : (tensor<?x8xf32>) -> tensor<?x8xf32>
    return %5 : tensor<?x8xf32>
  }
  func.func private @__inference_dense_1_layer_call_and_return_conditional_losses_2840(%arg0: tensor<?x4xf32> {tf._user_specified_name = "inputs"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}) -> tensor<?x8xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x4>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<!tf_type.resource>) -> tensor<8xf32>
    %1 = "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<!tf_type.resource>) -> tensor<4x8xf32>
    %2 = "tf.MatMul"(%arg0, %1) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x4xf32>, tensor<4x8xf32>) -> tensor<?x8xf32>
    %3 = "tf.BiasAdd"(%2, %0) {data_format = "NHWC", device = ""} : (tensor<?x8xf32>, tensor<8xf32>) -> tensor<?x8xf32>
    %4 = "tf.Softmax"(%3) {device = ""} : (tensor<?x8xf32>) -> tensor<?x8xf32>
    %5 = "tf.Identity"(%4) {device = ""} : (tensor<?x8xf32>) -> tensor<?x8xf32>
    return %5 : tensor<?x8xf32>
  }
  func.func private @__inference_dense_1_layer_call_fn_2730(%arg0: tensor<?x4xf32> {tf._user_specified_name = "inputs"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "267"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "269"}) -> tensor<?x?xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x4>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = "tf.StatefulPartitionedCall"(%arg0, %arg1, %arg2) {_collective_manager_ids = [], _read_only_resource_inputs = [1, 2], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_dense_1_layer_call_and_return_conditional_losses_1790} : (tensor<?x4xf32>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> tensor<?x?xf32>
    %1 = "tf.Identity"(%0) {device = ""} : (tensor<?x?xf32>) -> tensor<?x?xf32>
    return %1 : tensor<?x?xf32>
  }
  func.func private @__inference_dense_layer_call_and_return_conditional_losses_1630(%arg0: tensor<?x2xf32> {tf._user_specified_name = "inputs"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}) -> tensor<?x4xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x2>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<!tf_type.resource>) -> tensor<4xf32>
    %1 = "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<!tf_type.resource>) -> tensor<2x4xf32>
    %2 = "tf.MatMul"(%arg0, %1) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x2xf32>, tensor<2x4xf32>) -> tensor<?x4xf32>
    %3 = "tf.BiasAdd"(%2, %0) {data_format = "NHWC", device = ""} : (tensor<?x4xf32>, tensor<4xf32>) -> tensor<?x4xf32>
    %4 = "tf.Relu"(%3) {device = ""} : (tensor<?x4xf32>) -> tensor<?x4xf32>
    %5 = "tf.Identity"(%4) {device = ""} : (tensor<?x4xf32>) -> tensor<?x4xf32>
    return %5 : tensor<?x4xf32>
  }
  func.func private @__inference_dense_layer_call_and_return_conditional_losses_2640(%arg0: tensor<?x2xf32> {tf._user_specified_name = "inputs"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "resource"}) -> tensor<?x4xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x2>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = "tf.ReadVariableOp"(%arg2) {device = ""} : (tensor<!tf_type.resource>) -> tensor<4xf32>
    %1 = "tf.ReadVariableOp"(%arg1) {device = ""} : (tensor<!tf_type.resource>) -> tensor<2x4xf32>
    %2 = "tf.MatMul"(%arg0, %1) {device = "", transpose_a = false, transpose_b = false} : (tensor<?x2xf32>, tensor<2x4xf32>) -> tensor<?x4xf32>
    %3 = "tf.BiasAdd"(%2, %0) {data_format = "NHWC", device = ""} : (tensor<?x4xf32>, tensor<4xf32>) -> tensor<?x4xf32>
    %4 = "tf.Relu"(%3) {device = ""} : (tensor<?x4xf32>) -> tensor<?x4xf32>
    %5 = "tf.Identity"(%4) {device = ""} : (tensor<?x4xf32>) -> tensor<?x4xf32>
    return %5 : tensor<?x4xf32>
  }
  func.func private @__inference_dense_layer_call_fn_2530(%arg0: tensor<?x2xf32> {tf._user_specified_name = "inputs"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "247"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "249"}) -> tensor<?x?xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x2>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = "tf.StatefulPartitionedCall"(%arg0, %arg1, %arg2) {_collective_manager_ids = [], _read_only_resource_inputs = [1, 2], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_dense_layer_call_and_return_conditional_losses_1630} : (tensor<?x2xf32>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> tensor<?x?xf32>
    %1 = "tf.Identity"(%0) {device = ""} : (tensor<?x?xf32>) -> tensor<?x?xf32>
    return %1 : tensor<?x?xf32>
  }
  func.func private @__inference_sequential_layer_call_and_return_conditional_losses_1860(%arg0: tensor<?x2xf32> {tf._user_specified_name = "dense_input"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "164"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "166"}, %arg3: tensor<!tf_type.resource> {tf._user_specified_name = "180"}, %arg4: tensor<!tf_type.resource> {tf._user_specified_name = "182"}) -> tensor<?x?xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x2>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = "tf.StatefulPartitionedCall"(%arg0, %arg1, %arg2) {_collective_manager_ids = [], _read_only_resource_inputs = [1, 2], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_dense_layer_call_and_return_conditional_losses_1630} : (tensor<?x2xf32>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> tensor<?x?xf32>
    %1 = "tf.StatefulPartitionedCall"(%0, %arg3, %arg4) {_collective_manager_ids = [], _read_only_resource_inputs = [1, 2], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_dense_1_layer_call_and_return_conditional_losses_1790} : (tensor<?x?xf32>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> tensor<?x?xf32>
    %2 = "tf.Identity"(%1) {device = ""} : (tensor<?x?xf32>) -> tensor<?x?xf32>
    return %2 : tensor<?x?xf32>
  }
  func.func private @__inference_sequential_layer_call_and_return_conditional_losses_2000(%arg0: tensor<?x2xf32> {tf._user_specified_name = "dense_input"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "189"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "191"}, %arg3: tensor<!tf_type.resource> {tf._user_specified_name = "194"}, %arg4: tensor<!tf_type.resource> {tf._user_specified_name = "196"}) -> tensor<?x?xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x2>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = "tf.StatefulPartitionedCall"(%arg0, %arg1, %arg2) {_collective_manager_ids = [], _read_only_resource_inputs = [1, 2], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_dense_layer_call_and_return_conditional_losses_1630} : (tensor<?x2xf32>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> tensor<?x?xf32>
    %1 = "tf.StatefulPartitionedCall"(%0, %arg3, %arg4) {_collective_manager_ids = [], _read_only_resource_inputs = [1, 2], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_dense_1_layer_call_and_return_conditional_losses_1790} : (tensor<?x?xf32>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> tensor<?x?xf32>
    %2 = "tf.Identity"(%1) {device = ""} : (tensor<?x?xf32>) -> tensor<?x?xf32>
    return %2 : tensor<?x?xf32>
  }
  func.func private @__inference_sequential_layer_call_fn_2130(%arg0: tensor<?x2xf32> {tf._user_specified_name = "dense_input"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "203"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "205"}, %arg3: tensor<!tf_type.resource> {tf._user_specified_name = "207"}, %arg4: tensor<!tf_type.resource> {tf._user_specified_name = "209"}) -> tensor<?x?xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x2>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = "tf.StatefulPartitionedCall"(%arg0, %arg1, %arg2, %arg3, %arg4) {_collective_manager_ids = [], _read_only_resource_inputs = [1, 2, 3, 4], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_sequential_layer_call_and_return_conditional_losses_1860} : (tensor<?x2xf32>, tensor<!tf_type.resource>, tensor<!tf_type.resource>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> tensor<?x?xf32>
    %1 = "tf.Identity"(%0) {device = ""} : (tensor<?x?xf32>) -> tensor<?x?xf32>
    return %1 : tensor<?x?xf32>
  }
  func.func private @__inference_sequential_layer_call_fn_2260(%arg0: tensor<?x2xf32> {tf._user_specified_name = "dense_input"}, %arg1: tensor<!tf_type.resource> {tf._user_specified_name = "216"}, %arg2: tensor<!tf_type.resource> {tf._user_specified_name = "218"}, %arg3: tensor<!tf_type.resource> {tf._user_specified_name = "220"}, %arg4: tensor<!tf_type.resource> {tf._user_specified_name = "222"}) -> tensor<?x?xf32> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<?x2>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
    %0 = "tf.StatefulPartitionedCall"(%arg0, %arg1, %arg2, %arg3, %arg4) {_collective_manager_ids = [], _read_only_resource_inputs = [1, 2, 3, 4], config = "", config_proto = "\0A\07\0A\03CPU\10\01\0A\07\0A\03GPU\10\002\02J\008\01\82\01\00", device = "", executor_type = "", f = @__inference_sequential_layer_call_and_return_conditional_losses_2000} : (tensor<?x2xf32>, tensor<!tf_type.resource>, tensor<!tf_type.resource>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> tensor<?x?xf32>
    %1 = "tf.Identity"(%0) {device = ""} : (tensor<?x?xf32>) -> tensor<?x?xf32>
    return %1 : tensor<?x?xf32>
  }
}

