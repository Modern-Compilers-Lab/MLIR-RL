#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, 0)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map9 = affine_map<(d0, d1, d2) -> (0, 0, d2)>
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1482 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node27__model.layer-1._query_dense.kernel", tf_saved_model.exported_names = [], type = tensor<5x3x5xf32>, value = dense<[[[0.310287893, 0.274820089, 5.43415546E-4, -0.00919929146, -0.250944972], [0.156003237, -0.0846630632, -0.30663994, 0.0787280499, 0.334349692], [-0.241997376, 0.146094322, 0.0849350988, -0.238803551, -0.142152473]], [[-0.0897827148, 0.226250231, 0.191081762, -0.119488209, 0.146939337], [-0.350444227, -0.138169527, 0.315861642, 0.366289437, -0.377245665], [-0.335510701, -0.353315055, 0.371301174, -0.0364615917, -0.340348244]], [[0.0691623687, -0.196790367, -0.372358888, 0.274694026, 0.138106525], [-0.328699291, -0.315824717, 0.315979183, 3.590340e-01, 0.0414014459], [-0.10489291, 0.368188143, 0.201158464, -0.177676991, -0.14520669]], [[0.31262064, -0.268262327, -0.384290963, 0.0656236708, 0.330219448], [0.346164525, 0.133215129, 0.232835472, -0.36107561, 0.181677222], [0.078620404, -0.217953101, -0.0449540913, -0.137283057, -3.404810e-01]], [[-0.11414519, -0.135874897, 0.137863398, -0.0356529653, -0.0513389707], [-0.309102058, -0.333007097, 0.376749873, 0.108444184, -0.14111191], [-0.369793802, -0.261295259, -0.086956948, -0.170106575, -0.327220857]]]> : tensor<5x3x5xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node28__model.layer-1._query_dense.bias", tf_saved_model.exported_names = [], type = tensor<3x5xf32>, value = dense<0.000000e+00> : tensor<3x5xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node29__model.layer-1._key_dense.kernel", tf_saved_model.exported_names = [], type = tensor<5x3x5xf32>, value = dense<[[[0.324363351, -0.0902127325, 0.185980201, 0.261387229, 0.185061514], [0.0138212144, -0.10823974, 0.202930391, -0.357366979, -0.0553325415], [-0.109206527, -0.146983758, 0.274110258, -0.0407250524, -0.333563268]], [[-0.35220468, 0.282635629, -0.120861202, -0.0854712128, 0.0452415645], [0.240018547, 0.257152915, -0.127075076, -0.306939572, 0.325205386], [0.22814095, 0.160902977, 0.103462577, -0.0429808199, 0.172422171]], [[-0.0144159794, 0.0386853814, 0.365421057, 0.369458139, -0.225760475], [-0.198099181, 0.316948116, -0.105479807, -0.0755209327, -0.0511599481], [-0.0837241709, -3.822630e-01, 0.384873033, -2.088710e-01, 0.327047706]], [[-0.0677655637, -0.141753018, 0.366386831, -0.291330934, -0.200122058], [0.28406471, 0.227398753, -0.223438427, -0.232365772, -0.221889704], [0.102197438, 0.297752023, 0.203199267, 0.289845288, -0.357672155]], [[-0.369906545, -0.135603338, -0.358202577, 0.293190658, 0.0197403431], [0.232274055, 0.0499054193, 0.049036324, 0.136489868, -0.314502984], [-0.380503118, 0.177803576, -0.194654465, -0.247575685, -0.383314282]]]> : tensor<5x3x5xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node30__model.layer-1._key_dense.bias", tf_saved_model.exported_names = [], type = tensor<3x5xf32>, value = dense<0.000000e+00> : tensor<3x5xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node31__model.layer-1._value_dense.kernel", tf_saved_model.exported_names = [], type = tensor<5x3x5xf32>, value = dense<[[[-0.210780203, 0.310085654, 0.169174492, 0.308731675, 0.36276412], [0.0504656434, 0.285769939, -0.0525137186, -0.201444805, 0.311411858], [0.116225958, -0.312494874, 0.160975456, 0.136797667, -0.382725984]], [[0.278635502, -0.309523702, 0.280287206, 0.20568496, 0.338483632], [0.208339691, -0.0243619084, -0.0767028629, 0.0802135169, -0.292571425], [0.0737955868, -0.0992563366, -0.175886258, 0.319514513, 0.123166323]], [[0.135054529, -0.181781188, 0.330280244, -0.0419903696, 0.121675789], [-0.371611208, 0.0081307292, -0.164197519, 0.0066716969, 0.330342352], [0.28943485, -5.187860e-02, 0.2667557, 0.293436289, -0.355670363]], [[0.193354785, 0.103153795, -0.16443114, 0.0395831168, 0.0274646878], [-0.332411408, -0.318964809, -0.330746919, -0.37583971, 0.286465228], [-0.185938671, 0.0282998979, 0.334109783, 0.23926127, -0.283893406]], [[0.334831715, 0.0337272286, -0.218974099, -0.387129813, 0.335513473], [-0.269506872, 0.144676268, 0.162777364, 0.0984055101, -0.11986503], [-0.293383449, 0.210223138, -0.318047225, -0.0892969965, -0.321155101]]]> : tensor<5x3x5xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node32__model.layer-1._value_dense.bias", tf_saved_model.exported_names = [], type = tensor<3x5xf32>, value = dense<0.000000e+00> : tensor<3x5xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node33__model.layer-1._output_dense.kernel", tf_saved_model.exported_names = [], type = tensor<3x5x5xf32>, value = dense<[[[0.316501796, -0.0722778737, 0.326826096, 0.394996822, -0.328547984], [0.0607203841, -0.0711639821, -0.444023281, -0.0575793087, 0.299967825], [0.406200469, -0.19685173, -5.098760e-02, -0.0178295672, -0.341129839], [3.831430e-01, 0.343245864, 0.156046629, -0.241003469, 0.0228432417], [-0.412869453, -0.365723073, -0.36510691, -0.374539286, 0.269160628]], [[0.238722801, -0.205095366, -0.426838368, 0.37775743, -0.0237810016], [-0.446660638, -0.33216542, -0.350717038, -0.0678566098, -0.249638095], [-0.306576073, 0.445805728, 0.0600081086, -0.401822031, 0.0209073722], [-0.401941776, 0.351715147, -0.349473357, 0.209907532, -0.305728495], [0.343320429, -0.353443414, -0.396292925, 0.0903673768, 0.194914281]], [[0.43753159, -0.195364654, -0.338448644, -0.447035968, 0.188521087], [-0.112257302, -0.42572394, -0.0125853717, -0.276022315, 0.384799302], [-0.130522847, -0.0595476031, 0.199029505, 0.372424304, 0.279297888], [0.277517378, 0.183211088, -0.0528746545, -0.149683177, 0.127622962], [0.331316054, 0.19403249, 0.436611414, -0.36348781, 0.0310319662]]]> : tensor<3x5x5xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node34__model.layer-1._output_dense.bias", tf_saved_model.exported_names = [], type = tensor<5xf32>, value = dense<0.000000e+00> : tensor<5xf32>} : () -> ()
  func.func @__inference_my_predict_1490(%arg0: tensor<16x7x5xf32> {tf._user_specified_name = "x", tf_saved_model.index_path = [0]}, %arg1: tensor<!tf_type.resource<tensor<5x3x5xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node27__model.layer-1._query_dense.kernel"}, %arg2: tensor<!tf_type.resource<tensor<3x5xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node28__model.layer-1._query_dense.bias"}, %arg3: tensor<!tf_type.resource<tensor<5x3x5xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node29__model.layer-1._key_dense.kernel"}, %arg4: tensor<!tf_type.resource<tensor<3x5xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node30__model.layer-1._key_dense.bias"}, %arg5: tensor<!tf_type.resource<tensor<5x3x5xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node31__model.layer-1._value_dense.kernel"}, %arg6: tensor<!tf_type.resource<tensor<3x5xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node32__model.layer-1._value_dense.bias"}, %arg7: tensor<!tf_type.resource<tensor<3x5x5xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node33__model.layer-1._output_dense.kernel"}, %arg8: tensor<!tf_type.resource<tensor<5xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node34__model.layer-1._output_dense.bias"}) -> (tensor<16x7x5xf32> {tf_saved_model.index_path = []}) attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<16x7x5>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful, tf_saved_model.exported_names = ["my_predict"]} {
    %cst = arith.constant dense<0.44721359> : tensor<1x1x1x1xf32>
    %cst_0 = arith.constant dense<[0, 2, 1, 3]> : tensor<4xi32>
    %cst_1 = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
    %cst_2 = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi32>
    %0 = "tf.ReadVariableOp"(%arg8) : (tensor<!tf_type.resource<tensor<5xf32>>>) -> tensor<5xf32>
    %1 = "tf.ReadVariableOp"(%arg7) : (tensor<!tf_type.resource<tensor<3x5x5xf32>>>) -> tensor<3x5x5xf32>
    %2 = "tf.ReadVariableOp"(%arg4) : (tensor<!tf_type.resource<tensor<3x5xf32>>>) -> tensor<3x5xf32>
    %3 = "tf.ReadVariableOp"(%arg3) : (tensor<!tf_type.resource<tensor<5x3x5xf32>>>) -> tensor<5x3x5xf32>
    %4 = "tf.ReadVariableOp"(%arg2) : (tensor<!tf_type.resource<tensor<3x5xf32>>>) -> tensor<3x5xf32>
    %5 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf_type.resource<tensor<5x3x5xf32>>>) -> tensor<5x3x5xf32>
    %6 = "tf.ReadVariableOp"(%arg6) : (tensor<!tf_type.resource<tensor<3x5xf32>>>) -> tensor<3x5xf32>
    %7 = "tf.ReadVariableOp"(%arg5) : (tensor<!tf_type.resource<tensor<5x3x5xf32>>>) -> tensor<5x3x5xf32>
    %collapsed = tensor.collapse_shape %3 [[0], [1, 2]] : tensor<5x3x5xf32> into tensor<5x15xf32>
    %8 = "tf.BatchMatMulV2"(%arg0, %collapsed) {adj_x = false, adj_y = false} : (tensor<16x7x5xf32>, tensor<5x15xf32>) -> tensor<16x7x15xf32>
    %expanded = tensor.expand_shape %8 [[0], [1], [2, 3]] : tensor<16x7x15xf32> into tensor<16x7x3x5xf32>
    %expanded_3 = tensor.expand_shape %2 [[0, 1, 2], [3]] : tensor<3x5xf32> into tensor<1x1x3x5xf32>
    %9 = tensor.empty() : tensor<16x7x3x5xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded, %expanded_3 : tensor<16x7x3x5xf32>, tensor<1x1x3x5xf32>) outs(%9 : tensor<16x7x3x5xf32>) {
    ^bb0(%in: f32, %in_25: f32, %out: f32):
      %52 = arith.addf %in, %in_25 : f32
      linalg.yield %52 : f32
    } -> tensor<16x7x3x5xf32>
    %collapsed_4 = tensor.collapse_shape %5 [[0], [1, 2]] : tensor<5x3x5xf32> into tensor<5x15xf32>
    %11 = "tf.BatchMatMulV2"(%arg0, %collapsed_4) {adj_x = false, adj_y = false} : (tensor<16x7x5xf32>, tensor<5x15xf32>) -> tensor<16x7x15xf32>
    %expanded_5 = tensor.expand_shape %11 [[0], [1], [2, 3]] : tensor<16x7x15xf32> into tensor<16x7x3x5xf32>
    %expanded_6 = tensor.expand_shape %4 [[0, 1, 2], [3]] : tensor<3x5xf32> into tensor<1x1x3x5xf32>
    %12 = tensor.empty() : tensor<16x7x3x5xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_5, %expanded_6 : tensor<16x7x3x5xf32>, tensor<1x1x3x5xf32>) outs(%12 : tensor<16x7x3x5xf32>) {
    ^bb0(%in: f32, %in_25: f32, %out: f32):
      %52 = arith.addf %in, %in_25 : f32
      linalg.yield %52 : f32
    } -> tensor<16x7x3x5xf32>
    %14 = tensor.empty() : tensor<16x7x3x5xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13, %cst : tensor<16x7x3x5xf32>, tensor<1x1x1x1xf32>) outs(%14 : tensor<16x7x3x5xf32>) {
    ^bb0(%in: f32, %in_25: f32, %out: f32):
      %52 = arith.mulf %in, %in_25 : f32
      linalg.yield %52 : f32
    } -> tensor<16x7x3x5xf32>
    %16 = tensor.empty() : tensor<16x3x7x5xf32>
    %17 = linalg.generic {indexing_maps = [#map3, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10 : tensor<16x7x3x5xf32>) outs(%16 : tensor<16x3x7x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x3x7x5xf32>
    %18 = tensor.empty() : tensor<16x3x5x7xf32>
    %19 = linalg.generic {indexing_maps = [#map4, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%15 : tensor<16x7x3x5xf32>) outs(%18 : tensor<16x3x5x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x3x5x7xf32>
    %collapsed_7 = tensor.collapse_shape %17 [[0, 1], [2], [3]] : tensor<16x3x7x5xf32> into tensor<48x7x5xf32>
    %collapsed_8 = tensor.collapse_shape %19 [[0, 1], [2], [3]] : tensor<16x3x5x7xf32> into tensor<48x5x7xf32>
    %cst_9 = arith.constant 0.000000e+00 : f32
    %20 = tensor.empty() : tensor<48x7x7xf32>
    %21 = linalg.fill ins(%cst_9 : f32) outs(%20 : tensor<48x7x7xf32>) -> tensor<48x7x7xf32>
    %22 = linalg.batch_matmul ins(%collapsed_7, %collapsed_8 : tensor<48x7x5xf32>, tensor<48x5x7xf32>) outs(%21 : tensor<48x7x7xf32>) -> tensor<48x7x7xf32>
    %expanded_10 = tensor.expand_shape %22 [[0, 1], [2], [3]] : tensor<48x7x7xf32> into tensor<16x3x7x7xf32>
    %23 = tensor.empty() : tensor<16x3x7x7xf32>
    %24 = linalg.generic {indexing_maps = [#map5, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_10 : tensor<16x3x7x7xf32>) outs(%23 : tensor<16x3x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x3x7x7xf32>
    %25 = tensor.empty() : tensor<16x3x7xf32>
    %cst_11 = arith.constant -3.40282347E+38 : f32
    %26 = linalg.fill ins(%cst_11 : f32) outs(%25 : tensor<16x3x7xf32>) -> tensor<16x3x7xf32>
    %27 = linalg.generic {indexing_maps = [#map, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%24 : tensor<16x3x7x7xf32>) outs(%26 : tensor<16x3x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %52 = arith.maxf %in, %out : f32
      linalg.yield %52 : f32
    } -> tensor<16x3x7xf32>
    %expanded_12 = tensor.expand_shape %27 [[0], [1], [2, 3]] : tensor<16x3x7xf32> into tensor<16x3x7x1xf32>
    %28 = tensor.empty() : tensor<16x3x7x7xf32>
    %29 = linalg.generic {indexing_maps = [#map, #map7, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24, %expanded_12 : tensor<16x3x7x7xf32>, tensor<16x3x7x1xf32>) outs(%28 : tensor<16x3x7x7xf32>) {
    ^bb0(%in: f32, %in_25: f32, %out: f32):
      %52 = arith.subf %in, %in_25 : f32
      linalg.yield %52 : f32
    } -> tensor<16x3x7x7xf32>
    %30 = tensor.empty() : tensor<16x3x7x7xf32>
    %31 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%29 : tensor<16x3x7x7xf32>) outs(%30 : tensor<16x3x7x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %52 = math.exp %in : f32
      linalg.yield %52 : f32
    } -> tensor<16x3x7x7xf32>
    %32 = tensor.empty() : tensor<16x3x7xf32>
    %cst_13 = arith.constant 0.000000e+00 : f32
    %33 = linalg.fill ins(%cst_13 : f32) outs(%32 : tensor<16x3x7xf32>) -> tensor<16x3x7xf32>
    %34 = linalg.generic {indexing_maps = [#map, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%31 : tensor<16x3x7x7xf32>) outs(%33 : tensor<16x3x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %52 = arith.addf %in, %out : f32
      linalg.yield %52 : f32
    } -> tensor<16x3x7xf32>
    %expanded_14 = tensor.expand_shape %34 [[0], [1], [2, 3]] : tensor<16x3x7xf32> into tensor<16x3x7x1xf32>
    %35 = tensor.empty() : tensor<16x3x7x1xf32>
    %36 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_14 : tensor<16x3x7x1xf32>) outs(%35 : tensor<16x3x7x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_25 = arith.constant 1.000000e+00 : f32
      %52 = arith.divf %cst_25, %in : f32
      linalg.yield %52 : f32
    } -> tensor<16x3x7x1xf32>
    %37 = tensor.empty() : tensor<16x3x7x7xf32>
    %38 = linalg.generic {indexing_maps = [#map, #map7, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31, %36 : tensor<16x3x7x7xf32>, tensor<16x3x7x1xf32>) outs(%37 : tensor<16x3x7x7xf32>) {
    ^bb0(%in: f32, %in_25: f32, %out: f32):
      %52 = arith.mulf %in, %in_25 : f32
      linalg.yield %52 : f32
    } -> tensor<16x3x7x7xf32>
    %collapsed_15 = tensor.collapse_shape %7 [[0], [1, 2]] : tensor<5x3x5xf32> into tensor<5x15xf32>
    %39 = "tf.BatchMatMulV2"(%arg0, %collapsed_15) {adj_x = false, adj_y = false} : (tensor<16x7x5xf32>, tensor<5x15xf32>) -> tensor<16x7x15xf32>
    %expanded_16 = tensor.expand_shape %39 [[0], [1], [2, 3]] : tensor<16x7x15xf32> into tensor<16x7x3x5xf32>
    %expanded_17 = tensor.expand_shape %6 [[0, 1, 2], [3]] : tensor<3x5xf32> into tensor<1x1x3x5xf32>
    %40 = tensor.empty() : tensor<16x7x3x5xf32>
    %41 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_16, %expanded_17 : tensor<16x7x3x5xf32>, tensor<1x1x3x5xf32>) outs(%40 : tensor<16x7x3x5xf32>) {
    ^bb0(%in: f32, %in_25: f32, %out: f32):
      %52 = arith.addf %in, %in_25 : f32
      linalg.yield %52 : f32
    } -> tensor<16x7x3x5xf32>
    %42 = tensor.empty() : tensor<16x3x7x5xf32>
    %43 = linalg.generic {indexing_maps = [#map3, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%41 : tensor<16x7x3x5xf32>) outs(%42 : tensor<16x3x7x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x3x7x5xf32>
    %collapsed_18 = tensor.collapse_shape %38 [[0, 1], [2], [3]] : tensor<16x3x7x7xf32> into tensor<48x7x7xf32>
    %collapsed_19 = tensor.collapse_shape %43 [[0, 1], [2], [3]] : tensor<16x3x7x5xf32> into tensor<48x7x5xf32>
    %cst_20 = arith.constant 0.000000e+00 : f32
    %44 = tensor.empty() : tensor<48x7x5xf32>
    %45 = linalg.fill ins(%cst_20 : f32) outs(%44 : tensor<48x7x5xf32>) -> tensor<48x7x5xf32>
    %46 = linalg.batch_matmul ins(%collapsed_18, %collapsed_19 : tensor<48x7x7xf32>, tensor<48x7x5xf32>) outs(%45 : tensor<48x7x5xf32>) -> tensor<48x7x5xf32>
    %expanded_21 = tensor.expand_shape %46 [[0, 1], [2], [3]] : tensor<48x7x5xf32> into tensor<16x3x7x5xf32>
    %47 = tensor.empty() : tensor<16x7x3x5xf32>
    %48 = linalg.generic {indexing_maps = [#map3, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_21 : tensor<16x3x7x5xf32>) outs(%47 : tensor<16x7x3x5xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<16x7x3x5xf32>
    %collapsed_22 = tensor.collapse_shape %48 [[0], [1], [2, 3]] : tensor<16x7x3x5xf32> into tensor<16x7x15xf32>
    %collapsed_23 = tensor.collapse_shape %1 [[0, 1], [2]] : tensor<3x5x5xf32> into tensor<15x5xf32>
    %49 = "tf.BatchMatMulV2"(%collapsed_22, %collapsed_23) {adj_x = false, adj_y = false} : (tensor<16x7x15xf32>, tensor<15x5xf32>) -> tensor<16x7x5xf32>
    %expanded_24 = tensor.expand_shape %0 [[0, 1, 2]] : tensor<5xf32> into tensor<1x1x5xf32>
    %50 = tensor.empty() : tensor<16x7x5xf32>
    %51 = linalg.generic {indexing_maps = [#map8, #map9, #map8], iterator_types = ["parallel", "parallel", "parallel"]} ins(%49, %expanded_24 : tensor<16x7x5xf32>, tensor<1x1x5xf32>) outs(%50 : tensor<16x7x5xf32>) {
    ^bb0(%in: f32, %in_25: f32, %out: f32):
      %52 = arith.addf %in, %in_25 : f32
      linalg.yield %52 : f32
    } -> tensor<16x7x5xf32>
    return %51 : tensor<16x7x5xf32>
  }
}

