#map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (0, d1)>
#map6 = affine_map<(d0, d1) -> (d0, 0)>
#map7 = affine_map<(d0, d1) -> (0, 0)>
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1482 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node26__model.layer-0.kernel", tf_saved_model.exported_names = [], type = tensor<3x3x3x2xf32>, value = dense<[[[[0.176463425, -2.369690e-01], [-0.304605335, -0.0860649049], [-0.165641204, -0.266251624]], [[-0.0499321222, 0.207697511], [0.0934430062, -0.246268898], [-0.361939043, 0.138852894]], [[0.257687092, -0.101477236], [-0.329039872, 0.155028462], [-0.193229228, 0.0859983861]]], [[[-0.325796425, 0.195808649], [0.325243711, -0.115343526], [-0.325931102, -0.357967108]], [[0.284842849, -0.131421775], [-0.265716016, 0.261599243], [0.0133949518, 0.0997334718]], [[0.234076023, -0.225259662], [-0.0445687175, -0.00787910819], [0.183279335, -0.0708890259]]], [[[-0.0425833762, 0.314798653], [-3.829810e-02, -0.136955455], [-0.0789331793, 0.217654169]], [[0.241518974, 0.108961552], [-0.123185113, -0.145454064], [-0.16551584, 0.128013432]], [[-0.0468637645, -0.100344539], [-0.16066356, 0.112919897], [0.244914293, 0.0809878408]]]]> : tensor<3x3x3x2xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node27__model.layer-0.bias", tf_saved_model.exported_names = [], type = tensor<2xf32>, value = dense<0.000000e+00> : tensor<2xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node41__model.layer-2.kernel", tf_saved_model.exported_names = [], type = tensor<3x3x2x3xf32>, value = dense<[[[[0.100532502, -0.168829441, 0.102570087], [0.212870359, 0.013250947, -0.104474396]], [[-0.262461871, -0.326837718, -0.324725688], [0.279223859, 0.205869734, -0.132944763]], [[0.240372539, -0.343261868, 0.0995227992], [0.317929149, -0.113965303, -0.102093875]]], [[[-0.155890778, -0.134274229, 0.128072381], [0.151166141, -0.111402392, -0.185141876]], [[0.00898438692, 0.238462627, 0.278365254], [-0.299663216, -0.231795728, 0.00850391387]], [[0.0516980886, -0.145300925, 0.334896088], [0.260956109, -0.0887488127, 0.231064856]]], [[[0.343109131, 0.225938737, -0.346620202], [0.311179638, 0.148108304, 0.137205243]], [[-0.357029587, -0.264785469, -0.0252652466], [-0.29172805, 0.0683663189, -0.358008206]], [[-0.0615435839, -0.130657926, 0.176007032], [0.317584574, 0.213780344, -0.204653785]]]]> : tensor<3x3x2x3xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node42__model.layer-2.bias", tf_saved_model.exported_names = [], type = tensor<3xf32>, value = dense<0.000000e+00> : tensor<3xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node62__model.layer-5.kernel", tf_saved_model.exported_names = [], type = tensor<75x4xf32>, value = dense<"0x22B845BED91C823E9CD186BD0C4E983D788FDFBC860725BE02F672BE4492F7BDA1A9823E00B6AC3C99445BBE381884BEF545843E700A803C1432273E92956B3E68AA593D08DC383E9C59CB3DFEE8333EFEB3A5BD607C6DBC8C25DB3DF8D6AC3D0AB4173ED25747BEFE2597BDC0B2D73B005A28BDD6FC6D3E00B23F3E5056C2BCB89649BE42BF4F3E2E6720BED86368BD6BDB80BE086D353DB960853E6036863CE0C727BE98C0DDBC7238563EE892BA3D48CE8B3D9C1614BE10B5083DDD9B84BE88D823BE00515F3C265C403E18DE6D3D00133F3BC0DA37BC3B3A00BE9E0850BEF15510BEEC9E70BD3017EC3CA0521FBE18980F3E2EBA8BBE10A2533D6A161DBE026D6A3EB08E533E9DD577BE0A466B3E125080BEC5B48B3EB0677A3D0A9165BEE44F16BE1033563D24F82EBEC708813E89D405BE6D9689BE50485DBDA0EF8E3DA3D7853EFE86BCBD04F3633EDA7F80BE30697ABE9A1CC0BD082F693DF2E2263E949768BD007094BC78231D3DB4E13CBD80EBF03C3C88F8BD5AE7283E9A29683E8E6F683E483067BDD8F4D6BD483416BEE080703D3D2B813E684703BE4C77B83DD032DD3D4CDA07BE64F52F3E02BF783ED8E82E3E7E3E3E3E9EA8393ED76B15BE72E1C3BD198675BEC0D4283EC7767EBE98957B3DF2AD1E3E10A1DEBC4EC1723E949260BD166225BE4067353C8E76263E12D8403E843F1ABE05ED8ABE1EB94D3EAFA60CBE1A7681BEE2226D3E301FE53DD05985BE0256B1BD3A9C71BEA20620BE189834BE50771FBE7038C3BD50B795BC5618E2BD78370E3DEE47123E385F313E6471D3BD804C4CBEC66A703E6896E8BCC6F21E3E5E796D3E687E873D94CE203E2CA32E3E58F53A3E84A9063E80DC6EBCD0F1FD3C3738863E6041E83C6026B03CD64C623E61BC73BEB09AFBBC0AB818BE5A2FC2BDC13715BE560F2B3E77480FBEA01F163EECB789BE9CB9FD3DF0C7DE3C4EE110BE9418453E94C892BD007FF83D000E7B3DD32B87BE9A4AB1BD06EC513E26D9BABD049FD53D75FC8BBEA21FC0BDD0FCC13DFFA015BE08A2BA3D7E174F3E2C60533E266B713EE0CC7CBD3222083E5C71BF3DAFD8843E7419E63DD04F973D4061D6BC334B0EBEB896243EF244723ED0535D3D9EAA503EE6A8663EFBC43BBE368D56BE8C24D5BD42556F3E7EAFD4BDA68C793E006F05BE34822ABEF28EFBBDB031933C365D94BD0096D5BB610189BEB22E34BEA0A59F3D1CB243BE2AFD4C3EAA21213E706D323E96B8403EDE82583EC0C1143E7C1B0ABEFC6A26BDF5068B3EF02732BD407B223CCCE9ECBD6C940E3E283F4D3EAEAF673EE023B23D18FE333EC8E1C63D023487BE5E0F45BE9E3D7FBE88E7D83D54EDAF3D44B6FABD7CF7313E84CC83BD8838423E24FE40BE6AFF8DBDE4F7633EFCA3A4BDD2776BBE809E85BD3881913DE4F38C3D6C708A3D53A702BEC53989BE90918A3DE0F0A43D596D68BE0325873E50D3CF3CFC87FE3D4D70833EE8A93A3DBE8310BEF079423D78ABA63DBA55EABDF6D9563EB2E0083E81F909BE8DAA11BEC69B653E70A9FEBDEE4F703EA1B88B3E5319833EC01E443DB2FFA9BD44E6433EE064563D78AC2BBDE48A52BDAE997FBEF4117FBD7BBE7ABE32116D3E70A1A4BC04ADE53D00AB583DA450C1BD8067193B78AA78BD9001043DAE1782BE572655BEFDCC88BEB0B49C3D3068F2BC"> : tensor<75x4xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node63__model.layer-5.bias", tf_saved_model.exported_names = [], type = tensor<4xf32>, value = dense<0.000000e+00> : tensor<4xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node70__model.layer-6.kernel", tf_saved_model.exported_names = [], type = tensor<4x1xf32>, value = dense<[[0.725377917], [-0.0616925955], [0.417314529], [0.662358046]]> : tensor<4x1xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node71__model.layer-6.bias", tf_saved_model.exported_names = [], type = tensor<1xf32>, value = dense<0.000000e+00> : tensor<1xf32>} : () -> ()
  func.func @__inference_my_predict_1640(%arg0: tensor<16x28x28x3xf32> {tf._user_specified_name = "x", tf_saved_model.index_path = [0]}, %arg1: tensor<!tf_type.resource<tensor<3x3x3x2xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node26__model.layer-0.kernel"}, %arg2: tensor<!tf_type.resource<tensor<2xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node27__model.layer-0.bias"}, %arg3: tensor<!tf_type.resource<tensor<3x3x2x3xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node41__model.layer-2.kernel"}, %arg4: tensor<!tf_type.resource<tensor<3xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node42__model.layer-2.bias"}, %arg5: tensor<!tf_type.resource<tensor<75x4xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node62__model.layer-5.kernel"}, %arg6: tensor<!tf_type.resource<tensor<4xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node63__model.layer-5.bias"}, %arg7: tensor<!tf_type.resource<tensor<4x1xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node70__model.layer-6.kernel"}, %arg8: tensor<!tf_type.resource<tensor<1xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node71__model.layer-6.bias"}) -> (tensor<16x1xf32> {tf_saved_model.index_path = []}) attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<16x28x28x3>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful, tf_saved_model.exported_names = ["my_predict"]} {
    %cst = arith.constant dense<[3, 0, 1, 2]> : tensor<4xi32>
    %0 = "tf.ReadVariableOp"(%arg4) : (tensor<!tf_type.resource<tensor<3xf32>>>) -> tensor<3xf32>
    %1 = "tf.ReadVariableOp"(%arg3) : (tensor<!tf_type.resource<tensor<3x3x2x3xf32>>>) -> tensor<3x3x2x3xf32>
    %2 = "tf.ReadVariableOp"(%arg2) : (tensor<!tf_type.resource<tensor<2xf32>>>) -> tensor<2xf32>
    %3 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf_type.resource<tensor<3x3x3x2xf32>>>) -> tensor<3x3x3x2xf32>
    %4 = "tf.ReadVariableOp"(%arg8) : (tensor<!tf_type.resource<tensor<1xf32>>>) -> tensor<1xf32>
    %5 = "tf.ReadVariableOp"(%arg7) : (tensor<!tf_type.resource<tensor<4x1xf32>>>) -> tensor<4x1xf32>
    %6 = "tf.ReadVariableOp"(%arg6) : (tensor<!tf_type.resource<tensor<4xf32>>>) -> tensor<4xf32>
    %7 = "tf.ReadVariableOp"(%arg5) : (tensor<!tf_type.resource<tensor<75x4xf32>>>) -> tensor<75x4xf32>
    %8 = tensor.empty() : tensor<2x3x3x3xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3 : tensor<3x3x3x2xf32>) outs(%8 : tensor<2x3x3x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<2x3x3x3xf32>
    %cst_0 = arith.constant dense<[1, 2, 3, 0]> : tensor<4xi64>
    %10 = tensor.empty() : tensor<3x3x3x2xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9 : tensor<2x3x3x3xf32>) outs(%10 : tensor<3x3x3x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x3x3x2xf32>
    %12 = tensor.empty() : tensor<16x26x26x2xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %13 = linalg.fill ins(%cst_1 : f32) outs(%12 : tensor<16x26x26x2xf32>) -> tensor<16x26x26x2xf32>
    %14 = tensor.empty() : tensor<16x26x26x2xf32>
    %15 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %11 : tensor<16x28x28x3xf32>, tensor<3x3x3x2xf32>) outs(%13 : tensor<16x26x26x2xf32>) -> tensor<16x26x26x2xf32>
    %16 = linalg.generic {indexing_maps = [#map3, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %15 : tensor<2xf32>, tensor<16x26x26x2xf32>) outs(%14 : tensor<16x26x26x2xf32>) {
    ^bb0(%in: f32, %in_15: f32, %out: f32):
      %58 = arith.addf %in, %in_15 : f32
      linalg.yield %58 : f32
    } -> tensor<16x26x26x2xf32>
    %17 = tensor.empty() : tensor<16x26x26x2xf32>
    %18 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16 : tensor<16x26x26x2xf32>) outs(%17 : tensor<16x26x26x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_15 = arith.constant 0.000000e+00 : f32
      %cst_16 = arith.constant 3.40282347E+38 : f32
      %58 = arith.minf %in, %cst_16 : f32
      %59 = arith.maxf %58, %cst_15 : f32
      linalg.yield %59 : f32
    } -> tensor<16x26x26x2xf32>
    %cst_2 = arith.constant -3.40282347E+38 : f32
    %19 = tensor.empty() : tensor<16x13x13x2xf32>
    %20 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<16x13x13x2xf32>) -> tensor<16x13x13x2xf32>
    %21 = tensor.empty() : tensor<2x2xf32>
    %22 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%18, %21 : tensor<16x26x26x2xf32>, tensor<2x2xf32>) outs(%20 : tensor<16x13x13x2xf32>) -> tensor<16x13x13x2xf32>
    %23 = tensor.empty() : tensor<3x3x3x2xf32>
    %24 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<3x3x2x3xf32>) outs(%23 : tensor<3x3x3x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x3x3x2xf32>
    %cst_3 = arith.constant dense<[1, 2, 3, 0]> : tensor<4xi64>
    %25 = tensor.empty() : tensor<3x3x2x3xf32>
    %26 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24 : tensor<3x3x3x2xf32>) outs(%25 : tensor<3x3x2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x3x2x3xf32>
    %27 = tensor.empty() : tensor<16x11x11x3xf32>
    %cst_4 = arith.constant 0.000000e+00 : f32
    %28 = linalg.fill ins(%cst_4 : f32) outs(%27 : tensor<16x11x11x3xf32>) -> tensor<16x11x11x3xf32>
    %29 = tensor.empty() : tensor<16x11x11x3xf32>
    %30 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%22, %26 : tensor<16x13x13x2xf32>, tensor<3x3x2x3xf32>) outs(%28 : tensor<16x11x11x3xf32>) -> tensor<16x11x11x3xf32>
    %31 = linalg.generic {indexing_maps = [#map3, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0, %30 : tensor<3xf32>, tensor<16x11x11x3xf32>) outs(%29 : tensor<16x11x11x3xf32>) {
    ^bb0(%in: f32, %in_15: f32, %out: f32):
      %58 = arith.addf %in, %in_15 : f32
      linalg.yield %58 : f32
    } -> tensor<16x11x11x3xf32>
    %32 = tensor.empty() : tensor<16x11x11x3xf32>
    %33 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31 : tensor<16x11x11x3xf32>) outs(%32 : tensor<16x11x11x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_15 = arith.constant 0.000000e+00 : f32
      %cst_16 = arith.constant 3.40282347E+38 : f32
      %58 = arith.minf %in, %cst_16 : f32
      %59 = arith.maxf %58, %cst_15 : f32
      linalg.yield %59 : f32
    } -> tensor<16x11x11x3xf32>
    %cst_5 = arith.constant -3.40282347E+38 : f32
    %34 = tensor.empty() : tensor<16x5x5x3xf32>
    %35 = linalg.fill ins(%cst_5 : f32) outs(%34 : tensor<16x5x5x3xf32>) -> tensor<16x5x5x3xf32>
    %36 = tensor.empty() : tensor<2x2xf32>
    %37 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%33, %36 : tensor<16x11x11x3xf32>, tensor<2x2xf32>) outs(%35 : tensor<16x5x5x3xf32>) -> tensor<16x5x5x3xf32>
    %collapsed = tensor.collapse_shape %37 [[0], [1, 2, 3]] : tensor<16x5x5x3xf32> into tensor<16x75xf32>
    %expanded = tensor.expand_shape %collapsed [[0, 1], [2]] : tensor<16x75xf32> into tensor<1x16x75xf32>
    %expanded_6 = tensor.expand_shape %7 [[0, 1], [2]] : tensor<75x4xf32> into tensor<1x75x4xf32>
    %cst_7 = arith.constant 0.000000e+00 : f32
    %38 = tensor.empty() : tensor<1x16x4xf32>
    %39 = linalg.fill ins(%cst_7 : f32) outs(%38 : tensor<1x16x4xf32>) -> tensor<1x16x4xf32>
    %40 = linalg.batch_matmul ins(%expanded, %expanded_6 : tensor<1x16x75xf32>, tensor<1x75x4xf32>) outs(%39 : tensor<1x16x4xf32>) -> tensor<1x16x4xf32>
    %collapsed_8 = tensor.collapse_shape %40 [[0, 1], [2]] : tensor<1x16x4xf32> into tensor<16x4xf32>
    %expanded_9 = tensor.expand_shape %6 [[0, 1]] : tensor<4xf32> into tensor<1x4xf32>
    %41 = tensor.empty() : tensor<16x4xf32>
    %42 = linalg.generic {indexing_maps = [#map4, #map5, #map4], iterator_types = ["parallel", "parallel"]} ins(%collapsed_8, %expanded_9 : tensor<16x4xf32>, tensor<1x4xf32>) outs(%41 : tensor<16x4xf32>) {
    ^bb0(%in: f32, %in_15: f32, %out: f32):
      %58 = arith.addf %in, %in_15 : f32
      linalg.yield %58 : f32
    } -> tensor<16x4xf32>
    %43 = tensor.empty() : tensor<16x4xf32>
    %44 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%42 : tensor<16x4xf32>) outs(%43 : tensor<16x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_15 = arith.constant 0.000000e+00 : f32
      %cst_16 = arith.constant 3.40282347E+38 : f32
      %58 = arith.minf %in, %cst_16 : f32
      %59 = arith.maxf %58, %cst_15 : f32
      linalg.yield %59 : f32
    } -> tensor<16x4xf32>
    %expanded_10 = tensor.expand_shape %44 [[0, 1], [2]] : tensor<16x4xf32> into tensor<1x16x4xf32>
    %expanded_11 = tensor.expand_shape %5 [[0, 1], [2]] : tensor<4x1xf32> into tensor<1x4x1xf32>
    %cst_12 = arith.constant 0.000000e+00 : f32
    %45 = tensor.empty() : tensor<1x16x1xf32>
    %46 = linalg.fill ins(%cst_12 : f32) outs(%45 : tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %47 = linalg.batch_matmul ins(%expanded_10, %expanded_11 : tensor<1x16x4xf32>, tensor<1x4x1xf32>) outs(%46 : tensor<1x16x1xf32>) -> tensor<1x16x1xf32>
    %collapsed_13 = tensor.collapse_shape %47 [[0, 1], [2]] : tensor<1x16x1xf32> into tensor<16x1xf32>
    %expanded_14 = tensor.expand_shape %4 [[0, 1]] : tensor<1xf32> into tensor<1x1xf32>
    %48 = tensor.empty() : tensor<16x1xf32>
    %49 = linalg.generic {indexing_maps = [#map6, #map7, #map4], iterator_types = ["parallel", "parallel"]} ins(%collapsed_13, %expanded_14 : tensor<16x1xf32>, tensor<1x1xf32>) outs(%48 : tensor<16x1xf32>) {
    ^bb0(%in: f32, %in_15: f32, %out: f32):
      %58 = arith.addf %in, %in_15 : f32
      linalg.yield %58 : f32
    } -> tensor<16x1xf32>
    %50 = tensor.empty() : tensor<16x1xf32>
    %51 = linalg.generic {indexing_maps = [#map6, #map6, #map4], iterator_types = ["parallel", "parallel"]} ins(%49, %49 : tensor<16x1xf32>, tensor<16x1xf32>) outs(%50 : tensor<16x1xf32>) {
    ^bb0(%in: f32, %in_15: f32, %out: f32):
      %58 = arith.subf %in, %in_15 : f32
      linalg.yield %58 : f32
    } -> tensor<16x1xf32>
    %52 = tensor.empty() : tensor<16x1xf32>
    %53 = linalg.generic {indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel"]} ins(%51 : tensor<16x1xf32>) outs(%52 : tensor<16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %58 = math.exp %in : f32
      linalg.yield %58 : f32
    } -> tensor<16x1xf32>
    %54 = tensor.empty() : tensor<16x1xf32>
    %55 = linalg.generic {indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel"]} ins(%53 : tensor<16x1xf32>) outs(%54 : tensor<16x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_15 = arith.constant 1.000000e+00 : f32
      %58 = arith.divf %cst_15, %in : f32
      linalg.yield %58 : f32
    } -> tensor<16x1xf32>
    %56 = tensor.empty() : tensor<16x1xf32>
    %57 = linalg.generic {indexing_maps = [#map6, #map6, #map4], iterator_types = ["parallel", "parallel"]} ins(%53, %55 : tensor<16x1xf32>, tensor<16x1xf32>) outs(%56 : tensor<16x1xf32>) {
    ^bb0(%in: f32, %in_15: f32, %out: f32):
      %58 = arith.mulf %in, %in_15 : f32
      linalg.yield %58 : f32
    } -> tensor<16x1xf32>
    return %57 : tensor<16x1xf32>
  }
}

