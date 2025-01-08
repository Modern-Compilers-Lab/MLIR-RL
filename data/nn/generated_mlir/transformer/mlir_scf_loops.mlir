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
    %cst = arith.constant 0.44721359 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant -3.40282347E+38 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = "tf.ReadVariableOp"(%arg8) : (tensor<!tf_type.resource<tensor<5xf32>>>) -> tensor<5xf32>
    %1 = bufferization.to_memref %0 : memref<5xf32>
    %2 = "tf.ReadVariableOp"(%arg7) : (tensor<!tf_type.resource<tensor<3x5x5xf32>>>) -> tensor<3x5x5xf32>
    %3 = "tf.ReadVariableOp"(%arg4) : (tensor<!tf_type.resource<tensor<3x5xf32>>>) -> tensor<3x5xf32>
    %4 = bufferization.to_memref %3 : memref<3x5xf32>
    %5 = "tf.ReadVariableOp"(%arg3) : (tensor<!tf_type.resource<tensor<5x3x5xf32>>>) -> tensor<5x3x5xf32>
    %6 = "tf.ReadVariableOp"(%arg2) : (tensor<!tf_type.resource<tensor<3x5xf32>>>) -> tensor<3x5xf32>
    %7 = bufferization.to_memref %6 : memref<3x5xf32>
    %8 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf_type.resource<tensor<5x3x5xf32>>>) -> tensor<5x3x5xf32>
    %9 = "tf.ReadVariableOp"(%arg6) : (tensor<!tf_type.resource<tensor<3x5xf32>>>) -> tensor<3x5xf32>
    %10 = bufferization.to_memref %9 : memref<3x5xf32>
    %11 = "tf.ReadVariableOp"(%arg5) : (tensor<!tf_type.resource<tensor<5x3x5xf32>>>) -> tensor<5x3x5xf32>
    %collapsed = tensor.collapse_shape %5 [[0], [1, 2]] : tensor<5x3x5xf32> into tensor<5x15xf32>
    %12 = "tf.BatchMatMulV2"(%arg0, %collapsed) {adj_x = false, adj_y = false} : (tensor<16x7x5xf32>, tensor<5x15xf32>) -> tensor<16x7x15xf32>
    %expanded = tensor.expand_shape %12 [[0], [1], [2, 3]] : tensor<16x7x15xf32> into tensor<16x7x3x5xf32>
    %13 = bufferization.to_memref %expanded : memref<16x7x3x5xf32>
    %collapsed_3 = tensor.collapse_shape %8 [[0], [1, 2]] : tensor<5x3x5xf32> into tensor<5x15xf32>
    %14 = "tf.BatchMatMulV2"(%arg0, %collapsed_3) {adj_x = false, adj_y = false} : (tensor<16x7x5xf32>, tensor<5x15xf32>) -> tensor<16x7x15xf32>
    %expanded_4 = tensor.expand_shape %14 [[0], [1], [2, 3]] : tensor<16x7x15xf32> into tensor<16x7x3x5xf32>
    %15 = bufferization.to_memref %expanded_4 : memref<16x7x3x5xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x3x7x5xf32>
    affine.for %arg9 = 0 to 16 {
      affine.for %arg10 = 0 to 3 {
        affine.for %arg11 = 0 to 7 {
          affine.for %arg12 = 0 to 5 {
            %34 = affine.load %13[%arg9, %arg11, %arg10, %arg12] : memref<16x7x3x5xf32>
            %35 = affine.load %4[%arg10, %arg12] : memref<3x5xf32>
            %36 = arith.addf %34, %35 : f32
            affine.store %36, %alloc[%arg9, %arg10, %arg11, %arg12] : memref<16x3x7x5xf32>
          }
        }
      }
    }
    %16 = bufferization.to_tensor %alloc : memref<16x3x7x5xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<16x3x5x7xf32>
    affine.for %arg9 = 0 to 16 {
      affine.for %arg10 = 0 to 3 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 7 {
            %34 = affine.load %15[%arg9, %arg12, %arg10, %arg11] : memref<16x7x3x5xf32>
            %35 = affine.load %7[%arg10, %arg11] : memref<3x5xf32>
            %36 = arith.addf %34, %35 : f32
            %37 = arith.mulf %36, %cst : f32
            affine.store %37, %alloc_5[%arg9, %arg10, %arg11, %arg12] : memref<16x3x5x7xf32>
          }
        }
      }
    }
    %17 = bufferization.to_tensor %alloc_5 : memref<16x3x5x7xf32>
    %collapsed_6 = tensor.collapse_shape %16 [[0, 1], [2], [3]] : tensor<16x3x7x5xf32> into tensor<48x7x5xf32>
    %18 = bufferization.to_memref %collapsed_6 : memref<48x7x5xf32>
    %collapsed_7 = tensor.collapse_shape %17 [[0, 1], [2], [3]] : tensor<16x3x5x7xf32> into tensor<48x5x7xf32>
    %19 = bufferization.to_memref %collapsed_7 : memref<48x5x7xf32>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<48x7x7xf32>
    affine.for %arg9 = 0 to 48 {
      affine.for %arg10 = 0 to 7 {
        affine.for %arg11 = 0 to 7 {
          affine.store %cst_2, %alloc_8[%arg9, %arg10, %arg11] : memref<48x7x7xf32>
        }
      }
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<48x7x7xf32>
    memref.copy %alloc_8, %alloc_9 : memref<48x7x7xf32> to memref<48x7x7xf32>
    affine.for %arg9 = 0 to 48 {
      affine.for %arg10 = 0 to 7 {
        affine.for %arg11 = 0 to 7 {
          affine.for %arg12 = 0 to 5 {
            %34 = affine.load %18[%arg9, %arg10, %arg12] : memref<48x7x5xf32>
            %35 = affine.load %19[%arg9, %arg12, %arg11] : memref<48x5x7xf32>
            %36 = affine.load %alloc_9[%arg9, %arg10, %arg11] : memref<48x7x7xf32>
            %37 = arith.mulf %34, %35 : f32
            %38 = arith.addf %36, %37 : f32
            affine.store %38, %alloc_9[%arg9, %arg10, %arg11] : memref<48x7x7xf32>
          }
        }
      }
    }
    %20 = bufferization.to_tensor %alloc_9 : memref<48x7x7xf32>
    %expanded_10 = tensor.expand_shape %20 [[0, 1], [2], [3]] : tensor<48x7x7xf32> into tensor<16x3x7x7xf32>
    %21 = bufferization.to_memref %expanded_10 : memref<16x3x7x7xf32>
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<16x3x7x7xf32>
    affine.for %arg9 = 0 to 16 {
      affine.for %arg10 = 0 to 3 {
        affine.for %arg11 = 0 to 7 {
          affine.for %arg12 = 0 to 7 {
            %34 = affine.load %21[%arg9, %arg10, %arg12, %arg11] : memref<16x3x7x7xf32>
            affine.store %34, %alloc_11[%arg9, %arg10, %arg11, %arg12] : memref<16x3x7x7xf32>
          }
        }
      }
    }
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<16x3x7xf32>
    affine.for %arg9 = 0 to 16 {
      affine.for %arg10 = 0 to 3 {
        affine.for %arg11 = 0 to 7 {
          affine.store %cst_1, %alloc_12[%arg9, %arg10, %arg11] : memref<16x3x7xf32>
        }
      }
    }
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<16x3x7xf32>
    memref.copy %alloc_12, %alloc_13 : memref<16x3x7xf32> to memref<16x3x7xf32>
    affine.for %arg9 = 0 to 16 {
      affine.for %arg10 = 0 to 3 {
        affine.for %arg11 = 0 to 7 {
          affine.for %arg12 = 0 to 7 {
            %34 = affine.load %alloc_11[%arg9, %arg10, %arg11, %arg12] : memref<16x3x7x7xf32>
            %35 = affine.load %alloc_13[%arg9, %arg10, %arg11] : memref<16x3x7xf32>
            %36 = arith.maximumf %34, %35 : f32
            affine.store %36, %alloc_13[%arg9, %arg10, %arg11] : memref<16x3x7xf32>
          }
        }
      }
    }
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<16x3x7x7xf32>
    affine.for %arg9 = 0 to 16 {
      affine.for %arg10 = 0 to 3 {
        affine.for %arg11 = 0 to 7 {
          affine.for %arg12 = 0 to 7 {
            %34 = affine.load %alloc_11[%arg9, %arg10, %arg11, %arg12] : memref<16x3x7x7xf32>
            %35 = affine.load %alloc_13[%arg9, %arg10, %arg11] : memref<16x3x7xf32>
            %36 = arith.subf %34, %35 : f32
            %37 = math.exp %36 : f32
            affine.store %37, %alloc_14[%arg9, %arg10, %arg11, %arg12] : memref<16x3x7x7xf32>
          }
        }
      }
    }
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<16x3x7xf32>
    affine.for %arg9 = 0 to 16 {
      affine.for %arg10 = 0 to 3 {
        affine.for %arg11 = 0 to 7 {
          affine.store %cst_2, %alloc_15[%arg9, %arg10, %arg11] : memref<16x3x7xf32>
        }
      }
    }
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<16x3x7xf32>
    memref.copy %alloc_15, %alloc_16 : memref<16x3x7xf32> to memref<16x3x7xf32>
    affine.for %arg9 = 0 to 16 {
      affine.for %arg10 = 0 to 3 {
        affine.for %arg11 = 0 to 7 {
          affine.for %arg12 = 0 to 7 {
            %34 = affine.load %alloc_14[%arg9, %arg10, %arg11, %arg12] : memref<16x3x7x7xf32>
            %35 = affine.load %alloc_16[%arg9, %arg10, %arg11] : memref<16x3x7xf32>
            %36 = arith.addf %34, %35 : f32
            affine.store %36, %alloc_16[%arg9, %arg10, %arg11] : memref<16x3x7xf32>
          }
        }
      }
    }
    %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<16x3x7x7xf32>
    affine.for %arg9 = 0 to 16 {
      affine.for %arg10 = 0 to 3 {
        affine.for %arg11 = 0 to 7 {
          affine.for %arg12 = 0 to 7 {
            %34 = affine.load %alloc_14[%arg9, %arg10, %arg11, %arg12] : memref<16x3x7x7xf32>
            %35 = affine.load %alloc_16[%arg9, %arg10, %arg11] : memref<16x3x7xf32>
            %36 = arith.divf %cst_0, %35 : f32
            %37 = arith.mulf %34, %36 : f32
            affine.store %37, %alloc_17[%arg9, %arg10, %arg11, %arg12] : memref<16x3x7x7xf32>
          }
        }
      }
    }
    %22 = bufferization.to_tensor %alloc_17 : memref<16x3x7x7xf32>
    %collapsed_18 = tensor.collapse_shape %11 [[0], [1, 2]] : tensor<5x3x5xf32> into tensor<5x15xf32>
    %23 = "tf.BatchMatMulV2"(%arg0, %collapsed_18) {adj_x = false, adj_y = false} : (tensor<16x7x5xf32>, tensor<5x15xf32>) -> tensor<16x7x15xf32>
    %expanded_19 = tensor.expand_shape %23 [[0], [1], [2, 3]] : tensor<16x7x15xf32> into tensor<16x7x3x5xf32>
    %24 = bufferization.to_memref %expanded_19 : memref<16x7x3x5xf32>
    %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<16x3x7x5xf32>
    affine.for %arg9 = 0 to 16 {
      affine.for %arg10 = 0 to 3 {
        affine.for %arg11 = 0 to 7 {
          affine.for %arg12 = 0 to 5 {
            %34 = affine.load %24[%arg9, %arg11, %arg10, %arg12] : memref<16x7x3x5xf32>
            %35 = affine.load %10[%arg10, %arg12] : memref<3x5xf32>
            %36 = arith.addf %34, %35 : f32
            affine.store %36, %alloc_20[%arg9, %arg10, %arg11, %arg12] : memref<16x3x7x5xf32>
          }
        }
      }
    }
    %25 = bufferization.to_tensor %alloc_20 : memref<16x3x7x5xf32>
    %collapsed_21 = tensor.collapse_shape %22 [[0, 1], [2], [3]] : tensor<16x3x7x7xf32> into tensor<48x7x7xf32>
    %26 = bufferization.to_memref %collapsed_21 : memref<48x7x7xf32>
    %collapsed_22 = tensor.collapse_shape %25 [[0, 1], [2], [3]] : tensor<16x3x7x5xf32> into tensor<48x7x5xf32>
    %27 = bufferization.to_memref %collapsed_22 : memref<48x7x5xf32>
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<48x7x5xf32>
    affine.for %arg9 = 0 to 48 {
      affine.for %arg10 = 0 to 7 {
        affine.for %arg11 = 0 to 5 {
          affine.store %cst_2, %alloc_23[%arg9, %arg10, %arg11] : memref<48x7x5xf32>
        }
      }
    }
    %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<48x7x5xf32>
    memref.copy %alloc_23, %alloc_24 : memref<48x7x5xf32> to memref<48x7x5xf32>
    affine.for %arg9 = 0 to 48 {
      affine.for %arg10 = 0 to 7 {
        affine.for %arg11 = 0 to 5 {
          affine.for %arg12 = 0 to 7 {
            %34 = affine.load %26[%arg9, %arg10, %arg12] : memref<48x7x7xf32>
            %35 = affine.load %27[%arg9, %arg12, %arg11] : memref<48x7x5xf32>
            %36 = affine.load %alloc_24[%arg9, %arg10, %arg11] : memref<48x7x5xf32>
            %37 = arith.mulf %34, %35 : f32
            %38 = arith.addf %36, %37 : f32
            affine.store %38, %alloc_24[%arg9, %arg10, %arg11] : memref<48x7x5xf32>
            // %alloc_24[%arg9, %arg10, %arg11] <- %alloc_24[%arg9, %arg10, %arg11] + %26[%arg9, %arg10, %arg12] * %27[%arg9, %arg12, %arg11]
          }
        }
      }
    }
    %28 = bufferization.to_tensor %alloc_24 : memref<48x7x5xf32>
    %expanded_25 = tensor.expand_shape %28 [[0, 1], [2], [3]] : tensor<48x7x5xf32> into tensor<16x3x7x5xf32>
    %29 = bufferization.to_memref %expanded_25 : memref<16x3x7x5xf32>
    %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<16x7x3x5xf32>
    affine.for %arg9 = 0 to 16 {
      affine.for %arg10 = 0 to 7 {
        affine.for %arg11 = 0 to 3 {
          affine.for %arg12 = 0 to 5 {
            %34 = affine.load %29[%arg9, %arg11, %arg10, %arg12] : memref<16x3x7x5xf32>
            affine.store %34, %alloc_26[%arg9, %arg10, %arg11, %arg12] : memref<16x7x3x5xf32>
          }
        }
      }
    }
    %30 = bufferization.to_tensor %alloc_26 : memref<16x7x3x5xf32>
    %collapsed_27 = tensor.collapse_shape %30 [[0], [1], [2, 3]] : tensor<16x7x3x5xf32> into tensor<16x7x15xf32>
    %collapsed_28 = tensor.collapse_shape %2 [[0, 1], [2]] : tensor<3x5x5xf32> into tensor<15x5xf32>
    %31 = "tf.BatchMatMulV2"(%collapsed_27, %collapsed_28) {adj_x = false, adj_y = false} : (tensor<16x7x15xf32>, tensor<15x5xf32>) -> tensor<16x7x5xf32>
    %32 = bufferization.to_memref %31 : memref<16x7x5xf32>
    %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<16x7x5xf32>
    affine.for %arg9 = 0 to 16 {
      affine.for %arg10 = 0 to 7 {
        affine.for %arg11 = 0 to 5 {
          %34 = affine.load %32[%arg9, %arg10, %arg11] : memref<16x7x5xf32>
          %35 = affine.load %1[%arg11] : memref<5xf32>
          %36 = arith.addf %34, %35 : f32
          affine.store %36, %alloc_29[%arg9, %arg10, %arg11] : memref<16x7x5xf32>
        }
      }
    }
    %33 = bufferization.to_tensor %alloc_29 : memref<16x7x5xf32>
    return %33 : tensor<16x7x5xf32>
  }
}

