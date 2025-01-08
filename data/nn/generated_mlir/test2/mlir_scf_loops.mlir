module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1482 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node21__model.layer-0.kernel", tf_saved_model.exported_names = [], type = tensor<2x4xf32>, value = dense<[[-0.366711378, 0.366582394, 0.264327288, -0.515119076], [-0.232851505, -0.044178009, -0.418082476, -0.119741678]]> : tensor<2x4xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node22__model.layer-0.bias", tf_saved_model.exported_names = [], type = tensor<4xf32>, value = dense<0.000000e+00> : tensor<4xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node29__model.layer-1.kernel", tf_saved_model.exported_names = [], type = tensor<4x8xf32>, value = dense<[[-0.518020332, -0.0076329708, -0.273415715, 0.446661174, -0.602498293, -0.121825993, 0.0328769088, -0.199197114], [-0.368024915, 0.63398689, -0.601587892, 3.564610e-01, 0.312985241, -0.690208554, -0.603075683, -0.648099839], [0.237009108, 0.482332766, 0.0121293664, -0.034699142, 0.0742671489, -0.331233412, -0.581755817, -0.152040184], [-0.084192872, -0.14643079, -0.460976422, -0.41481635, 0.611420094, -0.458459258, -0.137784958, -0.554757535]]> : tensor<4x8xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node30__model.layer-1.bias", tf_saved_model.exported_names = [], type = tensor<8xf32>, value = dense<0.000000e+00> : tensor<8xf32>} : () -> ()
  func.func @__inference_my_predict_1000(%arg0: tensor<1x2xf32> {tf._user_specified_name = "x", tf_saved_model.index_path = [0]}, %arg1: tensor<!tf_type.resource<tensor<2x4xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node21__model.layer-0.kernel"}, %arg2: tensor<!tf_type.resource<tensor<4xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node22__model.layer-0.bias"}, %arg3: tensor<!tf_type.resource<tensor<4x8xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node29__model.layer-1.kernel"}, %arg4: tensor<!tf_type.resource<tensor<8xf32>>> {tf._user_specified_name = "resource", tf_saved_model.bound_input = @"__sm_node30__model.layer-1.bias"}) -> (tensor<1x8xf32> {tf_saved_model.index_path = []}) attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<1x2>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful, tf_saved_model.exported_names = ["my_predict"]} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant -3.40282347E+38 : f32
    %cst_1 = arith.constant 3.40282347E+38 : f32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = "tf.ReadVariableOp"(%arg4) : (tensor<!tf_type.resource<tensor<8xf32>>>) -> tensor<8xf32>
    %1 = bufferization.to_memref %0 : memref<8xf32>
    %2 = "tf.ReadVariableOp"(%arg3) : (tensor<!tf_type.resource<tensor<4x8xf32>>>) -> tensor<4x8xf32>
    %3 = "tf.ReadVariableOp"(%arg2) : (tensor<!tf_type.resource<tensor<4xf32>>>) -> tensor<4xf32>
    %4 = bufferization.to_memref %3 : memref<4xf32>
    %5 = "tf.ReadVariableOp"(%arg1) : (tensor<!tf_type.resource<tensor<2x4xf32>>>) -> tensor<2x4xf32>
    %expanded = tensor.expand_shape %arg0 [[0, 1], [2]] : tensor<1x2xf32> into tensor<1x1x2xf32>
    %6 = bufferization.to_memref %expanded : memref<1x1x2xf32>
    %expanded_3 = tensor.expand_shape %5 [[0, 1], [2]] : tensor<2x4xf32> into tensor<1x2x4xf32>
    %7 = bufferization.to_memref %expanded_3 : memref<1x2x4xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x4xf32>
    affine.for %arg5 = 0 to 1 {
      affine.for %arg6 = 0 to 1 {
        affine.for %arg7 = 0 to 4 {
          affine.store %cst_2, %alloc[%arg5, %arg6, %arg7] : memref<1x1x4xf32>
        }
      }
    }
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x1x4xf32>
    memref.copy %alloc, %alloc_4 : memref<1x1x4xf32> to memref<1x1x4xf32>
    affine.for %arg5 = 0 to 1 {
      affine.for %arg6 = 0 to 1 {
        affine.for %arg7 = 0 to 4 {
          affine.for %arg8 = 0 to 2 {
            %16 = affine.load %6[%arg5, %arg6, %arg8] : memref<1x1x2xf32>
            %17 = affine.load %7[%arg5, %arg8, %arg7] : memref<1x2x4xf32>
            %18 = affine.load %alloc_4[%arg5, %arg6, %arg7] : memref<1x1x4xf32>
            %19 = arith.mulf %16, %17 : f32
            %20 = arith.addf %18, %19 : f32
            affine.store %20, %alloc_4[%arg5, %arg6, %arg7] : memref<1x1x4xf32>
          }
        }
      }
    }
    %8 = bufferization.to_tensor %alloc_4 : memref<1x1x4xf32>
    %collapsed = tensor.collapse_shape %8 [[0, 1, 2]] : tensor<1x1x4xf32> into tensor<4xf32>
    %9 = bufferization.to_memref %collapsed : memref<4xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
    affine.for %arg5 = 0 to 4 {
      %16 = affine.load %9[%arg5] : memref<4xf32>
      %17 = affine.load %4[%arg5] : memref<4xf32>
      %18 = arith.addf %16, %17 : f32
      %19 = arith.minimumf %18, %cst_1 : f32
      %20 = arith.maximumf %19, %cst_2 : f32
      affine.store %20, %alloc_5[%arg5] : memref<4xf32>
    }
    %10 = bufferization.to_tensor %alloc_5 : memref<4xf32>
    %expanded_6 = tensor.expand_shape %10 [[0, 1, 2]] : tensor<4xf32> into tensor<1x1x4xf32>
    %11 = bufferization.to_memref %expanded_6 : memref<1x1x4xf32>
    %expanded_7 = tensor.expand_shape %2 [[0, 1], [2]] : tensor<4x8xf32> into tensor<1x4x8xf32>
    %12 = bufferization.to_memref %expanded_7 : memref<1x4x8xf32>
    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<1x1x8xf32>
    affine.for %arg5 = 0 to 1 {
      affine.for %arg6 = 0 to 1 {
        affine.for %arg7 = 0 to 8 {
          affine.store %cst_2, %alloc_8[%arg5, %arg6, %arg7] : memref<1x1x8xf32>
        }
      }
    }
    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<1x1x8xf32>
    memref.copy %alloc_8, %alloc_9 : memref<1x1x8xf32> to memref<1x1x8xf32>
    affine.for %arg5 = 0 to 1 {
      affine.for %arg6 = 0 to 1 {
        affine.for %arg7 = 0 to 8 {
          affine.for %arg8 = 0 to 4 {
            %16 = affine.load %11[%arg5, %arg6, %arg8] : memref<1x1x4xf32>
            %17 = affine.load %12[%arg5, %arg8, %arg7] : memref<1x4x8xf32>
            %18 = affine.load %alloc_9[%arg5, %arg6, %arg7] : memref<1x1x8xf32>
            %19 = arith.mulf %16, %17 : f32
            %20 = arith.addf %18, %19 : f32
            affine.store %20, %alloc_9[%arg5, %arg6, %arg7] : memref<1x1x8xf32>
          }
        }
      }
    }
    %13 = bufferization.to_tensor %alloc_9 : memref<1x1x8xf32>
    %collapsed_10 = tensor.collapse_shape %13 [[0, 1, 2]] : tensor<1x1x8xf32> into tensor<8xf32>
    %14 = bufferization.to_memref %collapsed_10 : memref<8xf32>
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<8xf32>
    affine.for %arg5 = 0 to 8 {
      %16 = affine.load %14[%arg5] : memref<8xf32>
      %17 = affine.load %1[%arg5] : memref<8xf32>
      %18 = arith.addf %16, %17 : f32
      affine.store %18, %alloc_11[%arg5] : memref<8xf32>
    }
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    affine.store %cst_0, %alloc_12[] : memref<f32>
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.copy %alloc_12, %alloc_13 : memref<f32> to memref<f32>
    affine.for %arg5 = 0 to 8 {
      %16 = affine.load %alloc_11[%arg5] : memref<8xf32>
      %17 = affine.load %alloc_13[] : memref<f32>
      %18 = arith.maximumf %16, %17 : f32
      affine.store %18, %alloc_13[] : memref<f32>
    }
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<8xf32>
    affine.for %arg5 = 0 to 8 {
      %16 = affine.load %alloc_11[%arg5] : memref<8xf32>
      %17 = affine.load %alloc_13[] : memref<f32>
      %18 = arith.subf %16, %17 : f32
      %19 = math.exp %18 : f32
      affine.store %19, %alloc_14[%arg5] : memref<8xf32>
    }
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    affine.store %cst_2, %alloc_15[] : memref<f32>
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    memref.copy %alloc_15, %alloc_16 : memref<f32> to memref<f32>
    affine.for %arg5 = 0 to 8 {
      %16 = affine.load %alloc_14[%arg5] : memref<8xf32>
      %17 = affine.load %alloc_16[] : memref<f32>
      %18 = arith.addf %16, %17 : f32
      affine.store %18, %alloc_16[] : memref<f32>
    }
    %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<8xf32>
    affine.for %arg5 = 0 to 8 {
      %16 = affine.load %alloc_14[%arg5] : memref<8xf32>
      %17 = affine.load %alloc_16[] : memref<f32>
      %18 = arith.divf %cst, %17 : f32
      %19 = arith.mulf %16, %18 : f32
      affine.store %19, %alloc_17[%arg5] : memref<8xf32>
    }
    %15 = bufferization.to_tensor %alloc_17 : memref<8xf32>
    %expanded_18 = tensor.expand_shape %15 [[0, 1]] : tensor<8xf32> into tensor<1x8xf32>
    return %expanded_18 : tensor<1x8xf32>
  }
}

