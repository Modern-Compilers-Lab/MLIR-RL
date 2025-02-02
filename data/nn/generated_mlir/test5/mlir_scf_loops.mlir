module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 1482 : i32}, tf_saved_model.semantics} {
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node21__model.layer-0.kernel", tf_saved_model.exported_names = [], type = tensor<2x4xf32>, value = dense<[[-0.0659065247, -0.71034503, -0.143171549, -0.260344028], [-0.69184041, -0.760239601, -0.711513996, -0.485834837]]> : tensor<2x4xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node22__model.layer-0.bias", tf_saved_model.exported_names = [], type = tensor<4xf32>, value = dense<0.000000e+00> : tensor<4xf32>} : () -> ()
  "tf_saved_model.global_tensor"() {is_mutable, sym_name = "__sm_node29__model.layer-1.kernel", tf_saved_model.exported_names = [], type = tensor<4x8xf32>, value = dense<[[-0.297534823, -0.0913531184, -0.409505337, 0.0201846361, 0.585041583, -0.560052037, 0.330370724, -0.0822428465], [-0.173333287, 0.462669194, 0.236135781, 0.55484134, 0.569796503, -0.187162876, 0.580581367, 0.25352478], [0.0387528539, -0.359654576, 0.0830879807, 0.244462192, 0.344339907, 0.616770446, -0.622811734, -0.163719416], [-0.704532623, -0.111978114, -0.384449035, 0.620973765, 0.159519911, -0.382416546, 0.422677457, 5.600330e-01]]> : tensor<4x8xf32>} : () -> ()
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

