func.func @func_call(%extracted_slice: tensor<16x3x229x229xf32>, %extracted_slice_1: tensor<4x3x7x7xf32>, %extracted_slice_2: tensor<16x4x112x112xf32>) -> tensor<16x4x112x112xf32> {
  %ret = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>, tag = "operation_1"} ins(%extracted_slice, %extracted_slice_1 : tensor<16x3x229x229xf32>, tensor<4x3x7x7xf32>) outs(%extracted_slice_2 : tensor<16x4x112x112xf32>) -> tensor<16x4x112x112xf32>
  return %ret : tensor<16x4x112x112xf32>
}