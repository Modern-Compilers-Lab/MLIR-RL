module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}

  func.func @main(
    %arg0: tensor<[N]x[C]x[H]x[W]xf64>,
    %arg1: tensor<[F]x[C]x[KH]x[KW]xf64>,
    %arg2: tensor<[N]x[F]x[OH]x[OW]xf64>
  ) -> (tensor<[N]x[F]x[OH]x[OW]xf64>, i64)
  attributes {llvm.emit_c_interface} {

    %0 = call @nanoTime() : () -> i64

    %1 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>,
        strides   = dense<1> : tensor<2xi64> }
      ins(
        %arg0, %arg1 :
        tensor<[N]x[C]x[H]x[W]xf64>,
        tensor<[F]x[C]x[KH]x[KW]xf64>
      )
      outs(
        %arg2 : tensor<[N]x[F]x[OH]x[OW]xf64>
      ) -> tensor<[N]x[F]x[OH]x[OW]xf64>

    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64

    return %1, %3 : tensor<[N]x[F]x[OH]x[OW]xf64>, i64
  }
}
