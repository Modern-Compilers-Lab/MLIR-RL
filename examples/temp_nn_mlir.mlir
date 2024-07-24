module attributes {torch.debug_module_name = "Net"} {
func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
func.func private @printFlops(f64)
func.func private @printI64(i64)
func.func private @printNewline()
func.func private @printMemrefF32(tensor<*xf32>)


func.func @matmul() -> tensor<32x64x112x112xf32>{

%val = arith.constant 2.00000e+00 : f32
%zero = arith.constant 0.00000e+00 : f32

%tmp_input = bufferization.alloc_tensor() : tensor<32x3x230x230xf32>
%input = linalg.fill ins(%val : f32) outs(%tmp_input : tensor<32x3x230x230xf32>) -> tensor<32x3x230x230xf32>
%tmp_filter = bufferization.alloc_tensor() : tensor<64x3x7x7xf32>
%filter = linalg.fill ins(%val : f32) outs(%tmp_filter : tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf32>
%tmp_init = bufferization.alloc_tensor() : tensor<32x64x112x112xf32>
%init = linalg.fill ins(%val : f32) outs(%tmp_init : tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xf32>

%t0 = func.call @nanoTime() : () -> (i64)

%return_arg = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins (%input, %filter: tensor<32x3x230x230xf32>, tensor<64x3x7x7xf32>) outs (%init: tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xf32>
%t = func.call @nanoTime() : () -> (i64)
%delta = arith.subi %t, %t0 : i64
%fp = arith.uitofp %delta : i64 to f64
// func.call @printFlops(%fp) : (f64) -> ()
func.call @printI64(%delta) : (i64) -> ()
func.call @printNewline() : () -> ()

return %return_arg : tensor<32x64x112x112xf32> 
}

func.func @main(){
    %c1 = arith.constant 1: index
    %c0 = arith.constant 0 : index
    %n = arith.constant 2: index
    scf.for %i = %c0 to %n step %c1 {
    %outputmain = func.call @matmul() : () -> tensor<32x64x112x112xf32>
    }
    return
}
}
