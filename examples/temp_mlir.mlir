module attributes {torch.debug_module_name = "Net"} {
func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
func.func private @printFlops(f64)
func.func private @printI64(i64)
func.func private @printNewline()
func.func private @printMemrefF32(tensor<*xf32>)


func.func @matmul() -> tensor<128xf32>{

%val = arith.constant 2.00000e+00 : f32
%zero = arith.constant 0.00000e+00 : f32

%tmp_arg0 = bufferization.alloc_tensor() : tensor<128xf32>
%arg0 = linalg.fill ins(%val : f32) outs(%tmp_arg0 : tensor<128xf32>) -> tensor<128xf32>
%tmp_arg1 = bufferization.alloc_tensor() : tensor<128xf32>
%arg1 = linalg.fill ins(%val : f32) outs(%tmp_arg1 : tensor<128xf32>) -> tensor<128xf32>
%tmp_arg2 = bufferization.alloc_tensor() : tensor<128xf32>
%arg2 = linalg.fill ins(%val : f32) outs(%tmp_arg2 : tensor<128xf32>) -> tensor<128xf32>

%t0 = func.call @nanoTime() : () -> (i64)

%return_arg = linalg.add ins(%arg0, %arg1: tensor<128xf32>, tensor<128xf32>) outs(%arg2: tensor<128xf32>) -> tensor<128xf32>
%t = func.call @nanoTime() : () -> (i64)
%delta = arith.subi %t, %t0 : i64
%fp = arith.uitofp %delta : i64 to f64
// func.call @printFlops(%fp) : (f64) -> ()
func.call @printI64(%delta) : (i64) -> ()
func.call @printNewline() : () -> ()

return %return_arg : tensor<128xf32> 
}

func.func @main(){
    %c1 = arith.constant 1: index
    %c0 = arith.constant 0 : index
    %n = arith.constant 2: index
    scf.for %i = %c0 to %n step %c1 {
    %outputmain = func.call @matmul() : () -> tensor<128xf32>
    }
    return
}
}
