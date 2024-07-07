module attributes {torch.debug_module_name = "Net"} {

  func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printMemrefF32(tensor<*xf32>)

  func.func @forward() -> tensor<32x120xf32>{

    %val = arith.constant 2.00000e+00 : f32
    %zero = arith.constant 0.00000e+00 : f32

    %out = bufferization.alloc_tensor() : tensor<32x10816xf32>
    %A = linalg.fill ins(%val : f32) outs(%out : tensor<32x10816xf32>) -> tensor<32x10816xf32>
    %out1 = bufferization.alloc_tensor() :tensor<10816x120xf32>
    %B = linalg.fill ins(%val : f32) outs(%out1 :tensor<10816x120xf32>) ->tensor<10816x120xf32>
    %out2 = bufferization.alloc_tensor() : tensor<32x120xf32>
    %C = linalg.fill ins(%zero : f32) outs(%out2 : tensor<32x120xf32>) -> tensor<32x120xf32>

    %t0 = func.call @nanoTime() : () -> (i64)

    %D = linalg.matmul ins(%A, %B: tensor<32x10816xf32>,tensor<10816x120xf32>) outs(%C: tensor<32x120xf32>) -> tensor<32x120xf32>
    
    %t = func.call @nanoTime() : () -> (i64)
    %delta = arith.subi %t, %t0 : i64
    %fp = arith.uitofp %delta : i64 to f64
    // func.call @printFlops(%fp) : (f64) -> ()
    func.call @printI64(%delta) : (i64) -> ()
    
    return %D : tensor<32x120xf32> 
  }

  func.func @main(){
      %outputmain = func.call @forward() : () -> tensor<32x120xf32>
      // %unranked = tensor.cast %outputmain : tensor<32x120xf32> to tensor<*xf32>
      // func.call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
      return
  }
}
