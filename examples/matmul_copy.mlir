func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
func.func private @printFlops(f64)
func.func private @printI64(i64)
func.func private @printMemrefF32(tensor<*xf32>)

!TTa = tensor<1200x1500xf32>
!TTb = tensor<1500x1000xf32>
!TTc = tensor<1200x1000xf32>


func.func @matmul() -> !TTc{


  %val = arith.constant 2.00000e+00 : f32
  %zero = arith.constant 0.00000e+00 : f32

  %out = bufferization.alloc_tensor() : !TTa
  %A = linalg.fill ins(%val : f32) outs(%out : !TTa) -> !TTa
  %out1 = bufferization.alloc_tensor() : !TTb
  %B = linalg.fill ins(%val : f32) outs(%out1 : !TTb) -> !TTb
  %out2 = bufferization.alloc_tensor() : !TTc
  %C = linalg.fill ins(%zero : f32) outs(%out2 : !TTc) -> !TTc




  %t0 = func.call @nanoTime() : () -> (i64)

  %D = linalg.matmul ins(%A, %B: !TTa, !TTb)
                    outs(%C: !TTc) -> !TTc
  
  %t = func.call @nanoTime() : () -> (i64)
  %delta = arith.subi %t, %t0 : i64
  %fp = arith.uitofp %delta : i64 to f64
  func.call @printFlops(%fp) : (f64) -> ()
  //func.call @printI64(%delta) : (i64) -> ()

  
  
  return %D : !TTc 
}

func.func @main(){


    %outputmain = func.call @matmul() : () -> !TTc
    
    //%unranked = tensor.cast %outputmain : !TTc to tensor<*xf32>
    //call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
}

  transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):

  // The original fill op which will be fused into the outer scf.forall created by
  // tiling the convolution.
  %original_fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op

  // TODO: Add a transform.structured.specialize that can match a few different ops
  // Then, this reduces to just a linalg.matmul and we can reuse existing strategies.
  %named_conv = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op

  // Step 1. Tile to forall and sequential scf.for.
  // ======================================================
  %forall_l1, %conv_l1 = transform.structured.tile_to_forall_op %named_conv tile_sizes [2, 200]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  
  transform.structured.fuse_into_containing_op %original_fill into %forall_l1
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

 %1 = transform.structured.generalize %conv_l1 : (!transform.any_op) -> !transform.any_op

  transform.structured.interchange %1 iterator_interchange = [1, 2, 0] : (!transform.any_op) -> !transform.any_op

}


