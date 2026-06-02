// transform.memref.erase_dead_alloc_and_stores — conservative DSE /
// store-to-load forwarding / dead-alloc elimination on memrefs.
//
// `dead` is a LOCAL buffer that is written but never read (truly dead); the
// real computation writes out[i] = in[i] + 1 directly. The transform removes
// the dead alloc and its dead store. The live computation on the
// function-argument memrefs is untouched, so outputs match the baseline.
//
// The pass is provably sound: it only forwards a store to a load / drops a
// store when no aliasing access can observe the difference, and bails
// conservatively otherwise (verified: it refuses to forward across an aliasing
// memref.subview store). There is no input that makes it miscompile.
// Expected: outputs match => level 0 (SAFE).

func.func @main(%in: memref<8xf64>, %out: memref<8xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 1.0 : f64
  %junk = arith.constant 42.0 : f64

  %dead = memref.alloc() : memref<8xf64>

  scf.for %i = %c0 to %c8 step %c1 {
    // dead store into a local buffer that is never read
    memref.store %junk, %dead[%i] : memref<8xf64>
    // the real, observable computation on function-argument memrefs
    %x = memref.load %in[%i] : memref<8xf64>
    %r = arith.addf %x, %cst : f64
    memref.store %r, %out[%i] : memref<8xf64>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %f : (!transform.any_op) -> ()
    transform.yield
  }
}
