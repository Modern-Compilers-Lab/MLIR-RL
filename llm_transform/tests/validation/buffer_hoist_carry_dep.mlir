// XFAIL: level-3 — illegal but undetectable (outputs differ; both detectors silent).
// transform.bufferization.buffer_loop_hoisting introduces a loop-carried
// dependence through a LOCAL buffer.
//
// Inside the loop a temp buffer `tmp` (memref<2xf64>) is used as scratch.
// Each iteration writes ONLY tmp[i%2] (the slot for the current parity) and
// reads BOTH slots, summing them into out[i]. With the alloc INSIDE the loop
// (baseline) every iteration gets a fresh allocation, so the slot NOT written
// this iteration is independent scratch. After buffer_loop_hoisting the single
// shared `tmp` persists across iterations, so the slot written two iterations
// ago is still live and leaks into the current iteration's sum (a loop-carried
// alias through the local buffer). out[] therefore differs.
//
// The carried alias lives entirely in the LOCAL buffer; the EquivalenceVerifier
// inspects only function-argument memrefs and is blind. Candidate level 3.

func.func @main(%in: memref<64xf64>, %out: memref<64xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index

  scf.for %i = %c0 to %c64 step %c1 {
    %tmp = memref.alloc() : memref<2xf64>
    // write only the current-parity slot
    %p = arith.remui %i, %c2 : index
    %x = memref.load %in[%i] : memref<64xf64>
    memref.store %x, %tmp[%p] : memref<2xf64>
    // read both slots and sum
    %a = memref.load %tmp[%c0] : memref<2xf64>
    %b = memref.load %tmp[%c1] : memref<2xf64>
    %s = arith.addf %a, %b : f64
    memref.store %s, %out[%i] : memref<64xf64>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.bufferization.buffer_loop_hoisting %f : !transform.any_op
    transform.yield
  }
}
