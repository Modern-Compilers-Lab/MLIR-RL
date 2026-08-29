// XFAIL: level-3 — illegal but undetectable (outputs differ; both detectors silent).
// transform.memref.multibuffer with {skip_analysis}.
//
// An scf.for loop carries a value through a LOCAL temp buffer `tmp`
// (memref.alloc declared just outside the loop): each iteration reads the
// running sum written by the PREVIOUS iteration (a prefix-sum / genuine
// loop-carried dependence through the local allocation), then writes the
// updated sum for the NEXT iteration. `out[i]` is the prefix sum up to i.
//
// multibuffer expands `tmp` by `factor = 2` and indexes it by `i mod 2`, so
// iteration i reads sub-buffer `i % 2` while iteration i-1 wrote sub-buffer
// `(i-1) % 2` — a DIFFERENT physical buffer. The carried read no longer sees
// the previous iteration's write, so the prefix sum is destroyed and out[]
// differs from the baseline.
//
// Default mode runs a dependence analysis and REFUSES (carried dep) => safe
// (level 0). {skip_analysis} applies the rewrite blindly => wrong result.
// The corruption flows entirely through the LOCAL buffer `tmp`; the
// EquivalenceVerifier only inspects function-argument memrefs, so it is blind.
// Candidate level 3.

func.func @main(%in: memref<64xf64>, %out: memref<64xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  %tmp = memref.alloc() : memref<1xf64>

  scf.for %i = %c0 to %c64 step %c1 {
    // read carried running sum written by previous iteration
    %prev = memref.load %tmp[%c0] : memref<1xf64>
    %x = memref.load %in[%i] : memref<64xf64>
    %s = arith.addf %x, %prev : f64
    memref.store %s, %out[%i] : memref<64xf64>
    // write the running sum for the next iteration
    memref.store %s, %tmp[%c0] : memref<1xf64>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %alloc = transform.structured.match ops{["memref.alloc"]} in %arg0
        : (!transform.any_op) -> !transform.op<"memref.alloc">
    %mb = transform.memref.multibuffer %alloc {factor = 2 : i64, skip_analysis}
        : (!transform.op<"memref.alloc">) -> !transform.any_op
    transform.yield
  }
}
