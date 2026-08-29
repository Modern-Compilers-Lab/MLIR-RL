// transform.memref.multibuffer in DEFAULT mode (no skip_analysis).
//
// Identical loop-carried-dependence kernel as multibuffer_carried_dep.mlir:
// `tmp` carries a running sum across iterations. In default mode multibuffer
// first runs an analysis and REFUSES to apply when it cannot prove the absence
// of a cross-iteration dependence — which is exactly the case here. The
// transform therefore fails to apply ("op failed to multibuffer"), the harness
// reports a [transformation exception], and no miscompile occurs.
//
// This is the SAFE counterpart proving the danger is gated on skip_analysis.
// Expected: transform refuses to apply => level 0.

func.func @main(%in: memref<64xf64>, %out: memref<64xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  %tmp = memref.alloc() : memref<1xf64>

  scf.for %i = %c0 to %c64 step %c1 {
    %prev = memref.load %tmp[%c0] : memref<1xf64>
    %x = memref.load %in[%i] : memref<64xf64>
    %s = arith.addf %x, %prev : f64
    memref.store %s, %out[%i] : memref<64xf64>
    memref.store %s, %tmp[%c0] : memref<1xf64>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %alloc = transform.structured.match ops{["memref.alloc"]} in %arg0
        : (!transform.any_op) -> !transform.op<"memref.alloc">
    %mb = transform.memref.multibuffer %alloc {factor = 2 : i64}
        : (!transform.op<"memref.alloc">) -> !transform.any_op
    transform.yield
  }
}
