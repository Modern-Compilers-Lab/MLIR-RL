// XFAIL: level-3 — illegal but undetectable (outputs differ; both detectors silent).
// transform.affine.simplify_bounded_affine_ops with a WRONG user-supplied bound.
//
// The kernel computes a loop trip count from `affine.min(d0, 50)` where d0 is
// the real array size (100, via memref.dim). The true value is min(100, 50) =
// 50, so the loop writes out[0..50) = in[..]+1 and leaves out[50..100)
// untouched.
//
// The schedule asserts (FALSELY) that the bounded value d0 lies within
// [0, 40]. Trusting this bound, the simplifier proves d0 < 50 and folds
// affine.min(d0, 50) to just d0 (= 100). The loop now runs 0..100, writing all
// 100 elements. out[50..100) therefore differs from the baseline.
//
// This is a Category "wrong trusted bounds" miscompile: the dependence
// structure is unchanged and no value is reversed, so MLIR is silent and the
// EquivalenceVerifier is silent — only the numeric comparison catches it.
// Candidate level 3 (harness [FAIL]).

#m = affine_map<(d0) -> (d0, 50)>
func.func @main(%in: memref<100xf64>, %out: memref<100xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.0 : f64
  %n = memref.dim %in, %c0 : memref<100xf64>
  %m = affine.min #m(%n)
  scf.for %j = %c0 to %m step %c1 {
    %x = memref.load %in[%j] : memref<100xf64>
    %r = arith.addf %x, %cst : f64
    memref.store %r, %out[%j] : memref<100xf64>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %minop = transform.structured.match ops{["affine.min"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %dimop = transform.structured.match ops{["memref.dim"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    // FALSE bound: claim the real size (100) is within [0, 40].
    transform.affine.simplify_bounded_affine_ops %minop with [%dimop : !transform.any_op]
        within [0] and [40] : !transform.any_op
    transform.yield
  }
}
