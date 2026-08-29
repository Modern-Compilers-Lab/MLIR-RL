// transform.affine.simplify_min_max_affine_ops — purely algebraic
// simplification of affine.min/max, no user-supplied bounds to lie about.
//
// Here affine.min(d0, d0 + 5) is always equal to d0 (since d0 <= d0 + 5 for
// all d0), so the transform soundly folds it to d0. The loop trip count is
// unchanged and outputs match the baseline exactly.
//
// Because the simplification is algebraically valid, there is no input the
// caller can supply to make it produce a wrong result: it does not trust any
// external bound. Expected: outputs match => level 0 (SAFE).

#m = affine_map<(d0) -> (d0, d0 + 5)>
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
    transform.affine.simplify_min_max_affine_ops %minop : !transform.any_op
    transform.yield
  }
}
