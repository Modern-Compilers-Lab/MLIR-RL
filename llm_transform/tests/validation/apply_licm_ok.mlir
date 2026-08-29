// apply_licm: move loop-invariant code out of loop-like ops. We first tile the
// elementwise op to create scf.for loops, then run LICM on those loops. LICM is
// semantics-preserving (only hoists side-effect-free invariant code), so output
// must MATCH the baseline.

func.func @main(%in: memref<128x128xf64>, %out: memref<128x128xf64>) {
    linalg.generic {tag = "operation",
        indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%in : memref<128x128xf64>)
      outs(%out : memref<128x128xf64>) {
    ^bb0(%a: f64, %b: f64):
        %cst = arith.constant 2.0 : f64
        %r = arith.mulf %a, %cst : f64
        linalg.yield %r : f64
    }
    return
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %op = transform.structured.match attributes {tag = "operation"} in %arg0
            : (!transform.any_op) -> !transform.any_op

        %tiled, %loops:2 = transform.structured.tile_using_for %op tile_sizes [32, 32]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        transform.apply_licm to %loops#1 : !transform.any_op
        transform.apply_licm to %loops#0 : !transform.any_op

        transform.yield
    }
}
