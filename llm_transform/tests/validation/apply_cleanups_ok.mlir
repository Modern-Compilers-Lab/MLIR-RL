// Cleanup ops: apply_cse / apply_dce / apply_licm applied to the func body.
// These are semantics-preserving normalizations. On a plain elementwise kernel
// they either fire harmlessly or are no-ops; output must MATCH the baseline.

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
        %c2 = arith.constant 2.0 : f64
        %r = arith.mulf %a, %cst : f64
        %s = arith.mulf %r, %c2 : f64
        linalg.yield %s : f64
    }
    return
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %func = transform.structured.match ops{["func.func"]} in %arg0
            : (!transform.any_op) -> !transform.any_op

        transform.apply_cse to %func : !transform.any_op
        transform.apply_dce to %func : !transform.any_op

        transform.yield
    }
}
