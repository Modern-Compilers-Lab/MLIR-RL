// All vector.* lowering/canonicalization pattern sets applied to a func that
// contains NO vector ops. Each pattern set only matches vector dialect ops, so
// here they are pure no-ops: nothing changes and output MATCHES the baseline.
// This confirms the pattern sets do not perturb non-vector IR (level 0).

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
        %func = transform.structured.match ops{["func.func"]} in %arg0
            : (!transform.any_op) -> !transform.any_op

        transform.apply_patterns to %func {
            transform.apply_patterns.vector.reduction_to_contract
            transform.apply_patterns.vector.transfer_permutation_patterns
            transform.apply_patterns.vector.lower_contraction
            transform.apply_patterns.vector.lower_outerproduct
            transform.apply_patterns.vector.lower_transfer
            transform.apply_patterns.vector.lower_transpose
            transform.apply_patterns.vector.lower_shape_cast
            transform.apply_patterns.vector.sink_ops
        } : !transform.any_op

        transform.yield
    }
}
