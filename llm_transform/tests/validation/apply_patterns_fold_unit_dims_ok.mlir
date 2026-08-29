// fold_unit_extent_dims_via_slices and _via_reshapes: these patterns rewrite
// linalg ops that have unit (size-1) extent dimensions into lower-rank forms.
// The kernel below operates on a tensor with a unit middle dimension
// (1x128 -> via a 128x1x128 op). The patterns are semantics-preserving rank
// reductions; output must MATCH the baseline.

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @main(%in: tensor<128x1x128xf64>, %out: tensor<128x1x128xf64>) -> tensor<128x1x128xf64> {
    %0 = linalg.generic {tag = "operation",
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%in : tensor<128x1x128xf64>)
      outs(%out : tensor<128x1x128xf64>) {
    ^bb0(%a: f64, %b: f64):
        %cst = arith.constant 3.0 : f64
        %r = arith.mulf %a, %cst : f64
        linalg.yield %r : f64
    } -> tensor<128x1x128xf64>
    return %0 : tensor<128x1x128xf64>
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %func = transform.structured.match ops{["func.func"]} in %arg0
            : (!transform.any_op) -> !transform.any_op

        transform.apply_patterns to %func {
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
        } : !transform.any_op
        transform.apply_patterns to %func {
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
        } : !transform.any_op

        transform.yield
    }
}
