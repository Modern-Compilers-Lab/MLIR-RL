// Legal tiling: pure elementwise op between two disjoint memrefs.
// There is no cross-iteration dependence, so 2D tiling is sound and the
// transformed code must produce the same result as the baseline.

func.func @tiling_elementwise_ok(%in: memref<128x128xf64>, %out: memref<128x128xf64>) {
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

        %tiled_op, %loops:2 = transform.structured.tile_using_for %op tile_sizes [32, 32]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}
