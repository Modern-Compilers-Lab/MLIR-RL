// Legal interchange: pure elementwise op between two disjoint memrefs.
// With no cross-iteration dependence, any permutation of the loops is
// semantically equivalent.

func.func @interchange_elementwise_ok(%in: memref<100x100xf64>, %out: memref<100x100xf64>) {
    linalg.generic {tag = "operation",
        indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%in : memref<100x100xf64>)
      outs(%out : memref<100x100xf64>) {
    ^bb0(%a: f64, %b: f64):
        %cst = arith.constant 1.0 : f64
        %r = arith.addf %a, %cst : f64
        linalg.yield %r : f64
    }
    return
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %op = transform.structured.match attributes {tag = "operation"} in %arg0
            : (!transform.any_op) -> !transform.any_op

        %interchanged = transform.structured.interchange %op iterator_interchange = [1, 0]
            : (!transform.any_op) -> !transform.any_op

        transform.yield
    }
}
