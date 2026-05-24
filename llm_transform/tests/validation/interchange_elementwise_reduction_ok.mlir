// Reduction-dimension analogue of interchange_elementwise_ok.mlir. Instead of
// a pure elementwise map, each output element is a sum over an innermost
// "reduction" dimension d2: out[d0, d1] = sum_d2 in[d0, d1, d2] + 1. The input
// and output memrefs are disjoint, so there is no cross-iteration dependence.
//
// The interchange swaps the two parallel dims (iterator_interchange = [1, 0, 2],
// reduction left innermost). With no cross-iteration dependence any permutation
// of the parallel loops is semantically equivalent, and the innermost reduction
// keeps its accumulation order, so the transform is legal.

func.func @interchange_elementwise_reduction_ok(%in: memref<100x100x4xf64>, %out: memref<100x100xf64>) {
    linalg.generic {tag = "operation",
        indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%in : memref<100x100x4xf64>)
      outs(%out : memref<100x100xf64>) {
    ^bb0(%a: f64, %acc: f64):
        %cst = arith.constant 1.0 : f64
        %r = arith.addf %a, %cst : f64
        %s = arith.addf %acc, %r : f64
        linalg.yield %s : f64
    }
    return
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %op = transform.structured.match attributes {tag = "operation"} in %arg0
            : (!transform.any_op) -> !transform.any_op

        %interchanged = transform.structured.interchange %op iterator_interchange = [1, 0, 2]
            : (!transform.any_op) -> !transform.any_op

        transform.yield
    }
}
