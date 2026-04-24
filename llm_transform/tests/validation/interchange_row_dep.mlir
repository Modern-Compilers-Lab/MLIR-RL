// Interchange breaker with anti-dependence going the other way.
//
// Iteration (d0, d1) reads  base[d0+1, d1]   and writes base[d0, d1+1].
// The value read at (d0, d1) was written by iteration (d0+1, d1-1).
//
// Baseline (i outer, j inner): (d0+1, d1-1) runs *after* (d0, d1)
// because d0+1 > d0, so the read picks up the *original* value.
// After interchange to (j outer, i inner): column d1-1 is processed
// strictly before column d1, so (d0+1, d1-1) runs *before* (d0, d1) and
// the read picks up the *overwritten* value.

func.func @interchange_row_dep(%base: memref<100x100xf64>) {
    %in_sub = memref.subview %base[1, 0] [99, 99] [1, 1]
        : memref<100x100xf64> to memref<99x99xf64, strided<[100, 1], offset: 100>>

    %out_sub = memref.subview %base[0, 1] [99, 99] [1, 1]
        : memref<100x100xf64> to memref<99x99xf64, strided<[100, 1], offset: 1>>

    linalg.generic {tag = "operation",
        indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%in_sub : memref<99x99xf64, strided<[100, 1], offset: 100>>)
      outs(%out_sub : memref<99x99xf64, strided<[100, 1], offset: 1>>) {
    ^bb0(%in: f64, %out: f64):
        %cst = arith.constant 1.0 : f64
        %add = arith.addf %in, %cst : f64
        linalg.yield %add : f64
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
