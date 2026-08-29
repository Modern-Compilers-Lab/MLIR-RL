// Reduction-dimension analogue of interchange_col_dep.mlir. The 2D parallel
// nest gains an innermost "reduction" dimension d2: each output element
// accumulates `red[d2]` plus the (loop-carried) input read, summed over d2.
// The interchange swaps only the two parallel dims (iterator_interchange =
// [1, 0, 2], reduction left innermost), so the spatial dependence and the
// legality verdict are unchanged from the 2D sibling.
//
// Iteration (d0, d1, d2) reads base[d0, d1+1] (broadcast over d2) and writes
// base[d0+1, d1] (accumulated over d2). The value read at (d0, d1) is produced
// by iteration (d0-1, d1+1).
//
// Baseline (i outer, j inner): (d0-1, d1+1) runs before (d0, d1) because all of
// row d0-1 is processed first, so the read picks up the *overwritten* value.
// After interchange to (j outer, i inner): column d1+1 is processed strictly
// after column d1, so (d0-1, d1+1) runs *after* (d0, d1) — the read picks up
// the *original* value. The reduction over d2 is unaffected.

func.func @interchange_col_dep_reduction(%base: memref<100x100xf64>, %red: memref<4xf64>) {
    %in_sub = memref.subview %base[0, 1] [99, 99] [1, 1]
        : memref<100x100xf64> to memref<99x99xf64, strided<[100, 1], offset: 1>>

    %out_sub = memref.subview %base[1, 0] [99, 99] [1, 1]
        : memref<100x100xf64> to memref<99x99xf64, strided<[100, 1], offset: 100>>

    linalg.generic {tag = "operation",
        indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d1)>,
            affine_map<(d0, d1, d2) -> (d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%in_sub, %red : memref<99x99xf64, strided<[100, 1], offset: 1>>, memref<4xf64>)
      outs(%out_sub : memref<99x99xf64, strided<[100, 1], offset: 100>>) {
    ^bb0(%in: f64, %r: f64, %acc: f64):
        %t = arith.addf %in, %r : f64
        %s = arith.addf %acc, %t : f64
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
