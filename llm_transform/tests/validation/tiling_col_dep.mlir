// Tiling breaker: loop-carried dependence with a cross-loop direction
// vector that the original order satisfies but 2D tiling violates.
//
// Iteration (d0, d1) reads  base[d0, d1+1]   and writes base[d0+1, d1].
// The value read at (d0, d1) was written by iteration (d0-1, d1+1).
//
// Baseline (i outer, j inner): row d0-1 is fully processed before row
// d0 starts, so (d0-1, d1+1) runs before (d0, d1) — the read picks up
// the *overwritten* value.
// After 2D tiling [32, 32]: for d0 in the interior of a tile-row and
// d1 on a tile-column boundary (d1 == T-1), (d0-1, d1+1) sits in the
// next tile-column, which has not yet executed — so the read picks up
// the *original* value.
//
// Iterator types are "parallel" so MLIR has no dependence information
// to check and silently performs the illegal tiling.
//
// The 96x96 iteration extent is a multiple of the 32x32 tile size, so
// tiling produces only full tiles (no dynamic partial-tile bounds).

func.func @tiling_col_dep(%base: memref<100x100xf64>) {
    %in_sub = memref.subview %base[0, 1] [96, 96] [1, 1]
        : memref<100x100xf64> to memref<96x96xf64, strided<[100, 1], offset: 1>>

    %out_sub = memref.subview %base[1, 0] [96, 96] [1, 1]
        : memref<100x100xf64> to memref<96x96xf64, strided<[100, 1], offset: 100>>

    linalg.generic {tag = "operation",
        indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%in_sub : memref<96x96xf64, strided<[100, 1], offset: 1>>)
      outs(%out_sub : memref<96x96xf64, strided<[100, 1], offset: 100>>) {
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

        %tiled_op, %loops:2 = transform.structured.tile_using_for %op tile_sizes [32, 32]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}
