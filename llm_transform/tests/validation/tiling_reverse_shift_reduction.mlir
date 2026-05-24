// Reduction-dimension analogue of tiling_reverse_shift.mlir. The 2D parallel
// nest gains an innermost "reduction" dimension d2: each output element now
// accumulates `red[d2]` plus the (loop-carried) input read, summed over d2.
// The spatial loop-carried dependence on the (d0, d1) sub-nest — and therefore
// the tiling legality verdict — is unchanged from the 2D sibling.
//
// Iteration (d0, d1, d2) reads base[d0+1, d1] (broadcast over d2) and writes
// base[d0, d1+1] (accumulated over d2). The value read at (d0, d1) is produced
// by iteration (d0+1, d1-1), so the direction vector on the parallel (d0, d1)
// dimensions is the reverse shift of tiling_col_dep.
//
// Baseline (i outer, j inner): (d0+1, d1-1) runs strictly after (d0, d1), so
// the read picks up the *original* value. After tiling [32, 32] of the parallel
// dims, a read on a tile-column boundary lands in the previous (already
// executed) tile-column and picks up an *overwritten* value. Tiling the
// reduction dim sequentially (tile size 2) is itself legal; the illegality
// comes entirely from reordering the parallel (d0, d1) iterations.
//
// The 96x96 parallel extent is a multiple of the 32x32 tile size, and the
// length-4 reduction is a multiple of 2, so tiling produces only full tiles.

func.func @tiling_reverse_shift_reduction(%base: memref<100x100xf64>, %red: memref<4xf64>) {
    %in_sub = memref.subview %base[1, 0] [96, 96] [1, 1]
        : memref<100x100xf64> to memref<96x96xf64, strided<[100, 1], offset: 100>>

    %out_sub = memref.subview %base[0, 1] [96, 96] [1, 1]
        : memref<100x100xf64> to memref<96x96xf64, strided<[100, 1], offset: 1>>

    linalg.generic {tag = "operation",
        indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d1)>,
            affine_map<(d0, d1, d2) -> (d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%in_sub, %red : memref<96x96xf64, strided<[100, 1], offset: 100>>, memref<4xf64>)
      outs(%out_sub : memref<96x96xf64, strided<[100, 1], offset: 1>>) {
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

        %tiled_op, %loops:3 = transform.structured.tile_using_for %op tile_sizes [32, 32, 2]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}
