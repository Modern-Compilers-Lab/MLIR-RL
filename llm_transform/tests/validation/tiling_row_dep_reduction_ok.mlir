// Reduction-dimension analogue of tiling_row_dep_ok.mlir: a legal tiling with
// a real loop-carried dependence. The 2D parallel nest gains an innermost
// "reduction" dimension d2: each output element accumulates `red[d2]` plus the
// (loop-carried) input read, summed over d2. The spatial dependence on the
// (d0, d1) sub-nest — and therefore the legality verdict — is unchanged.
//
// Iteration (d0, d1, d2) reads base[d0, d1] (broadcast over d2) and writes
// base[d0+1, d1] (accumulated over d2). The value read at (d0, d1) was written
// by iteration (d0-1, d1), so the direction vector on the parallel (d0, d1)
// dimensions is (1, 0).
//
// Both the original order and 2D [T, T] tiling of the parallel dims visit
// (d0-1, d1) strictly before (d0, d1), so that dependence is preserved. Tiling
// the reduction dim sequentially (tile size 2) keeps the accumulation order, so
// it is bit-for-bit preserved as well. The transform is legal and the baseline
// / tiled outputs must agree.
//
// The 96x96 parallel extent is a multiple of the 32x32 tile size, and the
// length-4 reduction is a multiple of 2, so tiling produces only full tiles.

func.func @tiling_row_dep_reduction_ok(%base: memref<100x100xf64>, %red: memref<4xf64>) {
    %in_sub = memref.subview %base[0, 0] [96, 96] [1, 1]
        : memref<100x100xf64> to memref<96x96xf64, strided<[100, 1], offset: 0>>

    %out_sub = memref.subview %base[1, 0] [96, 96] [1, 1]
        : memref<100x100xf64> to memref<96x96xf64, strided<[100, 1], offset: 100>>

    linalg.generic {tag = "operation",
        indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d1)>,
            affine_map<(d0, d1, d2) -> (d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%in_sub, %red : memref<96x96xf64, strided<[100, 1], offset: 0>>, memref<4xf64>)
      outs(%out_sub : memref<96x96xf64, strided<[100, 1], offset: 100>>) {
    ^bb0(%in: f64, %r: f64, %acc: f64):
        %cst = arith.constant 1.0 : f64
        %add = arith.addf %in, %cst : f64
        %t = arith.addf %add, %r : f64
        linalg.yield %t : f64
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
