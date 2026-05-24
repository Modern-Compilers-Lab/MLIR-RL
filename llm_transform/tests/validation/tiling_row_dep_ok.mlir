// Legal tiling with a real loop-carried dependence.
//
// Iteration (d0, d1) reads  base[d0, d1]   and writes base[d0+1, d1].
// The value read at (d0, d1) was written by iteration (d0-1, d1).
//
// The dependence direction vector is (1, 0). Both the original loop
// order and 2D [T, T] tiling visit (d0-1, d1) strictly before (d0, d1):
// within a tile along the column d1/T, the inner ii loop reaches d0-1
// before d0; across tile-rows, (d0-1, d1) is in tile-row (d0-1)/T,
// which runs before tile-row d0/T. So the transform is legal and the
// baseline / tiled outputs must agree.
//
// The 96x96 iteration extent is a multiple of the 32x32 tile size, so
// tiling produces only full tiles (no dynamic partial-tile bounds).

func.func @tiling_row_dep_ok(%base: memref<100x100xf64>) {
    %in_sub = memref.subview %base[0, 0] [96, 96] [1, 1]
        : memref<100x100xf64> to memref<96x96xf64, strided<[100, 1], offset: 0>>

    %out_sub = memref.subview %base[1, 0] [96, 96] [1, 1]
        : memref<100x100xf64> to memref<96x96xf64, strided<[100, 1], offset: 100>>

    linalg.generic {tag = "operation",
        indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%in_sub : memref<96x96xf64, strided<[100, 1], offset: 0>>)
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
