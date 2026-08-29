// transform.loop.peel: splits a loop into a main loop (trip count divisible by
// step) and a remainder loop, BOTH running in the original index order.
//
// We drive peel on a loop that carries a REAL dependence to try to break it.
// The col-dep kernel reads base[d0, d1+1] and writes base[d0+1, d1]. We tile
// ONLY the outer row dimension d0 with a tile size that does not divide the
// extent (extent 98, tile 32 -> 3 full tiles + remainder of 2 rows), then peel
// the row loop. Peeling produces [rows 0..95] then [rows 96..97]; rows still
// execute in strictly increasing order, so the d0-carried dependence direction
// is preserved and the result is identical to the baseline.
//
// Expected: outputs MATCH -> level 0. Peeling cannot reverse a dependence.

func.func @main(%base: memref<100x100xf64>) {
    %in_sub = memref.subview %base[0, 1] [98, 98] [1, 1]
        : memref<100x100xf64> to memref<98x98xf64, strided<[100, 1], offset: 1>>
    %out_sub = memref.subview %base[1, 0] [98, 98] [1, 1]
        : memref<100x100xf64> to memref<98x98xf64, strided<[100, 1], offset: 100>>

    linalg.generic {tag = "operation",
        indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%in_sub : memref<98x98xf64, strided<[100, 1], offset: 1>>)
      outs(%out_sub : memref<98x98xf64, strided<[100, 1], offset: 100>>) {
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
        // Tile only the outer (row) dimension -> one scf.for over d0, bound 98.
        %tiled_op, %loop = transform.structured.tile_using_for %op tile_sizes [32, 0]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %forloop = transform.cast %loop : !transform.any_op to !transform.op<"scf.for">
        %main, %remainder = transform.loop.peel %forloop
            : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
        transform.yield
    }
}
