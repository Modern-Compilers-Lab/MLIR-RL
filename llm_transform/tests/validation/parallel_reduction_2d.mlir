// 2D analogue of parallel_reduction.mlir: a row-wise sum. The first loop (d0,
// over rows) is "parallel" — each row writes a distinct out[d0], so rows are
// independent — but the second loop (d1, over columns) is "reduction": every
// column accumulates into the same per-row scalar out[d0].
//
// Tiling the reduction dimension with scf.forall parallelizes it, so the tiles
// concurrently read-modify-write the same out[d0] accumulator. With no combine
// step the partial sums overwrite each other (a data race when threaded, a
// wrong answer even when serialized), so the result is not the full row sum.
// The parallel d0 dimension *would* be safe to tile with forall; the reduction
// d1 is not.
//
// Legal alternatives: tile only the parallel dim (tile_sizes [N, 0]) with
// tile_using_forall, or use transform.structured.tile_reduction_using_forall on
// d1, which allocates per-thread partial sums and emits a final combine.


func.func @parallel_reduction_2d(%in: memref<256x1024xf32>, %out: memref<256xf32>) {
    linalg.generic {tag = "operation",
        indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
    } ins(%in : memref<256x1024xf32>)
      outs(%out : memref<256xf32>) {
    ^bb0(%a: f32, %acc: f32):
        %sum = arith.addf %a, %acc : f32
        linalg.yield %sum : f32
    }
    return
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %op = transform.structured.match attributes {tag = "operation"} in %arg0
            : (!transform.any_op) -> !transform.any_op

        // tile_sizes [0, 32]: leave the parallel d0 untiled, tile the reduction
        // d1 with scf.forall — illegally parallelizing the accumulation.
        %tiled_op, %forall_loop = transform.structured.tile_using_forall %op tile_sizes [0, 32]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        transform.yield
    }
}
