// Parallelizing a reduction: every iteration accumulates into the same
// scalar output, so the loop is "reduction" not "parallel". Tiling with
// scf.forall turns the outer tile loop into a parallel construct —
// multiple threads would then read-modify-write the shared accumulator
// concurrently, which is a data race regardless of how the inner tile
// body is scheduled.
//
// Legal alternative: transform.structured.tile_reduction_using_forall,
// which introduces a per-thread partial sum and a final combine. Plain
// tile_using_forall ignores the reduction semantics.


func.func @parallel_reduction(%in: memref<1024xf32>, %out: memref<f32>) {
    linalg.generic {tag = "operation",
        indexing_maps = [
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> ()>
        ],
        iterator_types = ["reduction"]
    } ins(%in : memref<1024xf32>)
      outs(%out : memref<f32>) {
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

        %tiled_op, %forall_loop = transform.structured.tile_using_forall %op tile_sizes [32]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        transform.yield
    }
}
