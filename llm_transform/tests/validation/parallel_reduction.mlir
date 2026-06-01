// Parallelizing a reduction: every iteration accumulates into the same
// scalar output, so the loop is "reduction" not "parallel". Tiling the
// reduction dimension with scf.forall turns it into a parallel construct
// that ignores the reduction semantics.
//
// A plain tile_using_forall alone is only *racy* (the bufferized tiles share
// one read-modify-write accumulator, so a sequential lowering still yields the
// correct total and only true parallel execution corrupts it — nondeterministic).
// To make the miscompile DETERMINISTIC, we also fuse the zero-fill into the
// forall: now every tile re-zeroes the shared scalar before accumulating its
// own 32-element slice, so the cross-tile accumulation is destroyed and only
// the last tile's partial sum survives (last-writer-wins). The result is a sum
// over a single 32-element tile instead of all 1024 elements, with the fixed
// harness seed it never matches the baseline total, and the outcome does not
// depend on thread scheduling.
//
// Legal alternative: transform.structured.tile_reduction_using_forall, which
// introduces a per-thread partial sum and a final combine.


func.func @parallel_reduction(%in: tensor<1024xf32>, %out: tensor<f32>) -> tensor<f32> {
    %cst = arith.constant 0.0 : f32
    %init = linalg.fill {tag = "fill"} ins(%cst : f32) outs(%out : tensor<f32>) -> tensor<f32>
    %res = linalg.generic {tag = "operation",
        indexing_maps = [
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> ()>
        ],
        iterator_types = ["reduction"]
    } ins(%in : tensor<1024xf32>)
      outs(%init : tensor<f32>) {
    ^bb0(%a: f32, %acc: f32):
        %sum = arith.addf %a, %acc : f32
        linalg.yield %sum : f32
    } -> tensor<f32>
    return %res : tensor<f32>
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %op = transform.structured.match attributes {tag = "operation"} in %arg0
            : (!transform.any_op) -> !transform.any_op
        %fill = transform.structured.match attributes {tag = "fill"} in %arg0
            : (!transform.any_op) -> !transform.any_op

        // Illegally parallelize the reduction dimension.
        %tiled_op, %forall_loop = transform.structured.tile_using_forall %op tile_sizes [32]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        // Pull the zero-fill inside the forall so each tile re-initializes the
        // shared accumulator -> only the last tile's partial sum survives.
        %fused, %new_loop = transform.structured.fuse_into_containing_op %fill into %forall_loop
            : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

        transform.yield
    }
}
