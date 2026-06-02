// `transform.structured.fuse` breaker: the transform cannot express, and
// therefore refuses, the one IR form in which a loop-carried dependence lives.
//
// This is the kernel of tiling_col_dep.mlir / interchange_col_dep.mlir: a
// linalg.generic on overlapping (aliasing) subviews of one `base` memref, which
// reads base[d0, d1+1] and writes base[d0+1, d1] — a genuine loop-carried
// dependence with direction vector (1, -1). With a tile-loop interchange [1, 0]
// such a tiling WOULD reverse that dependence and miscompile (exactly what
// tiling_col_dep demonstrates for `tile_using_for`).
//
// But `transform.structured.fuse` is tile-AND-fuse over value-semantic
// (tensor / destination-passing) structured ops: it builds the tiled loop nest
// from the op's tensor results and recomputes producer slices via def-use. A
// buffer-semantic (memref) linalg op has no tensor result to tile-and-fuse
// against, so the transform fails to apply (return code 1). `fuse` thus never
// operates on the only IR shape that can carry such a dependence — which is why
// it is classified safe-by-construction: it cannot produce the col_dep
// miscompile because it cannot run on this op at all.
//
// The harness records the failed transform as outputs that differ and the MLIR
// detector as having flagged it.
//
// Expected ground truth: outputs DIFFER (transform rejected; no transformed
// kernel is produced).
// Expected detectors: MLIR detected (transform application failure).
// Harness verdict: [PASS].

func.func @main(%base: memref<100x100xf64>) {
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

        // Tile-and-fuse with the dependence-reversing interchange. Fails to
        // apply: fuse requires tensor (value) semantics, not memref buffers.
        %fused, %loops:2 = transform.structured.fuse %op [32, 32] interchange [1, 0]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}
