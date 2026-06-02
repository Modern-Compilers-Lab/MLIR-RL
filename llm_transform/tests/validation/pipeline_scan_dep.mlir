// XFAIL: level-3 — illegal but undetectable (outputs differ; MLIR silent; the
// equivalence verifier cannot run on the pipelined loop, so it fails to flag it).
// transform.loop.pipeline breaker.
//
// linalg.generic over two ALIASING subviews of one function-arg memref:
//   in_sub  = base[0..]   (reads base[i])
//   out_sub = base[1..]   (writes base[i+1])
// Iteration i reads base[i] and writes base[i+1]; iteration i+1 reads
// base[i+1], which iteration i just wrote -> a serial cross-iteration RAW
// dependence satisfied only by the sequential loop order.
//
// Schedule: tile_using_for (-> scf.for) + vectorize (-> vector.transfer in
// the loop) + transform.loop.pipeline. Pipelining prefetches the
// transfer_read of a *following* iteration ahead of the current
// iteration's transfer_write, so the prefetched read observes the STALE
// original value -> wrong output.
//
// The reordering lives in the scf.for iteration schedule of vector
// transfers; the EquivalenceVerifier does not model it -> level 3.
//
// Empirical signals (this file):
//   Outputs: differ   MLIR: silent   (--tool mlir)
// The array-dataflow EquivalenceVerifier does NOT flag it -- it SEGFAULTS
// (RC=139) on the pipelined loop (prologue/epilogue + iter_args confound
// its analysis), so it surfaces as "equivalence verifier failed" rather
// than a verdict. Verified that the SAME kernel with tile+vectorize but no
// pipeline runs cleanly through the verifier, so pipeline is the trigger.
// Either way both active detectors fail to catch wrong output -> level 3.

func.func @main(%base: memref<130xf64>) {
    %in_sub = memref.subview %base[0] [128] [1]
        : memref<130xf64> to memref<128xf64, strided<[1], offset: 0>>
    %out_sub = memref.subview %base[1] [128] [1]
        : memref<130xf64> to memref<128xf64, strided<[1], offset: 1>>

    linalg.generic {tag = "operation",
        indexing_maps = [
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
    } ins(%in_sub : memref<128xf64, strided<[1], offset: 0>>)
      outs(%out_sub : memref<128xf64, strided<[1], offset: 1>>) {
    ^bb0(%a: f64, %o: f64):
        %cst = arith.constant 1.0 : f64
        %r = arith.addf %a, %cst : f64
        linalg.yield %r : f64
    }
    return
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %op = transform.structured.match attributes {tag = "operation"} in %arg0
            : (!transform.any_op) -> !transform.any_op
        %tiled, %loop = transform.structured.tile_using_for %op tile_sizes [1]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        %gen = transform.structured.match ops{["linalg.generic"]} in %arg0
            : (!transform.any_op) -> !transform.any_op
        transform.structured.vectorize %gen : !transform.any_op
        %forloop = transform.structured.match ops{["scf.for"]} in %arg0
            : (!transform.any_op) -> !transform.op<"scf.for">
        %pipelined = transform.loop.pipeline %forloop {iteration_interval = 1 : i64, read_latency = 5 : i64}
            : (!transform.op<"scf.for">) -> !transform.any_op
        transform.yield
    }
}
