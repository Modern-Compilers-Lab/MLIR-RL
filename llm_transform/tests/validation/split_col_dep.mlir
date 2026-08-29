// Attempt to break transform.structured.split with the P-ALIAS col dependence.
//
// Iteration (d0, d1) reads base[d0, d1+1] and writes base[d0+1, d1].
// The value read at (d0, d1) was written by iteration (d0-1, d1+1).
//
// Baseline (i outer, j inner): row d0-1 is fully processed before row d0, so
// (d0-1, d1+1) runs before (d0, d1) -> the read picks up the *overwritten*
// value.
//
// We split along dimension 1 (columns) after 48. The lower op processes
// columns [0,48) for ALL rows, then the upper op processes columns [48,96).
// At the column boundary the producer iteration (d0-1, d1+1=48) now sits in
// the upper op, which runs AFTER the lower op processed (d0, d1=47) -> the
// read in the lower op picks up the *original* value, reversing the dep.
//
// iterator_types are "parallel" so MLIR has no dependence info.

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
        %split = transform.structured.split %op after 48 { dimension = 1 }
            : !transform.any_op
        transform.yield
    }
}
