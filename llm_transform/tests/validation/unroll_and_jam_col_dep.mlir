// transform.loop.unroll_and_jam (P-ALIAS, expect detection).
//
// Hand-written affine.for nest carrying a real loop-carried dependence on a
// single function-arg buffer. Iteration (i, j) reads base[i, j+1] and writes
// base[i+1, j]. The value read at (i, j) was written by iteration (i-1, j+1).
//
// Baseline (i outer, j inner): all of row i-1 is processed before row i starts,
// so (i-1, j+1) runs before (i, j) -> the read at (i,j) picks up the
// *overwritten* value.
//
// unroll_and_jam of the OUTER (i) loop with factor 2 fuses two adjacent outer
// iterations i and i+1 into one inner pass: for each j it runs (i,j) then
// (i+1,j) before advancing to j+1. Now the store of (i-1, j+1) (in the jammed
// group of i-2/i-1... actually now interleaved) happens *after* the read of
// (i, j) -> read picks up the *original* value. The carried dependence is
// reversed -> wrong result.
//
// We use affine.for / affine.load / affine.store so the verifier's affine
// pipeline consumes the IR without an SCF->affine raise. unroll_and_jam keeps
// the indices affine, so the EquivalenceVerifier can analyze the conflict.

func.func @main(%base: memref<100x100xf64>) {
    affine.for %i = 0 to 96 {
        affine.for %j = 0 to 96 {
            %v = affine.load %base[%i, %j + 1] : memref<100x100xf64>
            %c1 = arith.constant 1.0 : f64
            %s = arith.addf %v, %c1 : f64
            affine.store %s, %base[%i + 1, %j] : memref<100x100xf64>
        }
    }
    return
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %outer = transform.structured.match ops{["affine.for"]} in %arg0
            : (!transform.any_op) -> !transform.any_op
        // The match returns all affine.for ops; split to the outer one.
        %o:2 = transform.split_handle %outer
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        // %o#1 is the OUTER (i) loop; jamming it interleaves rows i and i+1
        // within each inner-j step, reversing the row-carried dependence.
        transform.loop.unroll_and_jam %o#1 {factor = 2} : !transform.any_op
        transform.yield
    }
}
