// transform.structured.hoist_redundant_vector_broadcasts.
//
// The scf.for contains a loop-invariant vector.broadcast of a scalar that is
// constant across iterations. Hoisting the extract/broadcast pair out of the
// loop is value-preserving (the broadcast result is identical every
// iteration). Expected: outputs MATCH -> level 0.
//
// Empirical note: this op only moves vector.extract/vector.broadcast pairs
// whose source is invariant under the loop; it does not reorder memory and is
// purely a code-motion of a redundant SSA value, so it cannot change results.

func.func @main(%A: memref<64xf64>, %B: memref<64xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %p = ub.poison : f64
    // a loop-invariant scalar broadcast source
    %sv = vector.transfer_read %A[%c0], %p : memref<64xf64>, vector<1xf64>
    %scalar = vector.extract %sv[0] : f64 from vector<1xf64>
    scf.for %i = %c0 to %c64 step %c1 {
        %bcast = vector.broadcast %scalar : f64 to vector<1xf64>
        %bv = vector.transfer_read %B[%i], %p : memref<64xf64>, vector<1xf64>
        %s = arith.addf %bv, %bcast : vector<1xf64>
        vector.transfer_write %s, %B[%i] : vector<1xf64>, memref<64xf64>
    }
    return
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %func = transform.structured.match ops{["func.func"]} in %arg0
            : (!transform.any_op) -> !transform.any_op
        %h = transform.structured.hoist_redundant_vector_broadcasts %func
            : (!transform.any_op) -> !transform.any_op
        transform.yield
    }
}
