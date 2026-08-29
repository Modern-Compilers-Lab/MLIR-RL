// transform.loop.fuse_sibling (P-ALIAS, expect detection / miscompile).
//
// The doc says fuse_sibling performs only "rudimentary" legality checks and
// trusts the user that the two loops are independent. We give it two sibling
// scf.for loops over the SAME range/step that are NOT independent: loop2 reads
// elements that loop1 writes in a LATER iteration, via the same function-arg
// buffer.
//
// loop1:  for i in [0,95):  base[i]   = base[i] + 1          (increment elt i)
// loop2:  for i in [0,95):  base[i]   = base[i+1]            (shift-left read)
//
// Baseline (loop1 entirely, THEN loop2): every element is incremented first,
// so loop2 reads the *incremented* base[i+1].
//
// After fuse_sibling (bodies interleaved per i): iteration i runs
//   base[i] = base[i] + 1;  base[i] = base[i+1];
// At iteration i, loop2 reads base[i+1], but loop1 increments base[i+1] only at
// iteration i+1, which has not run yet -> loop2 reads the *un-incremented*
// value. The cross-loop read-after-write dependence is reversed -> wrong result.
//
// Both loops alias one function-arg memref, so after lowering the conflict is
// affine.load/store on a function arg -> EquivalenceVerifier can see it.

func.func @main(%base: memref<100xf64>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c95 = arith.constant 95 : index
    %one = arith.constant 1.000000e+00 : f64

    // loop1: increment each element in [0, 95).
    scf.for %i = %c0 to %c95 step %c1 {
        %v = memref.load %base[%i] : memref<100xf64>
        %inc = arith.addf %v, %one : f64
        memref.store %inc, %base[%i] : memref<100xf64>
    }

    // loop2: shift-left — base[i] = base[i+1].
    scf.for %i = %c0 to %c95 step %c1 {
        %ip1 = arith.addi %i, %c1 : index
        %v = memref.load %base[%ip1] : memref<100xf64>
        memref.store %v, %base[%i] : memref<100xf64>
    }
    return
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %loops = transform.structured.match ops{["scf.for"]} in %arg0
            : (!transform.any_op) -> !transform.any_op
        %l:2 = transform.split_handle %loops
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        // Fuse loop2 (target) into loop1 (source); bodies interleave per i.
        %fused = transform.loop.fuse_sibling %l#1 into %l#0
            : (!transform.any_op, !transform.any_op) -> !transform.any_op
        transform.yield
    }
}
