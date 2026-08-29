// transform.memref.make_loop_independent on a memref.alloca whose size
// depends on the enclosing scf.for induction variable.
//
// Each iteration i allocates a temp buffer of size (i+1), fills entry 0 with
// in[i], and writes that back to out[i]. The transform rewrites the dynamic
// alloca into a fixed upper-bound alloca + a memref.subview placed inside the
// loop, which is semantics-preserving by construction. Expected: MATCH.

func.func @main(%in: memref<16xf64>, %out: memref<16xf64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %sz = arith.addi %i, %c1 : index
    %tmp = memref.alloca(%sz) {tag = "operation"} : memref<?xf64>
    %v = memref.load %in[%i] : memref<16xf64>
    memref.store %v, %tmp[%c0] : memref<?xf64>
    %r = memref.load %tmp[%c0] : memref<?xf64>
    memref.store %r, %out[%i] : memref<16xf64>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match attributes {tag = "operation"} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %new = transform.memref.make_loop_independent %op { num_loops = 1 }
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
