// transform.loop.hoist_loop_invariant_subsets: hoists a loop-invariant
// extract_slice/insert_slice subset pair out of an scf.for, operating on a new
// iter_arg. This is a conservative, semantics-preserving rewrite: it only fires
// when the subset is genuinely loop-invariant, so the result is identical.
//
// Kernel: an scf.for that repeatedly reads a fixed slice of the iter_arg tensor,
// transforms it, and writes it back to the SAME fixed slice each iteration. The
// slice indices do not depend on the induction variable, so the subset is
// loop-invariant and can be hoisted. We then hoist it.
//
// Expected: outputs MATCH -> level 0. Conservative subset hoisting cannot
// change results.

func.func @main(%arg: tensor<16xf64>) -> tensor<16xf64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %cst = arith.constant 1.000000e+00 : f64
    %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%t = %arg) -> (tensor<16xf64>) {
        %s = tensor.extract_slice %t[0] [4] [1] : tensor<16xf64> to tensor<4xf64>
        // Update the slice IN PLACE (outs = the slice itself) so that after
        // hoisting the carried slice value aliases its iter_arg and one-shot
        // bufferization accepts the yield.
        %m = linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]
        } ins(%s : tensor<4xf64>) outs(%s : tensor<4xf64>) {
        ^bb0(%in: f64, %out: f64):
            %a = arith.addf %in, %cst : f64
            linalg.yield %a : f64
        } -> tensor<4xf64>
        %ins = tensor.insert_slice %m into %t[0] [4] [1] : tensor<4xf64> into tensor<16xf64>
        scf.yield %ins : tensor<16xf64>
    }
    return %r : tensor<16xf64>
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %loop = transform.structured.match ops{["scf.for"]} in %arg0
            : (!transform.any_op) -> !transform.any_op
        transform.loop.hoist_loop_invariant_subsets %loop : !transform.any_op
        transform.yield
    }
}
