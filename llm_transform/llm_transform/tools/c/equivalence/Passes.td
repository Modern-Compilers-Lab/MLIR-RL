#ifndef LLM_TRANSFORM_EQUIVALENCE_PASSES
#define LLM_TRANSFORM_EQUIVALENCE_PASSES

include "mlir/Pass/PassBase.td"

def ArrayDataflowEquivalence
    : Pass<"check-array-dataflow-equivalence", "::mlir::ModuleOp"> {
  let summary =
      "Verify semantic equivalence of two affine functions using stable "
      "memref array-dataflow analysis.";
  let description = [{
    Compares an `original` and a `transformed` `func::FuncOp` (looked up by
    name, or, when only two functions are present in the module, by walk
    order) and proves that every original read-after-write, write-after-read,
    or write-after-write dependence over function-argument memrefs is
    preserved by the transformed function. Local allocations are ignored;
    only accesses whose memref ultimately aliases a function `BlockArgument`
    are considered. The check is formulated as a Presburger emptiness query
    over the composed access relations and the lexicographic schedule built
    from surrounding `affine.for` ops. The pass emits an error and fails
    when no transformed counterpart preserves an original dependency.
  }];
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect",
  ];
  let options = [
    Option<"originalFuncName", "original-func", "std::string",
           /*default=*/"\"original\"",
           "Name of the original affine function.">,
    Option<"transformedFuncName", "transformed-func", "std::string",
           /*default=*/"\"transformed\"",
           "Name of the transformed affine function.">,
    Option<"verbose", "verbose", "bool", /*default=*/"false",
           "Emit verifier trace information to stdout.">,
  ];
}

def TagLinalgOpsForEquivalence
    : Pass<"tag-linalg-ops-for-equivalence", "::mlir::func::FuncOp"> {
  let summary =
      "Tag each linalg op with a unique equivalence id carried in a NameLoc.";
  let description = [{
    Walks the function in pre-order and rewrites every `linalg` op's location
    to a `NameLoc` named `eq_id_<n>`, wrapping (and thus preserving) the op's
    original location as the child location. The counter restarts at zero for
    each function, so the same linalg op appearing in an `original` and a
    `transformed` function receives the same `eq_id_<n>` tag as long as both
    are walked in the same order.

    This tag survives bufferization and the linalg-to-affine lowering, where it
    is consumed by `check-array-dataflow-equivalence` to match an original
    memory access against its transformed counterpart. Run this pass before
    lowering, on the still-`linalg` form of both functions.
  }];
  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::linalg::LinalgDialect",
  ];
}

def RaiseSCFToAffine
    : Pass<"raise-scf-to-affine", "::mlir::func::FuncOp"> {
  let summary =
      "Raise `scf.for` loops with affine-expressible bounds to `affine.for`.";
  let description = [{
    Rewrites every `scf.for` whose lower/upper bounds are affine-expressible
    (constants or valid affine symbols) and whose step is a positive constant
    into an `affine.for`, leaving loops that carry iteration arguments alone.

    Tiling via the transform dialect emits `scf.for` tile loops around the
    still-`linalg` body. If those remain `scf.for`, the inner `affine.for`
    produced by `convert-linalg-to-affine-loops` ends up with a bound that
    depends on an SCF induction variable, which is not a legal affine quantity
    and fails verification. Running this pass first makes the whole loop nest
    affine so `check-array-dataflow-equivalence` can reason about the tiled
    schedule. Pair it with `scf-forall-to-for` to also cover parallel tiling.
  }];
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::func::FuncDialect",
    "::mlir::scf::SCFDialect",
  ];
}

#endif // LLM_TRANSFORM_EQUIVALENCE_PASSES
