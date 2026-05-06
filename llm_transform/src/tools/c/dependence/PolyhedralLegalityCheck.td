#ifndef POLYHEDRAL_LEGALITY_CHECK_PASSES
#define POLYHEDRAL_LEGALITY_CHECK_PASSES

include "mlir/Pass/PassBase.td"

def PolyhedralLegalityCheck
    : Pass<"check-polyhedral-legality", "func::FuncOp"> {
  let summary =
      "Verify scf.for/scf.forall loop nest legality via affine dependence checks.";
  let description = [{
    Walks each `scf.for` / `scf.forall` loop nest in a function, collects
    accesses implementing `AffineReadOpInterface` / `AffineWriteOpInterface`
    (e.g. `affine.load` / `affine.store`), and runs polyhedral dependence
    analysis on every load/store pair on the same memref where at least one
    side is a write. A dependence whose direction vector is lexicographically
    negative in the current schedule indicates a legality violation (a
    consumer scheduled before its producer); the pass emits an error and
    fails. Plain `memref.load` / `memref.store` ops are not analyzed; raise
    them with `affine-raise-from-memref` first.
  }];
  let dependentDialects = [
    "::mlir::affine::AffineDialect",
    "::mlir::memref::MemRefDialect",
  ];
  let options = [
    Option<"verbose", "verbose", "bool", /*default=*/"false",
           "Emit per-step trace of access collection, pair-wise dependence "
           "checks, and legality decision to stdout.">,
  ];
}

#endif // POLYHEDRAL_LEGALITY_CHECK_PASSES
