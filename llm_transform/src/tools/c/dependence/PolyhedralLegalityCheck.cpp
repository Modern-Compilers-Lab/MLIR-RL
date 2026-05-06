#include "PolyhedralLegalityCheck.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
#define GEN_PASS_DEF_POLYHEDRALLEGALITYCHECK
#include "PolyhedralLegalityCheck.h.inc"

struct PolyhedralLegalityCheckPass
    : public impl::PolyhedralLegalityCheckBase<PolyhedralLegalityCheckPass> {
  using Base = impl::PolyhedralLegalityCheckBase<PolyhedralLegalityCheckPass>;
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    if (verbose)
      llvm::outs() << "[check-polyhedral-legality] enter function '"
                   << func.getSymName() << "'\n";

    // Collect every affine read/write in the function once. Per-loop
    // grouping isn't needed: checkMemrefAccessDependence accounts for
    // surrounding loops, and pairs in disjoint nests resolve trivially.
    SmallVector<Operation *> loadAndStores;
    func.walk([&](Operation *op) {
      if (isa<affine::AffineReadOpInterface, affine::AffineWriteOpInterface>(op))
        loadAndStores.push_back(op);
    });

    if (verbose)
      llvm::outs() << "[check-polyhedral-legality]   collected "
                   << loadAndStores.size() << " affine access(es)\n";

    for (unsigned i = 0, e = loadAndStores.size(); i < e; ++i) {
      Operation *srcOp = loadAndStores[i];
      affine::MemRefAccess srcAccess(srcOp);
      for (unsigned j = 0; j < e; ++j) {
        if (i == j) continue;

        Operation *dstOp = loadAndStores[j];
        affine::MemRefAccess dstAccess(dstOp);

        // A dependence must be tested at every common surrounding loop
        // depth from 1 up to numCommon+1: the analysis returns the
        // components carried at that specific depth, and a deeper loop
        // can carry a violation that shallower depths do not surface.
        unsigned numCommon =
            affine::getNumCommonSurroundingLoops(*srcOp, *dstOp);

        if (verbose) {
          llvm::outs() << "[check-polyhedral-legality]   pair (" << i << ","
                       << j << "): src=" << srcOp->getName()
                       << " dst=" << dstOp->getName()
                       << " numCommon=" << numCommon << "\n";
        }

        for (unsigned d = 1; d <= numCommon + 1; ++d) {
          SmallVector<affine::DependenceComponent, 2> depComps;
          affine::DependenceResult result =
              affine::checkMemrefAccessDependence(
                  srcAccess, dstAccess, /*loopDepth=*/d,
                  /*dependenceConstraints=*/nullptr, &depComps);

          if (!affine::hasDependence(result)) {
            if (verbose)
              llvm::outs() << "[check-polyhedral-legality]     depth " << d
                           << ": no dependence\n";
            continue;
          }
          bool lexNeg = isLexicographicallyNegative(depComps);
          if (verbose) {
            llvm::outs() << "[check-polyhedral-legality]     depth " << d
                         << ": dependence, components=[";
            for (unsigned k = 0; k < depComps.size(); ++k) {
              if (k) llvm::outs() << ", ";
              llvm::outs() << "(";
              if (depComps[k].lb.has_value())
                llvm::outs() << *depComps[k].lb;
              else
                llvm::outs() << "-inf";
              llvm::outs() << ",";
              if (depComps[k].ub.has_value())
                llvm::outs() << *depComps[k].ub;
              else
                llvm::outs() << "+inf";
              llvm::outs() << ")";
            }
            llvm::outs() << "] lex-negative=" << (lexNeg ? "yes" : "no")
                         << "\n";
          }
          if (!lexNeg) continue;

          InFlightDiagnostic diag = srcOp->emitError(
              "polyhedral legality violation: dependence direction is "
              "lexicographically negative at loop depth ");
          diag << d << " (a consumer is scheduled before its producer)";
          diag.attachNote(dstOp->getLoc()) << "conflicting access";
          if (verbose)
            llvm::outs() << "[check-polyhedral-legality] result: ILLEGAL\n";
          signalPassFailure();
          return;
        }
      }
    }

    if (verbose)
      llvm::outs() << "[check-polyhedral-legality] result: legal\n";
  }

private:
  // Returns true if the dependence direction vector is, or could be,
  // lexicographically negative under an all-zero prefix. Strict policy:
  // any level that is not definitely non-negative (lb >= 0 with both
  // bounds known) is treated as a violation, since a legality check
  // must reject schedules it cannot prove safe.
  //
  // Per AffineAnalysis.h, each component is an inclusive interval
  // [lb, ub] with lb <= ub; nullopt means unbounded.
  static bool isLexicographicallyNegative(
      const SmallVectorImpl<affine::DependenceComponent> &depComps) {
    for (const auto &dir : depComps) {
      // Definitely zero → continue
      if (dir.lb.has_value() && dir.ub.has_value() &&
          dir.lb.value() == 0 && dir.ub.value() == 0)
        continue;

      // Definitely negative → violation
      if (dir.ub.has_value() && dir.ub.value() < 0)
        return true;

      // Definitely positive → safe
      if (dir.lb.has_value() && dir.lb.value() > 0) return false;

      // Otherwise uncertain → cannot prove violation
      return false;
    }
    return false;
  }
};

// Self-register so the pass name resolves in PassManager.parse(...) as soon
// as this .so is dlopen'd (e.g. via ctypes from the Python test harness).
static PassRegistration<PolyhedralLegalityCheckPass> passRegistration;
} // namespace
