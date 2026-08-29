#include "Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

using namespace mlir;

namespace {
#define GEN_PASS_DEF_RAISESCFTOAFFINE
#include "Passes.h.inc"

/// Builds an affine bound (map + operands) for an scf loop bound `value`.
///
/// `affine.for` bounds must be affine: a constant becomes a 0-dim/0-symbol
/// constant map, while any other value must be a valid affine symbol and is
/// threaded through as a single symbol operand. Returns failure when the bound
/// is neither, in which case the loop is left as `scf.for`.
static LogicalResult makeAffineBound(Value value, AffineMap &map,
                                     SmallVectorImpl<Value> &operands,
                                     scf::ForOp forOp, MLIRContext *ctx) {
  if (std::optional<int64_t> constant = getConstantIntValue(value)) {
    map = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0,
                         getAffineConstantExpr(*constant, ctx));
    return success();
  }
  if (affine::isValidSymbol(value)) {
    map = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/1,
                         getAffineSymbolExpr(0, ctx));
    operands.push_back(value);
    return success();
  }
  emitWarning(value.getLoc(), "loop bound is not affine-expressible")
    .attachNote(forOp.getLoc()) << "in scf.for loop";
  return failure();
}

/// Rewrites a single `scf.for` with affine-expressible bounds and a constant
/// positive step into an `affine.for`, moving its body across and turning the
/// induction variable into a valid affine dimension.
static LogicalResult raiseForOp(scf::ForOp forOp, MLIRContext *ctx) {
  std::optional<int64_t> step = getConstantIntValue(forOp.getStep());
  if ( !step.has_value() || step.value() <= 0)
    return failure();

  AffineMap lbMap, ubMap;
  SmallVector<Value> lbOperands, ubOperands;
  if (failed(makeAffineBound(forOp.getLowerBound(), lbMap, lbOperands, forOp, ctx)) ||
      failed(makeAffineBound(forOp.getUpperBound(), ubMap, ubOperands, forOp, ctx)))
    return failure();

  Value forIndVar = forOp.getInductionVar();
  ValueRange forInitArgs = forOp.getInitArgs();
  ValueRange forIterArgs = forOp.getRegionIterArgs();

  OpBuilder builder(forOp);
  auto affineFor = builder.create<affine::AffineForOp>(
    forOp.getLoc(), lbOperands, lbMap, ubOperands, ubMap, step.value(), forInitArgs,
    [&](OpBuilder &nestedBuilder, Location nestedLoc, Value indVar, ValueRange iterArgs) {
      // Map block args
      IRMapping mapping;
      mapping.map(forIndVar, indVar);
      for (auto [forIterArg, affineIterArg] : llvm::zip(forIterArgs, iterArgs))
        mapping.map(forIterArg, affineIterArg);
      for (auto &op : forOp.getOps()) {
        if (isa<scf::SCFDialect>(op.getDialect())) {
          if (isa<scf::YieldOp>(op)) {
            scf::YieldOp mappedYield = cast<scf::YieldOp>(op.clone(mapping));
            nestedBuilder.create<affine::AffineYieldOp>(nestedLoc, mappedYield.getResults());
          } else {
            emitWarning(op.getLoc()) << "Unsupported op in scf.for body";
            return;
          }
        } else {
          nestedBuilder.clone(op, mapping);
        }
      }
    }
  );
  forOp.erase();
  return success();
}

/// Raises `scf.for` loops to `affine.for` so their induction variables become
/// valid affine dimensions. Tiling via the transform dialect emits `scf.for`
/// tile loops (with constant bounds) wrapped around the still-`linalg` body;
/// without this raise, `convert-linalg-to-affine-loops` would produce an inner
/// `affine.for` whose bound depends on an SCF induction variable, which is not
/// a legal affine quantity. Running this first makes the whole nest affine so
/// the array-dataflow EquivalenceVerifier can reason about the tiled schedule.
struct RaiseSCFToAffinePass
    : public impl::RaiseSCFToAffineBase<RaiseSCFToAffinePass> {
  using Base = impl::RaiseSCFToAffineBase<RaiseSCFToAffinePass>;
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();

    SmallVector<scf::ForOp> loops;
    func.walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

    // Process innermost/last loops first so a parent's body is stable (its
    // children already raised) by the time the parent is rewritten.
    for (scf::ForOp forOp : loops)
      (void)raiseForOp(forOp, ctx);
  }
};

static PassRegistration<RaiseSCFToAffinePass> passRegistration;

} // namespace

extern "C" ::mlir::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "RaiseSCFToAffine", "0.1", []() {}};
}
