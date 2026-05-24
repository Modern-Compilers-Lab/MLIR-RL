#include "Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

#include <string>

using namespace mlir;

namespace {
#define GEN_PASS_DEF_TAGLINALGOPSFOREQUIVALENCE
#include "Passes.h.inc"

/// Stamps every linalg op with a unique `eq_id_<n>` tag, stored as the name of
/// a `NameLoc` that wraps the op's original location. The wrapped location keeps
/// the original debug info intact while exposing a stable tag that
/// `check-array-dataflow-equivalence` can later match against.
///
/// The counter is local to each function invocation of the pass, so identical
/// linalg programs walked in the same order receive identical tags. This is what
/// lets the verifier line up an `original` access with its `transformed`
/// counterpart after the tag has propagated through bufferization and the
/// linalg-to-affine lowering.
struct TagLinalgOpsForEquivalencePass
    : public impl::TagLinalgOpsForEquivalenceBase<
          TagLinalgOpsForEquivalencePass> {
  using Base =
      impl::TagLinalgOpsForEquivalenceBase<TagLinalgOpsForEquivalencePass>;
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();

    int nextId = 0;
    func.walk([&](linalg::LinalgOp linalgOp) {
      std::string tag = "eq_id_" + std::to_string(nextId++);
      StringAttr tagAttr = StringAttr::get(ctx, tag);

      // Wrap the existing location in a NameLoc. This preserves original debug
      // info while adding our tag as the NameLoc name.
      Location existingLoc = linalgOp->getLoc();
      linalgOp->setLoc(NameLoc::get(tagAttr, existingLoc));
    });
  }
};

// Self-registers on dlopen so `tag-linalg-ops-for-equivalence` resolves in
// `PassManager.parse`.
static PassRegistration<TagLinalgOpsForEquivalencePass> passRegistration;

} // namespace

extern "C" ::mlir::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "TagLinalgOpsForEquivalence", "0.1",
          []() {}};
}
