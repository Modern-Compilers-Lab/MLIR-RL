#include <iostream>
#include <string>
#include <typeinfo>

// Core MLIR and LLVM Support
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"

// Individual Dialects for specific operations
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"


// Helper function to get a 'tag' attribute from a LinalgOp
std::string getLinalgOpTag(mlir::linalg::LinalgOp op) {
  auto tagAttr = op->getAttrOfType<mlir::StringAttr>("tag");
  if (tagAttr) {
    return tagAttr.str();
  }
  // It's better to let the caller handle the "not found" case
  return "";
}

int main(int argc, char **argv) {
  if (argc < 3) {
    llvm::errs() << "Usage: AstDumper <input.mlir> <tag>\n";
    return 1;
  }
  llvm::StringRef inputFilename = argv[1];
  llvm::StringRef opTag = argv[2];

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  mlir::MLIRContext context;
  mlir::Location loc = mlir::UnknownLoc::get(&context);

  // For a parser-like tool, it's best to register all known dialects
  // to be able to parse any valid MLIR file.
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  // registerAllDialects doesn't include Transform, so add it manually
  registry.insert<mlir::transform::TransformDialect>();
  mlir::registerAllToLLVMIRTranslations(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  // Parse the input MLIR file
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 1;
  }

  mlir::linalg::LinalgOp targetOp;
  module->walk([&](mlir::linalg::LinalgOp linalgOp){
    if (opTag != getLinalgOpTag(linalgOp))
      return mlir::WalkResult::advance();

    targetOp = linalgOp;
    return mlir::WalkResult::interrupt();
  });

  if (!targetOp) {
    llvm::errs() << "No LinalgOp with tag '" << opTag << "' found.\n";
    return 1;
  }

  mlir::IRRewriter rewriter(&context);
  rewriter.setInsertionPoint(targetOp);

  // 1. Create extract slice ops
  llvm::SmallDenseMap<int64_t, std::pair<mlir::Value, llvm::SmallVector<mlir::Range>>> outsRanges;
  bool mapsChanged = false;
  auto newMaps = targetOp.getIndexingMapsArray();
  for (auto &operand : targetOp->getOpOperands()) {
    auto map = targetOp.getMatchingIndexingMap(&operand);
    int64_t mapIndex = targetOp.getIndexingMapIndex(&operand);
    if (map.isProjectedPermutation(true)) continue;

    auto opTensor = operand.get();
    auto opTensorT = llvm::dyn_cast<mlir::RankedTensorType>(opTensor.getType());
    if(!opTensorT) {
      llvm::errs() << "Expected RankedTensorType, found ";
      opTensor.getType().print(llvm::errs());
      llvm::errs() << "\n";
      return 1;
    }

    llvm::SmallVector<mlir::Range> ranges;
    mlir::MutableAffineMap mutableMap = map;
    bool shouldTile = false;
    for (auto [dim, expr] : llvm::enumerate(map.getResults())) {
      auto dimExpr = llvm::dyn_cast<mlir::AffineDimExpr>(expr);
      auto constExpr = llvm::dyn_cast<mlir::AffineConstantExpr>(expr);
      if (!dimExpr && !constExpr) {
        shouldTile = false;
        break;
      }
      if (dimExpr || (constExpr.getValue() == 0)) {
        ranges.push_back({
          rewriter.getI64IntegerAttr(0),
          rewriter.getI64IntegerAttr(opTensorT.getDimSize(dim)),
          rewriter.getI64IntegerAttr(1),
        });
        continue;
      }
      assert(constExpr && constExpr.getValue() != 0);
      shouldTile = true;
      ranges.push_back({
        rewriter.getI64IntegerAttr(constExpr.getValue()),
        rewriter.getI64IntegerAttr(1),
        rewriter.getI64IntegerAttr(1),
      });
      mutableMap.setResult(dim, rewriter.getAffineConstantExpr(0));
    }

    if (!shouldTile) continue;
    assert(ranges.size() == size_t(opTensorT.getRank()));

    auto sliceOp = rewriter.create<mlir::tensor::ExtractSliceOp>(loc, opTensor, ranges);
    operand.set(sliceOp.getResult());

    newMaps[mapIndex] = mutableMap.getAffineMap();
    mapsChanged = true;

    if (targetOp.isDpsInit(&operand)) {
      auto outIdx = targetOp.getTiedOpResult(&operand).getResultNumber();
      assert(!outsRanges.contains(outIdx));
      outsRanges[outIdx] = {opTensor, ranges};
    }
  }

  // 2. Create a clone op with the new result types
  auto newTargetOp = mlir::clone(rewriter, targetOp, targetOp.getDpsInits().getTypes(), targetOp->getOperands());

  // 3. Apply changes to indexing maps after converting target to GenericOp
  if (mapsChanged) {
    auto genericOp = llvm::dyn_cast<mlir::linalg::GenericOp>(newTargetOp.getOperation());
    if (!genericOp) {
      auto res = mlir::linalg::generalizeNamedOp(rewriter, newTargetOp);
      if (!mlir::succeeded(res)) {
        llvm::errs() << "Failed to convert operation to Generic\n";
        return 1;
      }
      genericOp = res.value();
    }
    genericOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(newMaps));
    newTargetOp = llvm::cast<mlir::linalg::LinalgOp>(genericOp.getOperation());
  }

  // 4. Create insert slice op to insert back to original tensor
  llvm::SmallVector<mlir::Value> newOuts = newTargetOp->getResults();
  for (auto [resIdx, origins] : outsRanges) {
    auto &[originTensor, ranges] = origins;
    auto outTensor = newOuts[resIdx];
    auto insertOp = rewriter.create<mlir::tensor::InsertSliceOp>(loc, outTensor, originTensor, ranges);
    newOuts[resIdx] = insertOp.getResult();
  }

  // 5. Remove original target
  rewriter.replaceOpUsesWithIf(targetOp, newOuts, [](mlir::OpOperand &){return true;});
  rewriter.eraseOp(targetOp);

  module->print(llvm::outs());

  return 0;
}

// mkdir tools/ast_dumper/build
// cd tools/ast_dumper/build
// cmake .. -DMLIR_DIR=$LLVM_BUILD_PATH/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_PATH/bin/llvm-lit
// cd ../../..
// cmake --build tools/ast_dumper/build
// tools/ast_dumper/build/bin/AstDumper examples/x1.mlir