#include <iostream>
#include <cstdio>
#include <string>
// Include MLIR-related headers
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

// Include LLVM and other necessary headers
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

// Include MLIR passes and transformations
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

// Include custom headers
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include <optional>
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"


using namespace mlir;

std::string getLinalgOpTag(linalg::LinalgOp op) {
  // Get the 'tag' attribute from the operation

  auto tag = op->getAttr("tag");
  if (tag && isa<StringAttr>(tag)) {
      auto tagAttr = cast<StringAttr>(tag);
      std::string tagValue = tagAttr.getValue().str();
      return tagValue;
  } else {
      return "";
  }

}

int main(int argc, char **argv)
{
  std::string inputFilename(argv[1]);
  std::string tagName(argv[2]);
  std::string vectorSizesStr;
  if (argc > 3)
  {
    vectorSizesStr = argv[3];
  }
  // Parse the vector sizes from the string
  llvm::SmallVector<int64_t> vectorSizes;
  if (!vectorSizesStr.empty())
  {
    if (vectorSizesStr.front() != '[' || vectorSizesStr.back() != ']')
    {
      llvm::errs() << "Vector sizes must be enclosed in square brackets, e.g. [4,8,16]\n";
      return 1;
    }
    // Remove the square brackets
    vectorSizesStr = vectorSizesStr.substr(1, vectorSizesStr.size() - 2);
    // Split the string by commas and convert each size to an integer
    std::istringstream ss(vectorSizesStr);
    std::string size;
    while (std::getline(ss, size, ','))
    {
      try {
        vectorSizes.push_back(std::stoll(size));
      } catch (const std::invalid_argument &e) {
        llvm::errs() << "Invalid vector size: " << size << "\n";
        return 1;
      } catch (const std::out_of_range &e) {
        llvm::errs() << "Vector size out of range: " << size << "\n";
        return 1;
      }
    }
  }

  // Register MLIR command-line options
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  // Create an MLIR context
  mlir::MLIRContext context;

  // Create a dialect registry and register necessary dialects
  DialectRegistry registry;
  registerAllDialects(registry);
  mlir::registerAllToLLVMIRTranslations(registry);
  registry.insert<affine::AffineDialect, scf::SCFDialect,
                  linalg::LinalgDialect,
                  arith::ArithDialect,
                  func::FuncDialect,
                  memref::MemRefDialect,
                  transform::TransformDialect,
                  bufferization::BufferizationDialect,
                  tensor::TensorDialect,
                  vector::VectorDialect,
                  shape::ShapeDialect>();

  // Append the dialect registry to the MLIR context
  context.appendDialectRegistry(registry);
  context.loadDialect<scf::SCFDialect>();
  context.loadDialect<vector::VectorDialect>();
  context.loadDialect<transform::TransformDialect>();



  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();

  // mlir::OpAsmPrinter printer;
  // printer.printOperation(linalgOp); std::cout << "\n";

  // The input is '.mlir'.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code ec = fileOrErr.getError())
  {
      llvm::errs() << "Could not open input file: " << ec.message() << "\n";
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  mlir::OwningOpRef<Operation *> module1 = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  Operation *ClonedTarget = module1.get();

  // Get linalg operation with specified tag
  Operation *targetOp;
  bool opFound = false;
  ClonedTarget->walk([&](Operation *op){
    if (linalg::LinalgOp linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      if (linalgOp->hasAttr("tag")) {
        std::string opTagName = getLinalgOpTag(linalgOp);
        if(opTagName == tagName)
        {
          targetOp = op;
          opFound = true;
        }
      }
    }
  });

  if(opFound)
  {
    // Get the parent operation of the target operation
    // mlir::Operation *OpVectParent = targetOp->getParentOp();
    // Vectorize operation
    IRRewriter rewriter(&context);
    llvm::SmallVector<bool> scalableVecDims(vectorSizes.size(), false);
    mlir::LogicalResult vectorized = mlir::linalg::vectorize(
      rewriter,
      targetOp,
      vectorSizes,
      scalableVecDims
    );

    RewritePatternSet patterns(&context);

    // Add vectorization canonicalization and lowering patterns to the set
    // mlir::transform::detail::VectorizeOpGenericAdaptorBase::Properties props;

    // if (!props.getDisableTransferPermutationMapLoweringPatterns())
    mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);

    // if (!props.getDisableMultiReductionToContractPatterns())
    vector::populateVectorReductionToContractPatterns(patterns);

    vector::populateSinkVectorBroadcastPatterns(patterns);

    // Add additional vectorization patterns for specific operations
    patterns.add<linalg::LinalgCopyVTRForwardingPattern,
                linalg::LinalgCopyVTWForwardingPattern>(&context, 2);
    vector::TransferReadOp::getCanonicalizationPatterns(patterns, &context);
    vector::TransferWriteOp::getCanonicalizationPatterns(patterns, &context);
    tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);

    patterns.add<mlir::linalg::CopyVectorizationPattern>(&context);

    // if (props.getVectorizePadding())
    // linalg::populatePadOpVectorizationPatterns(patterns);
    if (vectorized.succeeded() && !failed(applyPatternsAndFoldGreedily(ClonedTarget, std::move(patterns))))
    {
      module1->print(llvm::outs());
    }
  }
}

// mkdir tools/vectorizer/build
// cd tools/vectorizer/build
// cmake .. -DMLIR_DIR=$LLVM_BUILD_PATH/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_PATH/bin/llvm-lit
// cd ../../..
// cmake --build tools/vectorizer/build
// tools/vectorizer/build/bin/AstDumper example.mlir operation_1