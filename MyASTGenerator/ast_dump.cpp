#include <iostream>
#include <cstdio>
#include <cstring>
// Include MLIR-related headers
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"

// Include LLVM and other necessary headers
#include "llvm/ADT/StringRef.h"
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


using namespace mlir;




std::string getLinalgOpTag(linalg::LinalgOp op) {
  // Get the 'tag' attribute from the operation

  auto tag = op->getAttr("tag");
  if (tag && tag.isa<StringAttr>()) {
      auto tagAttr = tag.cast<StringAttr>();
      std::string tagValue = tagAttr.getValue().str();
      return tagValue;
  } else {
      std::cout << "'tag' attribute not found or is not a StringAttr." << std::endl;
      return "";
  }

}



int main(int argc, char **argv)
{

  llvm::StringRef inputFilename = argv[1];

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


  int i = 0;
  ClonedTarget->walk([&](Operation *op){
    if (linalg::LinalgOp linalgOp = dyn_cast<linalg::LinalgOp>(op)) {

      std::string tagName = "operation_" + std::to_string(i);
      mlir::Attribute strAttr = mlir::StringAttr::get(&context, tagName);
      linalgOp->setAttr("tag", strAttr);

      llvm::outs() << "#START_OPERATION" << "\n";
      // printer << linalgOp; std::cout << "\n";
      llvm::outs() << linalgOp << "\n";

      // linalgOp->print(llvm::outs());
      llvm::outs() << "#START_TAG" << "\n";
      llvm::outs() << tagName << "\n";
      llvm::outs() << "#END_OPERATION" << "\n";
      llvm::outs() << "\n\n\n\n\n" << "\n";

      i += 1;
    }
  });


  // ClonedTarget->walk([&](Operation *op){
  //   if (linalg::LinalgOp linalgOp = dyn_cast<linalg::LinalgOp>(op)) {

  //     std::string tagValue = getLinalgOpTag(linalgOp);

  //     if (!tagValue.empty()) {
  //         std::cout << "Tag attribute value: " << tagValue << std::endl;
  //     }

  //   }
  // });

  llvm::outs() << "\n\n\n\n" << "\n";
  llvm::outs() << "#BEGIN_GRAPH" << "\n";

  // https://github.com/llvm/llvm-project/blob/main/mlir/test/lib/IR/TestPrintDefUse.cpp
  ClonedTarget->walk([&](Operation *op_){
    if (linalg::LinalgOp op = dyn_cast<linalg::LinalgOp>(op_)) {

      std::string opTagValue = getLinalgOpTag(op);

      // Print information about the producer of each of the operands.
      for (mlir::Value operand : op->getOperands()) {
        if (Operation *producer = operand.getDefiningOp()) {
          if (linalg::LinalgOp producerLinalgOp = dyn_cast<linalg::LinalgOp>(producer)){
            std::string producerTagValue = getLinalgOpTag(producerLinalgOp);
            llvm::outs() << producerTagValue << " --> " << opTagValue << "\n";
          }
        }
      }
    }
  });

  llvm::outs() << "#END_GRAPH\n";


  llvm::outs() << "########################################\n";



  // module1->dump();
  module1->print(llvm::outs());

}


// cmake .. -DMLIR_DIR=/scratch/mt5383/llvm-project/build-mlir/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=/scratch/mt5383/llvm-project/build-mlir/bin/llvm-lit
// cmake --build MyASTGenerator/build/ && MyASTGenerator/build/bin/AstDumper examples/x1.mlir