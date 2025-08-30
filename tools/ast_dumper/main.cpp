#include <iostream>
#include <cstdio>
#include <cstring>
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
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
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
  if (tag && isa<StringAttr>(tag)) {
      auto tagAttr = cast<StringAttr>(tag);
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
  llvm::SmallVector<linalg::LinalgOp> ops_list;
  ClonedTarget->walk([&](Operation *op){
    linalg::LinalgOp linalgOp = dyn_cast<linalg::LinalgOp>(op);

    // If it is not a LinalgOp, skip
    if (!linalgOp) return;

    // If iteration space is zero, skip
    if (linalgOp.getNumLoops() == 0) {
      return;
    }

    std::string tagName;
    if (!linalgOp->hasAttr("tag")) {
      tagName = "operation_" + std::to_string(i);
      mlir::Attribute strAttr = mlir::StringAttr::get(&context, tagName);
      linalgOp->setAttr("tag", strAttr);
    } else {
      tagName = getLinalgOpTag(linalgOp);
    }

    ops_list.push_back(linalgOp);

    llvm::outs() << "#START_OPERATION" << "\n";
    llvm::outs() << linalgOp->getName() << "\n";

    llvm::outs() << "#START_VECTORIZABLE" << "\n";
    if (failed(linalg::vectorizeOpPrecondition(linalgOp))) {
      llvm::outs() << "false" << "\n";
    } else {
      llvm::outs() << "true" << "\n";
    }

    llvm::outs() << "#START_NESTED_LOOPS" << "\n";
    llvm::SmallVector<int64_t> loop_ranges = linalgOp.getStaticLoopRanges();
    llvm::SmallVector<utils::IteratorType> iterator_types = linalgOp.getIteratorTypesArray();
    for (auto [index, loop_range, iterator_type] : llvm::enumerate(loop_ranges, iterator_types)){
      llvm::outs() << "d" << index << " " << 0 << " " << loop_range << " " << 1 << " " << iterator_type << "\n";
    }
    llvm::outs() << "#START_LOAD_DATA" << "\n";
    llvm::SmallVector<BlockArgument> used_operands;
    bool found_use;
    for (BlockArgument arg : linalgOp.getBlock()->getArguments()) {
      found_use = false;
      linalgOp.getBlock()->walk([&](Operation *nested_op){
        if (found_use) return;
        for (OpOperand &operand : nested_op->getOpOperands()) {
          if (operand.get() == arg) {
            used_operands.push_back(arg);
            found_use = true;
            break;
          }
        }
      });
    }
    for (BlockArgument used_operand : used_operands) {
      AffineMap operand_map = linalgOp.getMatchingIndexingMap(linalgOp.getMatchingOpOperand(used_operand));
      uint results_nbr = operand_map.getNumResults();
      for (auto [index, map_result] : llvm::enumerate(operand_map.getResults())) {
        map_result.print(llvm::outs());
        if (index < results_nbr - 1) {
          llvm::outs() << ", ";
        } else {
          llvm::outs() << "\n";
        }
      }
    }
    llvm::outs() << "#START_STORE_DATA" << "\n";
    for (BlockArgument out_arg : linalgOp.getRegionOutputArgs()) {
      AffineMap operand_map = linalgOp.getMatchingIndexingMap(linalgOp.getMatchingOpOperand(out_arg));
      uint results_nbr = operand_map.getNumResults();
      for (auto [index, map_result] : llvm::enumerate(operand_map.getResults())) {
        map_result.print(llvm::outs());
        if (index < results_nbr - 1) {
          llvm::outs() << ", ";
        } else {
          llvm::outs() << "\n";
        }
      }
    }
    llvm::outs() << "#START_OP_COUNT" << "\n";
    int add_count = 0, sub_count = 0, mul_count = 0, div_count = 0, exp_count = 0;
    linalgOp.walk([&](Operation *nested_op){
      if (isa<arith::AddFOp>(nested_op)) {
        add_count += 1;
      } else if (isa<arith::SubFOp>(nested_op)) {
        sub_count += 1;
      } else if (isa<arith::MulFOp>(nested_op)) {
        mul_count += 1;
      } else if (isa<arith::DivFOp>(nested_op)) {
        div_count += 1;
      } else if (isa<math::ExpOp>(nested_op)) {
        exp_count += 1;
      }
    });
    llvm::outs() << "+ " << add_count << "\n";
    llvm::outs() << "- " << sub_count << "\n";
    llvm::outs() << "* " << mul_count << "\n";
    llvm::outs() << "/ " << div_count << "\n";
    llvm::outs() << "exp " << exp_count << "\n";
    llvm::outs() << "#START_TAG" << "\n";
    llvm::outs() << tagName << "\n";
    llvm::outs() << "#END_OPERATION" << "\n";
    llvm::outs() << "\n\n\n\n\n" << "\n";

    i += 1;
  });

  llvm::outs() << "\n\n\n\n" << "\n";
  llvm::outs() << "#BEGIN_GRAPH" << "\n";

  for (auto producer_op : ops_list) {
    std::string producerTag = getLinalgOpTag(producer_op);
    for (auto potential_consumer : ops_list) {
      for (auto any_consumer : producer_op->getUsers()) {
        if (!(potential_consumer == any_consumer)) continue;

        std::string consumerTag = getLinalgOpTag(potential_consumer);
        llvm::outs() << producerTag << " --> " << consumerTag << "\n";
      }
    }
  }

  llvm::outs() << "#END_GRAPH\n";

  llvm::outs() << "########################################\n";

  module1->print(llvm::outs());

}

// mkdir tools/ast_dumper/build
// cd tools/ast_dumper/build
// cmake .. -DMLIR_DIR=$LLVM_BUILD_PATH/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_PATH/bin/llvm-lit
// cd ../../..
// cmake --build tools/ast_dumper/build
// tools/ast_dumper/build/bin/AstDumper examples/x1.mlir