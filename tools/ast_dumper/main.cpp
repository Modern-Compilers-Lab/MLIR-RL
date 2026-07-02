#include <iostream>
#include <string>

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


using namespace mlir;


// Helper function to get a 'tag' attribute from a LinalgOp
std::string getLinalgOpTag(mlir::linalg::LinalgOp op) {
  auto tagAttr = op->getAttrOfType<mlir::StringAttr>("tag");
  if (tagAttr) {
    return tagAttr.getValue().str();
  }
  // It's better to let the caller handle the "not found" case
  return "";
}


int main(int argc, char **argv) {
  if (argc < 2) {
    llvm::errs() << "Usage: AstDumper <input.mlir>\n";
    return 1;
  }
  llvm::StringRef inputFilename = argv[1];

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  mlir::MLIRContext context;

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

  llvm::SmallVector<linalg::LinalgOp> ops_list;
  module->walk([&](mlir::linalg::LinalgOp linalgOp){
    // If iteration space is zero, skip
    if (linalgOp.getNumLoops() == 0) {
      return;
    }

    std::string tagName = "operation_" + std::to_string(ops_list.size());
    linalgOp->setAttr("tag", mlir::StringAttr::get(&context, tagName));
    // std::string tagName = getLinalgOpTag(linalgOp);
    // if (tagName.empty()) {
    // }

    ops_list.push_back(linalgOp);

    llvm::outs() << "#START_OPERATION" << "\n";
    llvm::outs() << linalgOp->getName() << "\n";

    llvm::outs() << "#START_VECTORIZABLE" << "\n";
    llvm::outs() << (failed(mlir::linalg::vectorizeOpPrecondition(linalgOp)) ? "false" : "true") << "\n";

    llvm::outs() << "#START_NESTED_LOOPS" << "\n";
    auto loop_ranges = linalgOp.getStaticLoopRanges();
    auto iterator_types = linalgOp.getIteratorTypesArray();
    for (auto [index, loop_range, iterator_type] : llvm::enumerate(loop_ranges, iterator_types)){
      llvm::outs() << "d" << index << " " << 0 << " " << loop_range << " " << 1 << " " << iterator_type << "\n";
    }
    llvm::outs() << "#START_LOAD_DATA" << "\n";
    for (auto in_operand : linalgOp.getDpsInputOperands()) {
      AffineMap operand_map = linalgOp.getMatchingIndexingMap(in_operand);
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
    for (auto &out_val : linalgOp.getDpsInitsMutable()) {
      AffineMap operand_map = linalgOp.getMatchingIndexingMap(&out_val);
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
  });

  llvm::outs() << "\n\n\n\n" << "\n";
  llvm::outs() << "#BEGIN_GRAPH" << "\n";

  for (auto producer_op : ops_list) {
    std::string producerTag = getLinalgOpTag(producer_op);
    for (auto &consumption : producer_op->getUses()) {
      auto consumer_op = dyn_cast<mlir::linalg::LinalgOp>(consumption.getOwner());
      if (!consumer_op || !llvm::is_contained(ops_list, consumer_op)) continue;

      std::string consumerTag = getLinalgOpTag(consumer_op);
      int prod_res_nbr = cast<OpResult>(consumption.get()).getResultNumber();
      int op_order = consumption.getOperandNumber();
      llvm::outs() << producerTag << " " << prod_res_nbr << " --> " << consumerTag << " " << op_order << "\n";
    }
  }

  llvm::outs() << "#END_GRAPH\n";

  llvm::outs() << "########################################\n";

  module->print(llvm::outs());

  return 0;
}

// mkdir tools/ast_dumper/build
// cd tools/ast_dumper/build
// cmake .. -DMLIR_DIR=$LLVM_BUILD_PATH/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_PATH/bin/llvm-lit
// cd ../../..
// cmake --build tools/ast_dumper/build
// tools/ast_dumper/build/bin/AstDumper examples/x1.mlir