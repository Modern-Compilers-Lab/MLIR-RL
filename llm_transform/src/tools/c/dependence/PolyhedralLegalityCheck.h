#ifndef LLM_TRANSFORM_POLYHEDRAL_LEGALITY_CHECK_H
#define LLM_TRANSFORM_POLYHEDRAL_LEGALITY_CHECK_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

// Pulls in `PolyhedralLegalityCheckBase<T>` (in namespace `impl`) generated
// by mlir-tblgen from `PolyhedralLegalityCheck.td`.
#define GEN_PASS_DECL
#include "PolyhedralLegalityCheck.h.inc"

#endif // LLM_TRANSFORM_POLYHEDRAL_LEGALITY_CHECK_H
