# Build the PolyhedralLegalityCheck pass as a shared library plugin.
#
# Usage:
#   make            # runs mlir-tblgen, then builds build/libPolyhedralLegalityCheck.so
#   make clean
#
# The pass is described in PolyhedralLegalityCheck.td; mlir-tblgen emits
# PolyhedralLegalityCheck.h.inc (containing the GEN_PASS_* sections) into
# $(BUILD_DIR). The pass self-registers via static `PassRegistration` on
# dlopen, so the resulting .so can be loaded from Python (ctypes) and have
# its pass name `check-polyhedral-legality` resolve in `PassManager.parse`.

PREFIX      ?= $(CONDA_PREFIX)
ifeq ($(PREFIX),)
$(error CONDA_PREFIX is not set; activate the conda env or pass PREFIX=/path/to/env)
endif

CXX          = clang++
MLIR_TBLGEN  = $(PREFIX)/bin/mlir-tblgen
CXXFLAGS     = -std=c++17 -O2 -fPIC -fno-exceptions -fno-rtti -D_GNU_SOURCE \
               -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS
INCLUDES     = -I$(PREFIX)/include -I. -I$(BUILD_DIR)/include
LDFLAGS      = -L$(PREFIX)/lib -Wl,-rpath,$(PREFIX)/lib
LIBS         = -lMLIR -lLLVM
TBLGEN_FLAGS = -I$(PREFIX)/include

SRC          = PolyhedralLegalityCheck.cpp
TD           = PolyhedralLegalityCheck.td
BUILD_DIR    = build
GEN_INC      = $(BUILD_DIR)/include/PolyhedralLegalityCheck.h.inc
LIB          = $(BUILD_DIR)/lib/libPolyhedralLegalityCheck.so

.PHONY: all clean
all: $(LIB)

$(GEN_INC): $(TD)
	mkdir -p $(@D)
	$(MLIR_TBLGEN) -gen-pass-decls -name=PolyhedralLegalityCheck \
	    $(TBLGEN_FLAGS) $(TD) -o $@

$(LIB): $(SRC) PolyhedralLegalityCheck.h $(GEN_INC)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared $(SRC) $(LDFLAGS) $(LIBS) -o $@

clean:
	rm -rf $(BUILD_DIR)/*
