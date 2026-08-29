# Build the equivalence passes, one shared library plugin per pass.
#
# Usage:
#   make            # runs mlir-tblgen, then builds both .so plugins
#   make clean
#
# Builds:
#   build/lib/libEquivalenceVerifier.so  (check-array-dataflow-equivalence)
#   build/lib/libTagLinalgOps.so         (tag-linalg-ops-for-equivalence)
#   build/lib/libRaiseSCFToAffine.so     (raise-scf-to-affine)
#
# Both passes are described in Passes.td; mlir-tblgen emits Passes.h.inc
# (containing the GEN_PASS_* sections) into $(BUILD_DIR). Each .so carries its
# own `mlirGetPassPluginInfo` entry point and self-registers its pass via
# static `PassRegistration` on dlopen, so it can be loaded from Python (ctypes)
# and have its pass name resolve in `PassManager.parse`.

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

TD                = Passes.td
BUILD_DIR         = build
GEN_INC           = $(BUILD_DIR)/include/Passes.h.inc

SRC_VERIFIER      = EquivalenceVerifier.cpp
SRC_TAG           = TagLinalgOps.cpp
SRC_RAISE         = RaiseSCFToAffine.cpp
LIB_VERIFIER      = $(BUILD_DIR)/lib/libEquivalenceVerifier.so
LIB_TAG           = $(BUILD_DIR)/lib/libTagLinalgOps.so
LIB_RAISE         = $(BUILD_DIR)/lib/libRaiseSCFToAffine.so

.PHONY: all clean
all: $(LIB_VERIFIER) $(LIB_TAG) $(LIB_RAISE)

$(GEN_INC): $(TD)
	mkdir -p $(@D)
	$(MLIR_TBLGEN) -gen-pass-decls -name=Equivalence \
	    $(TBLGEN_FLAGS) $(TD) -o $@

$(LIB_VERIFIER): $(SRC_VERIFIER) Passes.h $(GEN_INC)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared $(SRC_VERIFIER) $(LDFLAGS) $(LIBS) -o $@

$(LIB_TAG): $(SRC_TAG) Passes.h $(GEN_INC)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared $(SRC_TAG) $(LDFLAGS) $(LIBS) -o $@

$(LIB_RAISE): $(SRC_RAISE) Passes.h $(GEN_INC)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared $(SRC_RAISE) $(LDFLAGS) $(LIBS) -o $@

clean:
	rm -rf $(BUILD_DIR)/*
