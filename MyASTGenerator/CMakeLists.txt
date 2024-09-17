cmake_minimum_required(VERSION 3.13.4)
project(AutoSchedulerML LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# Add the include directories
include_directories(include)



if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  # Build via external projects mechanism
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()


set(STANDALONE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(STANDALONE_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories($ENV{LLVM_INCLUDE})
include_directories($ENV{LLVM_BUILD_INCLUDE})
include_directories($ENV{MLIR_INCLUDE})
include_directories($ENV{MLIR_BUILD_INCLUDE})
include_directories(${STANDALONE_SOURCE_DIR}/include)
include_directories(${STANDALONE_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})


get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(AstDumper 
ast_dump.cpp
)

llvm_update_compile_flags(AstDumper)

target_link_libraries(AstDumper
  PRIVATE
    ${dialect_libs}
  
    MLIRIR
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTargetLLVMIRExport
    
    MLIRAffineDialect
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRTransformDialect
    MLIROptLib
    MLIRSCFToControlFlow
  
    MLIRMathToLLVM
    MLIRMemRefToLLVM
    MLIRLinalgToStandard
    MLIRFuncToLLVM
    MLIROpenMPToLLVM
    MLIRSCFToOpenMP
    MLIRBufferizationToMemRef
  
    MLIRToLLVMIRTranslationRegistration
    MLIRLLVMToLLVMIRTranslation
    MLIRExecutionEngine
    MLIRJitRunner
  
    MLIRAnalysis
    MLIRSupport
  
    )

mlir_check_all_link_libraries(AstDumper)
