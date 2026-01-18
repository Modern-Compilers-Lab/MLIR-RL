import os

from dotenv import load_dotenv

load_dotenv()

VERBOSE = False

CONDA_ENV = os.getenv("CONDA_ENV")
if CONDA_ENV is None:
    raise ValueError("CONDA_ENV environment variable is not set.")
elif VERBOSE:
    print(f"Using conda environment: {CONDA_ENV}")

NEPTUNE_PROJECT = os.getenv("NEPTUNE_PROJECT")
if NEPTUNE_PROJECT is None:
    raise ValueError("NEPTUNE_PROJECT environment variable is not set.")
elif VERBOSE:
    print(f"Using Neptune project: {NEPTUNE_PROJECT}")

NEPTUNE_TOKEN = os.getenv("NEPTUNE_TOKEN")
if NEPTUNE_TOKEN is None:
    raise ValueError("NEPTUNE_TOKEN environment variable is not set.")
elif VERBOSE:
    print("Neptune token is set.")

LLVM_BUILD_PATH = os.getenv("LLVM_BUILD_PATH")
if LLVM_BUILD_PATH is None:
    raise ValueError("LLVM_BUILD_PATH environment variable is not set.")
elif VERBOSE:
    print(f"Using LLVM build path: {LLVM_BUILD_PATH}")

MLIR_SHARED_LIBS = os.getenv("MLIR_SHARED_LIBS")
if MLIR_SHARED_LIBS is None:
    raise ValueError("MLIR_SHARED_LIBS environment variable is not set.")
elif VERBOSE:
    print(f"Using MLIR shared libs: {MLIR_SHARED_LIBS}")

AST_DUMPER_BIN_PATH = os.getenv("AST_DUMPER_BIN_PATH")
if AST_DUMPER_BIN_PATH is None:
    raise ValueError("AST_DUMPER_BIN_PATH environment variable is not set.")
elif VERBOSE:
    print(f"Using AST dumper bin path: {AST_DUMPER_BIN_PATH}")

VECTORIZER_BIN_PATH = os.getenv("VECTORIZER_BIN_PATH")
if VECTORIZER_BIN_PATH is None:
    raise ValueError("VECTORIZER_BIN_PATH environment variable is not set.")
elif VERBOSE:
    print(f"Using vectorizer bin path: {VECTORIZER_BIN_PATH}")
    
PRE_VEC_BIN_PATH = os.getenv("PRE_VEC_BIN_PATH")
if PRE_VEC_BIN_PATH is None:
    raise ValueError("PRE_VEC_BIN_PATH environment variable is not set.")
elif VERBOSE:
    print(f"Using pre-vectorizer bin path: {PRE_VEC_BIN_PATH}")

CONFIG_FILE_PATH = os.getenv("CONFIG_FILE_PATH")
if CONFIG_FILE_PATH is None:
    raise ValueError("CONFIG_FILE_PATH environment variable is not set.")
elif VERBOSE:
    print(f"Using config file path: {CONFIG_FILE_PATH}")
