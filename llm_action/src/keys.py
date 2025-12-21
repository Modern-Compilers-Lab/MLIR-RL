import os
import dotenv

dotenv.load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

MLIR_SHARED_LIBS = os.getenv("MLIR_SHARED_LIBS")
AST_DUMPER_BIN_PATH = os.getenv("AST_DUMPER_BIN_PATH")
