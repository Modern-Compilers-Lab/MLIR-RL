import sys
from mlir._mlir_libs._mlir.ir import Context, Module  # type: ignore
from utils import bufferize, lower


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            code = f.read()
    else:
        code = sys.stdin.read()

    with Context():
        module = Module.parse(code)

    bufferize(module)
    lower(module)

    print(module)


if __name__ == "__main__":
    main()
