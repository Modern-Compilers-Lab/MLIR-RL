from mlir._mlir_libs._mlir.ir import Module  # type: ignore


def move_module(source: Module, destination: Module):
    """Copy all operations from source module to destination module.

    Args:
        source: The source MLIR module.
        destination: The destination MLIR module where operations will be copied.
    """
    for op in destination.body.operations:
        op.erase()
    for op in source.body.operations:
        destination.body.append(op.clone())
