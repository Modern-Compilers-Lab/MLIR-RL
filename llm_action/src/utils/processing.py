from rl_autoschedular.state import extract_bench_features_from_code

def preprocess_code(code: str) -> str:
    """Preprocess the given MLIR code to extract benchmark features.

    Args:
        code (str): The MLIR code as a string.

    Returns:
        str: The preprocessed MLIR code.
    """
    # Extract benchmark features (this may include transformations)
    bench_features = extract_bench_features_from_code("", code, 0)
    
    # Return the (possibly modified) code
    return bench_features.code
