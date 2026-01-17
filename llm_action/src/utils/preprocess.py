from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code

from rl_autoschedular.state import extract_bench_features_from_code

def preprocess_code(code: str) -> str:
    features = extract_bench_features_from_code("synthetic", code, 0)
    return features.code

if __name__ == "__main__":
    for kernel in [KernelType.MATMUL, KernelType.CONV2D]:
        print(f"Processing kernel: {kernel.name}")
        code = load_kernel_code(kernel)
        print(f"Original Code:\n{code}")
        preprocessed_code = preprocess_code(code)
        print(f"Preprocessed Code:\n{preprocessed_code}")
        print("========================================")
    
    