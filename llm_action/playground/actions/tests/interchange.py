from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code

from llm_action.playground.actions.candidates.InterchangeSonnet import LoopInterchange

params_per_kernel = {
    KernelType.MATMUL: {
        "iterator_interchange": [1, 0, 2],
    },
    KernelType.CONV2D: {
        "iterator_interchange": [1, 0, 2, 3],
    },
    KernelType.GENERIC: {
        "iterator_interchange": [1, 0],
    },
}

if __name__ == "__main__":
    
    ACTION = LoopInterchange

    for kernel_type in [KernelType.MATMUL, KernelType.CONV2D, KernelType.GENERIC]:
        print(f"--- Testing {ACTION.__name__} Action on {kernel_type.value} Kernel ---\n")
        code = load_kernel_code(kernel_type)
        print(f"Original Code:\n{code}\n")

        parameters = params_per_kernel[kernel_type]
        
        print(f"Using Parameters: {parameters}\n")

        if ACTION.precondition(code, parameters):
            transformed_code = ACTION.implement(code, parameters)
            print(f"Transformed Code:\n{transformed_code}\n")
            if ACTION.postcondition(code, transformed_code, parameters):
                print(f"Postcondition satisfied: {ACTION.__name__} applied successfully.")
            else:
                print(f"Postcondition failed: {ACTION.__name__} not applied as expected.")
        else:
            print(f"Precondition not met; {ACTION.__name__} not applied.")
        print("=" * 80 + "\n")
