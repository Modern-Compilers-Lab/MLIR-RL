import os
import json

from llm_action.src.utils.misc import random_id
from llm_action.src.models import KernelType, ActionEnumeration

def load_kernel_code_template(kernel_type: KernelType) -> str:
    dir = f"llm_action/data/{kernel_type.value}"
    match kernel_type:
        case KernelType.MATMUL:
            template_path = f"{dir}/matmul_template.mlir"
        case KernelType.CONV2D:
            template_path = f"{dir}/conv_2d_nchw_fchw_template.mlir"
        case KernelType.ATTENTION:
            template_path = f"{dir}/attention_template.mlir"
        case KernelType.GENERIC:
            template_path = f"{dir}/generic_template.mlir"
    with open(template_path, "r") as f:
        code_template = f.read()
    return code_template

def save_action_enumeration_result(result: ActionEnumeration, kernel_type: KernelType) -> str:
    dir = f"llm_action/results/action_enumeration/{kernel_type.value}"
    os.makedirs(dir, exist_ok=True)
    file_path = f"{dir}/{random_id()}.json"
    with open(file_path, "w") as f:
        json.dump(result.model_dump(), f, indent=4)
    return file_path
