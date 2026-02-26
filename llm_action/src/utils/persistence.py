import os
import json
from typing import List, Tuple, Union, Optional

from llm_action.src.utils.misc import random_id
from llm_action.src.models import ClaudeModel, KernelType, ActionEnumeration, ActionPackage, DocTreeNode, Documentation
from llm_action.src.utils.scrape import collect_md_tree, collect_md_doc

from llm_action.src.config import ACTION_ENUMERATION_CACHE, CLAUDE_LLM_MODEL

def load_kernel_code(kernel_type: KernelType, kernel_number: int = 2) -> str:
    dir = f"llm_action/data/{kernel_type.value}"
    match kernel_type:
        case KernelType.MATMUL:
            name = "Matrix Multiplication"
            match kernel_number:
                case 1:
                    code_path = f"{dir}/matmul_128_256_128.mlir"
                case 2:
                    code_path = f"{dir}/matmul_256_512_1024.mlir"
                case 3:
                    code_path = f"{dir}/matmul_512_512_512.mlir"
                case 4:
                    code_path = f"{dir}/matmul_24576_768_384.mlir"
                case _:
                    raise ValueError(f"Unsupported kernel number {kernel_number} for {kernel_type}")
        case KernelType.CONV2D:
            name = "2D Convolution"
            code_path = f"{dir}/conv_2d_nchw_fchw_128_32_7_7_256_1_1_7_7.mlir"
        case KernelType.ATTENTION:
            name = "Attention"
            code_path = f"{dir}/attention.mlir"
        case KernelType.GENERIC:
            name = "Generic"
            code_path = f"{dir}/generic_8_8_16_8_32.mlir"
    with open(code_path, "r") as f:
        code = f.read()
    return code

def load_kernel_code_template(kernel_type: KernelType) -> str:
    dir = f"llm_action/data/{kernel_type.value}"
    match kernel_type:
        case KernelType.MATMUL:
            name = "Matrix Multiplication"
            template_path = f"{dir}/matmul_template.mlir"
        case KernelType.CONV2D:
            name = "2D Convolution"
            template_path = f"{dir}/conv_2d_nchw_fchw_template.mlir"
        case KernelType.ATTENTION:
            name = "Attention"
            template_path = f"{dir}/attention_template.mlir"
        case KernelType.GENERIC:
            name = "Generic"
            template_path = f"{dir}/generic_template.mlir"
    with open(template_path, "r") as f:
        code_template = f.read()
    return f"Kernel Code Template for {name}:\n{code_template}"

def save_action_enumeration_result(reasoning: str, action_enumeration: ActionEnumeration, kernel_type: KernelType = None, model: ClaudeModel = CLAUDE_LLM_MODEL, run_id: str = None) -> str:
    if run_id:
        dir = f"llm_action/results/runs/{run_id}/action_enumeration"
    else:
        id = random_id()
        dir = f"llm_action/results/action_enumeration/{kernel_type.value}/{model.value}/{id}"
    os.makedirs(dir, exist_ok=True)
    json_file_path = f"{dir}/action_enumeration.json"
    with open(json_file_path, "w") as f:
        json.dump(action_enumeration.model_dump(), f, indent=4)
    reasoning_file_path = f"{dir}/reasoning.txt"
    with open(reasoning_file_path, "w") as f:
        f.write(reasoning)
    return dir

def load_cached_action_enumeration() -> ActionEnumeration:
    with open(ACTION_ENUMERATION_CACHE, "r") as f:
        data = json.load(f)
    return ActionEnumeration(**data)

def save_action_implementation_result(reasoning: str, action_package: ActionPackage, action_python_implementation: str, kernel_type: KernelType = None, model: ClaudeModel = CLAUDE_LLM_MODEL, run_id: str = None, save_to_playground: bool = False) -> Tuple[str, Optional[str]]:
    if run_id:
        dir = f"llm_action/results/runs/{run_id}/action_implementation/{action_package.name}"
    else:
        dir = f"llm_action/results/action_implementation/{kernel_type.value}/{model.value}/{action_package.name}_{random_id()}"
    os.makedirs(dir, exist_ok=True)
    txt_file_path = f"{dir}/reasoning.txt"
    with open(txt_file_path, "w") as f:
        f.write(reasoning)
    json_file_path = f"{dir}/action.json"
    with open(json_file_path, "w") as f:
        json.dump(action_package.model_dump(), f, indent=4)
    py_file_path = f"{dir}/action.py"
    with open(py_file_path, "w") as f:
        f.write(action_python_implementation)
    if save_to_playground:
        playground_dir = f"llm_action/playground/actions/candidates/"
        os.makedirs(playground_dir, exist_ok=True)
        playground_py_file_path = f"{playground_dir}/{action_package.name}_{random_id(short=True)}.py"
        with open(playground_py_file_path, "w") as f:
            f.write(action_python_implementation)
    return dir, playground_py_file_path if save_to_playground else None

def save_documentation_lookup_result(task: str, response: str, model: ClaudeModel = CLAUDE_LLM_MODEL, run_id: str = None) -> str:
    if run_id:
        dir = f"llm_action/results/runs/{run_id}/documentation_lookup"
    else:
        id = random_id()
        dir = f"llm_action/results/documentation_lookup/{model.value}/{id}"
    os.makedirs(dir, exist_ok=True)
    lookup_file_path = f"{dir}/lookup.txt"
    with open(lookup_file_path, "w") as f:
        f.write(f"TASK:\n{task}\n\nRESPONSE:\n{response}")
    return lookup_file_path

def save_documentation(tree: DocTreeNode, outdir: str, filename: str = "documentation.json") -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tree.model_dump(), f, ensure_ascii=False, indent=2)
    return path

def save_documentation_doc_md(doc: Documentation, outdir: str, filename: str = "documentation.md") -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(collect_md_doc(doc).rstrip() + "\n")
    return path

def save_documentation_tree_md(tree: DocTreeNode, outdir: str, filename: str = "documentation.md") -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(collect_md_tree(tree).rstrip() + "\n")
    return path

def save_documentation_outline(tree: DocTreeNode, outdir: str, filename: str = "outline.txt") -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)

    lines: List[str] = []
    
    def walk(n: DocTreeNode):
        if n.name != "ROOT":
            indent = "  " * (max(n.level, 1) - 1)
            lines.append(f"{indent}- {n.name}")
        for c in n.children:
            walk(c)

    for c in tree.children:
        walk(c)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    return path

def save_optimization_result(response: str) -> str:
    dir = f"llm_action/results/optimization/"
    os.makedirs(dir, exist_ok=True)
    id = random_id(short=True)
    file_path = f"{dir}/{id}.txt"
    with open(file_path, "w") as f:
        f.write(response)
    return file_path

def save_prompt(prompt: str, version: str, name: str) -> str:
    dir = f"llm_action/resources/prompts/v{version}"
    os.makedirs(dir, exist_ok=True)
    file_path = f"{dir}/{name}.md"
    with open(file_path, "w") as f:
        f.write(prompt)
    return file_path