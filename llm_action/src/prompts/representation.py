from llm_action.src.models import KernelType, OptimizationIntent, Transformation

from llm_action.src.utils.persistence import load_kernel_code_template, load_kernel_code

def get_training_code_templates_representation(include_instances: bool = True) -> str:
    representation = f"""RL Training Code Templates:
{load_kernel_code_template(KernelType.MATMUL)}
{load_kernel_code_template(KernelType.CONV2D)}
{load_kernel_code_template(KernelType.GENERIC)}"""

    if include_instances:
        representation += f"""Concrete Instances:
{load_kernel_code(KernelType.MATMUL)}
{load_kernel_code(KernelType.CONV2D)}
{load_kernel_code(KernelType.GENERIC)}"""

    return representation

def get_code_representation(kernel_type: KernelType) -> str:
    return f"""Code for {kernel_type.name}:
{load_kernel_code(kernel_type)}
"""

def get_optimization_intent_representation(optimization_intent: OptimizationIntent) -> str:
    return f"""Optimization Intent:
- name: {optimization_intent.name}
- description: {optimization_intent.description}
- rationale: {optimization_intent.rationale}"""

def get_transformation_representation(transformation: Transformation) -> str:
    return f"""Transformation:
- name: {transformation.name}
- description: {transformation.description}
- rationale: {transformation.rationale}
- action template: {transformation.action_template}
"""