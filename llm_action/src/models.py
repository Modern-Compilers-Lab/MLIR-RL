from typing import List, Optional, Union
from enum import Enum
from pydantic import BaseModel, ConfigDict

class ClaudeModel(str, Enum):
    HAIKU = "claude-haiku-4-5" # fastest model with near-frontier intelligence (1$/M-input, 5$/M-output)
    SONNET = "claude-sonnet-4-5" # smart model for complex agents and coding (3$/M-input, 15$/M-output)
    OPUS = "claude-opus-4-5" # premium model combining maximum intelligence with practical performance (5$/M-input, 25$/M-output)
    
class GroqModel(str, Enum):
    GPT_OSS_120B = "openai/gpt-oss-120b" # OpenAI open source model (120b)
    KIMI_K2 = "moonshotai/kimi-k2-instruct-0905" # KIMI
    LLAMA_8b = "llama-3.1-8b-instant" # LLaMA 3.1 8B Instant

class GeminiModel(str, Enum):
    GEMINI_2_5_FLASH = "gemini-2.5-flash" # Google's Gemini 2.5 Flash model, optimized for speed and efficiency.
    GEMINI_2_5_PRO = "gemini-2.5-pro" # Google's Gemini 2.5 Pro model, designed for high performance and advanced capabilities.

class InputType(str, Enum):
    TENSOR = "tensor"
    MEMREF = "memref"

class KernelType(str, Enum):
    MIXED = "mixed"
    MATMUL = "matmul"
    CONV2D = "conv2d"
    ATTENTION = "attention"
    GENERIC = "generic"

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Transformation(BaseModel):
    name: str
    description: str
    rationale: str
    action_template: str

class OptimizationIntent(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    name: str
    description: str
    rationale: str
    priority: Priority
    transformations: List[Transformation]
    
class ActionEnumeration(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    intents: List[OptimizationIntent]

class Parameter(BaseModel):
    name: str
    description: str
    type: str
    values: Optional[List[Union[str, int, float]]] = None

class ActionPackage(BaseModel):
    name: str
    description: str
    parameters: List[Parameter]

class DocTreeNode(BaseModel):
    name: str
    level: int
    content: str
    children: List["DocTreeNode"] = []

class TransformationDocumentation(BaseModel):
    name: str
    label: str
    content: str
    
class TransformationCategory(BaseModel):
    name: str
    transformations: List[TransformationDocumentation]

class Documentation(BaseModel):
    transformation_categories: List[TransformationCategory]
