from typing import List
from enum import Enum
from pydantic import BaseModel

class KernelType(str, Enum):
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

class OptimizationIntent(BaseModel):
    name: str
    description: str
    rationale: str
    priority: Priority
    transformations: List[Transformation]
    
class ActionEnumeration(BaseModel):
    intents: List[OptimizationIntent]
