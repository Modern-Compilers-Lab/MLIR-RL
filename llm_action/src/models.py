from typing import List
from enum import Enum
from pydantic import BaseModel

class Priority(Enum, str):
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
