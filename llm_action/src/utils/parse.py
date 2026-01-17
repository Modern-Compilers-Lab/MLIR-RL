import re
import json
from typing import Type, TypeVar, Tuple
from pydantic import BaseModel
from llm_action.src.models import ActionEnumeration, ActionPackage

T = TypeVar("T", bound=BaseModel)

def parse_json(text: str, model: Type[T]) -> T:
    """
    Extract JSON by slicing from the first '{' to the last '}' and
    validate it using a Pydantic v2 model.
    """

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in text\n{text}")

    try:
        data = json.loads(text[start : end + 1])
        return model(**data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format\n{text}") from e
    except Exception as e:
        raise ValueError(f"Data validation error\n{text}") from e

def extract_json_block(text: str) -> str:
    JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    match = JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"Missing JSON code block\n{text}")
    return match.group(1).strip()

def extract_python_block(text: str) -> str:
    PYTHON_BLOCK_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
    match = PYTHON_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"Missing Python code block\n{text}")
    return match.group(1).strip()

def parse_action_enumeration_output(text: str) -> Tuple[str, ActionEnumeration]:
    """
    Parse Layer-1 agent output.

    Expected format:
    - Reasoning text
    - One ```json``` block (ActionEnumeration)

    Returns:
        reasoning: reasoning text before the JSON block
        action_enumeration: validated ActionEnumeration model
    """

    json_block = extract_json_block(text)
    reasoning = text[: text.find("```json")].strip()
    action_enumeration = parse_json(json_block, ActionEnumeration)

    return reasoning, action_enumeration

def parse_action_implementation_output(
    text: str
) -> Tuple[str, ActionPackage, str]:
    """
    Parse Layer-2 agent output.

    Expected format:
    - One ```json``` block (ActionPackage metadata)
    - One ```python``` block (Action implementation)

    Returns:
        action_package: validated ActionPackage model
        python_source: raw Python source code
    """
    
    reasoning = text[: text.find("```json")].strip()

    json_block = extract_json_block(text)
    if not json_block:
        raise ValueError("JSON block is empty")
    
    python_source = extract_python_block(text)
    if not python_source:
        raise ValueError("Python block is empty")

    action_package = parse_json(json_block, ActionPackage)
    
    return reasoning, action_package, python_source
