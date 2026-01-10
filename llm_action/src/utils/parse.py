import json
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

def parse_json(text: str, model: Type[T]) -> T:
    """
    Extract JSON by slicing from the first '{' to the last '}' and
    validate it using a Pydantic v2 model.
    """

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in text")

    try:
        data = json.loads(text[start : end + 1])
        return model(**data)
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON format") from e
    except Exception as e:
        raise ValueError("Data validation error") from e
