from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code

class Name(ActionBase):
    """
    ...
    """

    unique_execution: bool = True

    @classmethod
    def parameters(cls) -> dict:
        """
        ...
        """
        pass

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        ...
        """
        pass

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        ...
        """
        pass

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        ...
        """
        pass

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        ...
        """
        pass
