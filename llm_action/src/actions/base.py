from abc import ABC, abstractmethod

class ActionBase(ABC):
    @classmethod
    @abstractmethod
    def parameters(cls) -> dict:
        pass

    @classmethod
    @abstractmethod
    def precondition(cls, code: str, params: dict) -> bool:
        pass

    @classmethod
    @abstractmethod
    def preprocess(cls, code: str, params: dict) -> str:
        pass

    @classmethod
    @abstractmethod
    def implement(cls, code: str, params: dict) -> str:
        pass

    @classmethod
    @abstractmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        pass

    @classmethod
    def params_size(cls) -> int:
        return 0

    @classmethod
    def classes_per_slot(cls, n_loops: int) -> list[int]:
        return []

    @classmethod
    def decode_params(cls, raw_slots: list[int], n_loops: int) -> dict:
        return {}
