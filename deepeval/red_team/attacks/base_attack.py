from typing import Optional
from abc import ABC, abstractmethod
from enum import Enum


class Exploitability(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class BaseAttack(ABC):
    weight: int = 1
    multi_turn: bool = False
    name: str
    description: Optional[str] = None
    exploitability: Exploitability

    def enhance(self, attack: str, *args, **kwargs) -> str:
        pass

    async def a_enhance(self, attack: str, *args, **kwargs) -> str:
        return self.enhance(attack, *args, **kwargs)

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError
