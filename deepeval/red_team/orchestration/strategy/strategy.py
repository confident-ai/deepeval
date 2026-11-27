from abc import ABC, abstractmethod
from enum import Enum
from typing import Awaitable, Callable, Optional

from deepeval.red_team.attacks.multi_turn.progression import Probe

Send = Callable[[str], Awaitable[Probe]]
Generate = Callable[[str], Awaitable[str]]


class Effectiveness(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Strategy(ABC):

    name: str
    description: str
    strategy: str
    effectiveness: Effectiveness
    generate: Optional[Generate] = None

    @abstractmethod
    async def a_execute(self, message: str, send: Send) -> None:
        raise NotImplementedError

    def decode(self, reply: str) -> Optional[str]:
        return None

    def get_name(self) -> str:
        return self.name
