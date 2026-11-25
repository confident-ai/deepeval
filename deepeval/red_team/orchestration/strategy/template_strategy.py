from typing import Optional

from deepeval.red_team.orchestration.strategy.strategy import (
    Effectiveness,
    Send,
    Strategy,
)


class TemplateStrategy(Strategy):

    LOW_TEMPLATE = "{message}"
    MEDIUM_TEMPLATE = "{message}"
    HIGH_TEMPLATE = "{message}"
    effectiveness = Effectiveness.MEDIUM

    def __init__(self, effectiveness: Optional[Effectiveness] = None):
        if effectiveness is not None:
            self.effectiveness = effectiveness

    def _template(self) -> str:
        return {
            Effectiveness.LOW: self.LOW_TEMPLATE,
            Effectiveness.MEDIUM: self.MEDIUM_TEMPLATE,
            Effectiveness.HIGH: self.HIGH_TEMPLATE,
        }[self.effectiveness]

    async def a_execute(self, message: str, send: Send) -> None:
        await send(self._template().replace("{message}", message))
