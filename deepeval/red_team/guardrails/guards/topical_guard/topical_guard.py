from typing import List, Optional, Union

from deepeval.models import DeepEvalBaseLLM
from deepteam.guardrails.guards.base_guard import BaseGuard
from deepteam.guardrails.guards.topical_guard.template import (
    TopicalGuardTemplate,
)
from deepteam.guardrails.types import GuardType


class TopicalGuard(BaseGuard):
    def __init__(
        self,
        allowed_topics: List[str] = None,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
    ):
        super().__init__(model=model)
        self.guard_type = GuardType.INPUT
        self.allowed_topics = allowed_topics or []

    @property
    def __name__(self):
        return "Topical Guard"

    def guard_input(self, input: str) -> str:
        self.guard_type = GuardType.INPUT
        guard_prompt = self._get_guard_prompt(input=input)
        return self._guard(guard_prompt=guard_prompt)

    def guard_output(self, input: str, output: str) -> str:
        self.guard_type = GuardType.OUTPUT
        guard_prompt = self._get_guard_prompt(input=input, output=output)
        return self._guard(guard_prompt=guard_prompt)

    async def a_guard_input(self, input: str) -> str:
        self.guard_type = GuardType.INPUT
        guard_prompt = self._get_guard_prompt(input=input)
        return await self.a_guard(guard_prompt=guard_prompt)

    async def a_guard_output(self, input: str, output: str) -> str:
        self.guard_type = GuardType.OUTPUT
        guard_prompt = self._get_guard_prompt(input=input, output=output)
        return await self.a_guard(guard_prompt=guard_prompt)

    def _get_guard_prompt(
        self, input: str, output: Optional[str] = None
    ) -> str:
        if self.guard_type == GuardType.INPUT:
            return TopicalGuardTemplate.judge_input_prompt(
                input_text=input, allowed_topics=self.allowed_topics
            )
        else:
            return TopicalGuardTemplate.judge_output_prompt(
                input_text=input,
                output_text=output,
                allowed_topics=self.allowed_topics,
            )
