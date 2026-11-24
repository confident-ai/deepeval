from abc import ABC, abstractmethod
from typing import Any, Optional, Union
from enum import Enum
import json
import re

import traceback

from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics.utils import initialize_model
from deepteam.guardrails.types import GuardType
from deepteam.guardrails.guards.schema import SafetyLevelSchema


class SafetyLevel(Enum):
    SAFE = "safe"
    BORDERLINE = "borderline"
    UNSAFE = "unsafe"


class BaseGuard(ABC):
    score: Optional[float] = None
    reason: Optional[str] = None
    model: Optional[DeepEvalBaseLLM] = None
    error: Optional[str] = None
    latency: Optional[float] = None
    safety_level: Optional[SafetyLevel] = None
    guard_type: GuardType
    evaluation_cost: Optional[float] = None

    def __init__(self, model: Union[str, DeepEvalBaseLLM] = "gpt-4.1"):
        self.model, self.using_native_model = initialize_model(model)
        self.evaluation_model = self.model.get_model_name()

    @property
    def __name__(self):
        return "Base Guard"

    def _guard(self, guard_prompt: str, **kwargs) -> Optional[str]:
        self.evaluation_cost = 0 if self.using_native_model else None
        if self.using_native_model:
            res, cost = self.model.generate(
                prompt=guard_prompt, schema=SafetyLevelSchema
            )
            self.evaluation_cost += cost
            self.safety_level = res.safety_level
            self.reason = res.reason
            self.score = 1.0 if res.safety_level == SafetyLevel.SAFE else 0.0
        else:
            try:
                res: SafetyLevelSchema = self.model.generate(
                    prompt=guard_prompt, schema=SafetyLevelSchema
                )
                self.safety_level = res.safety_level
                self.reason = res.reason
                self.score = (
                    1.0 if res.safety_level == SafetyLevel.SAFE else 0.0
                )
            except TypeError:
                res = self.model.generate(guard_prompt)
                data = trim_and_load_json(res, self)
                self.safety_level = data.get("safety_level")
                self.reason = data.get("reason")
                self.score = (
                    1.0 if self.safety_level == SafetyLevel.SAFE else 0.0
                )

        return self.safety_level

    async def a_guard(self, guard_prompt: str, **kwargs) -> Optional[str]:
        self.evaluation_cost = 0 if self.using_native_model else None
        if self.using_native_model:
            res, cost = await self.model.a_generate(
                prompt=guard_prompt, schema=SafetyLevelSchema
            )
            self.evaluation_cost += cost
            self.safety_level = res.safety_level
            self.reason = res.reason
            self.score = 1.0 if res.safety_level == SafetyLevel.SAFE else 0.0
        else:
            try:
                res: SafetyLevelSchema = await self.model.a_generate(
                    prompt=guard_prompt, schema=SafetyLevelSchema
                )
                self.safety_level = res.safety_level
                self.reason = res.reason
                self.score = (
                    1.0 if res.safety_level == SafetyLevel.SAFE else 0.0
                )
            except TypeError:
                res = await self.model.a_generate(guard_prompt)
                data = trim_and_load_json(res, self)
                self.safety_level = data.get("safety_level")
                self.reason = data.get("reason")
                self.score = (
                    1.0 if self.safety_level == SafetyLevel.SAFE else 0.0
                )

        return self.safety_level

    @abstractmethod
    def guard_input(
        self, input: str, guard_prompt: str, *args, **kwargs
    ) -> str:
        pass

    @abstractmethod
    def guard_output(
        self, input: str, output: str, guard_prompt: str, *args, **kwargs
    ) -> str:
        pass

    @abstractmethod
    async def a_guard_input(
        self, input: str, guard_prompt: str, *args, **kwargs
    ) -> str:
        pass

    @abstractmethod
    async def a_guard_output(
        self, input: str, output: str, guard_prompt: str, *args, **kwargs
    ) -> str:
        pass


def trim_and_load_json(
    input_string: str,
    guard: Optional[BaseGuard] = None,
) -> Any:
    start = input_string.find("{")
    end = input_string.rfind("}") + 1

    if end == 0 and start != -1:
        input_string = input_string + "}"
        end = len(input_string)

    jsonStr = input_string[start:end] if start != -1 and end != 0 else ""
    # Remove trailing comma if one is present
    jsonStr = re.sub(r",\s*([\]}])", r"\1", jsonStr)

    try:
        return json.loads(jsonStr)
    except json.JSONDecodeError:
        error_str = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
        if guard is not None:
            guard.error = error_str
        raise ValueError(error_str)
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")
