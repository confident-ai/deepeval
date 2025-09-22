from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Awaitable, Callable, Generic, TypeVar

from deepeval.tracing import observe
from deepeval.prompt import Prompt
from deepeval.tracing.tracing import Observer

try:
    from agents.agent import Agent as BaseAgent
    from agents.models.interface import Model, ModelProvider
except Exception as e:
    raise RuntimeError(
        "openai-agents is required for this integration. Please install it."
    ) from e

TContext = TypeVar("TContext")

class _ObservedModel(Model):
    def __init__(
        self,
        inner: Model,
    ) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _get_model_name(self) -> str:
        try:
            for attr in ("model", "model_name", "name"):
                if hasattr(self._inner, attr):
                    val = getattr(self._inner, attr)
                    if val is not None:
                        return str(val)
        except Exception:
            pass
        return "unknown"

    async def get_response(
        self,
        system_instructions,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing,
        *,
        previous_response_id,
        conversation_id,
        prompt,
        **kwargs,
    ):
        model_name = self._get_model_name()
        with Observer(
            span_type="llm",
            func_name="LLM",
            function_kwargs={
                "system_instructions": system_instructions, 
                "input": input, 
                "model_settings": model_settings, 
                "tools": tools, 
                "output_schema": output_schema, 
                "handoffs": handoffs, 
                # "tracing": tracing, # not important for llm spans
                # "previous_response_id": previous_response_id, # not important for llm spans
                # "conversation_id": conversation_id, # not important for llm spans
                "prompt": prompt,
                **kwargs,
            },
            observe_kwargs={"model": model_name},

        ) as observer:
            result = await self._inner.get_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=prompt,
                **kwargs,
            )

            observer.result = result.output
        
        return result

    def stream_response(
        self,
        system_instructions,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing,
        *,
        previous_response_id,
        conversation_id,
        prompt,
        **kwargs,
    ):
        model_name = self._get_model_name()

        with Observer(
            type="llm",
            model=model_name,
        ) as observer:
            
            pass

class _ObservedProvider(ModelProvider):
    def __init__(
        self,
        base: ModelProvider,
    ) -> None:
        self._base = base

    def get_model(self, model_name: str | None, **kwargs: Any) -> Model:
        model = self._base.get_model(model_name, **kwargs)
        return _ObservedModel(model)


@dataclass
class DeepEvalAgent(BaseAgent[TContext], Generic[TContext]):
    """
    A subclass of agents.Agent.
    """

    def __post_init__(self):
        super().__post_init__()
        if self.model is not None and not isinstance(self.model, str):
            try:
                from agents.models.interface import (
                    Model as _Model,
                ) 

                if isinstance(self.model, _Model):
                    self.model = _ObservedModel(self.model)
            except Exception:
                pass
