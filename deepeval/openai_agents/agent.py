from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Awaitable, Callable, Generic, TypeVar, List

from deepeval.tracing import observe
from deepeval.prompt import Prompt
from deepeval.tracing.tracing import Observer
from deepeval.metrics import BaseMetric
from deepeval.tracing.utils import make_json_serializable
from deepeval.tracing.types import LlmSpan
from deepeval.tracing.context import current_span_context

try:
    from agents.agent import Agent as BaseAgent
    from agents.models.interface import Model, ModelProvider
    from openai.types.responses import ResponseCompletedEvent
except Exception as e:
    raise RuntimeError(
        "openai-agents is required for this integration. Please install it."
    ) from e

TContext = TypeVar("TContext")


class _ObservedModel(Model):
    def __init__(
        self,
        inner: Model,
        llm_metric_collection: str = None,
        llm_metrics: List[BaseMetric] = None,
        confident_prompt: Prompt = None,
    ) -> None:
        self._inner = inner
        self._llm_metric_collection = llm_metric_collection
        self._llm_metrics = llm_metrics
        self._confident_prompt = confident_prompt

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
            metrics=self._llm_metrics,
            metric_collection=self._llm_metric_collection,
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
            llm_span: LlmSpan = current_span_context.get()
            llm_span.prompt = self._confident_prompt

            observer.result = make_json_serializable(result.output)

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

        async def _gen():
            observer = Observer(
                span_type="llm",
                func_name="LLM",
                function_kwargs={
                    "system_instructions": system_instructions,
                    "input": input,
                    "model_settings": model_settings,
                    "tools": tools,
                    "output_schema": output_schema,
                    "handoffs": handoffs,
                    # "tracing": tracing,
                    # "previous_response_id": previous_response_id,
                    # "conversation_id": conversation_id,
                    "prompt": prompt,
                    **kwargs,
                },
                observe_kwargs={"model": model_name},
                metrics=self._llm_metrics,
                metric_collection=self._llm_metric_collection,
            )
            observer.__enter__()

            llm_span: LlmSpan = current_span_context.get()
            llm_span.prompt = self._confident_prompt

            try:
                async for event in self._inner.stream_response(
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
                ):

                    if isinstance(event, ResponseCompletedEvent):
                        observer.result = make_json_serializable(
                            event.response.output
                        )

                    yield event

            except Exception as e:
                observer.__exit__(type(e), e, e.__traceback__)
                raise
            finally:

                observer.__exit__(None, None, None)

        return _gen()


@dataclass
class DeepEvalAgent(BaseAgent[TContext], Generic[TContext]):
    """
    A subclass of agents.Agent.
    """

    llm_metric_collection: str = None
    llm_metrics: List[BaseMetric] = None
    confident_prompt: Prompt = None

    def __post_init__(self):
        super().__post_init__()
