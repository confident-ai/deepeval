from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Awaitable, Callable

from deepeval.tracing import observe

try:
    from agents.agent import Agent as BaseAgent
    from agents.models.interface import Model, ModelProvider
except Exception as e:
    raise RuntimeError("openai-agents is required for this integration. Please install it.") from e


class _ObservedModel(Model):
    """
    Wraps a Model to observe get_response (and optionally stream_response) with deepeval.observe.
    """

    def __init__(
        self,
        inner: Model,
        *,
        metrics: Optional[list[Any]] = None,
        metric_collection: Optional[str] = None,
    ) -> None:
        self._inner = inner
        self._metrics = metrics
        self._metric_collection = metric_collection

    # Delegate attributes not overridden
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
    ):
        model_name = self._get_model_name()

        wrapped = observe(
            metrics=self._metrics,
            metric_collection=self._metric_collection,
            type="llm",
            model=model_name,
        )(self._inner.get_response)

        return await wrapped(
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
        )

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
    ):
        # Optional: if you also want to observe streaming, uncomment and wrap similarly.
        # wrapped = observe(
        #     metrics=self._metrics,
        #     metric_collection=self._metric_collection,
        #     type="llm",
        #     model=model_name,
        # )(self._inner.stream_response)
        # return wrapped(
        #     system_instructions,
        #     input,
        #     model_settings,
        #     tools,
        #     output_schema,
        #     handoffs,
        #     tracing,
        #     previous_response_id=previous_response_id,
        #     conversation_id=conversation_id,
        #     prompt=prompt,
        # )
        return self._inner.stream_response(
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
        )


class _ObservedProvider(ModelProvider):
    """
    Wraps a ModelProvider to return observed Models.
    """

    def __init__(
        self,
        base: ModelProvider,
        *,
        metrics: Optional[list[Any]] = None,
        metric_collection: Optional[str] = None,
    ) -> None:
        self._base = base
        self._metrics = metrics
        self._metric_collection = metric_collection

    def get_model(self, model_name: str | None) -> Model:
        model = self._base.get_model(model_name)
        return _ObservedModel(
            model,
            metrics=self._metrics,
            metric_collection=self._metric_collection,
        )


@dataclass
class DeepEvalAgent(BaseAgent[Any]):
    """
    A subclass of agents.Agent that accepts `metrics` and `metric_collection`
    and ensures the underlying model's `get_response` is wrapped with deepeval.observe.
    """

    metrics: list[Any] | None = field(default=None)
    metric_collection: str | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        # If a direct Model instance is set on the agent, wrap it here.
        if self.model is not None and not isinstance(self.model, str):
            try:
                from agents.models.interface import Model as _Model  # local import for safety
                if isinstance(self.model, _Model):
                    self.model = _ObservedModel(
                        self.model,
                        metrics=self.metrics,
                        metric_collection=self.metric_collection,
                    )
            except Exception:
                # If we can't import or wrap, silently skip.
                pass