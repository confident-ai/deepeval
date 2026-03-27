from __future__ import annotations

import logging
from typing import List, Optional

from deepeval.config.settings import get_settings
from deepeval.confident.api import get_confident_api_key
from deepeval.metrics.base_metric import BaseMetric
from deepeval.prompt import Prompt
from deepeval.telemetry import capture_tracing_integration
from deepeval.tracing.otel.exporter import ConfidentSpanExporter

logger = logging.getLogger(__name__)
settings = get_settings()

_base_url = str(settings.CONFIDENT_OTEL_URL).rstrip("/")
OTLP_ENDPOINT = f"{_base_url}/v1/traces"

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    _opentelemetry_installed = True
except ImportError:
    _opentelemetry_installed = False


def _require_opentelemetry() -> None:
    if not _opentelemetry_installed:
        raise ImportError(
            "OpenTelemetry SDK is not available. "
            "Install it with: pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
        )


def instrument_openinference(
    api_key: Optional[str] = None,
    name: Optional[str] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    tags: Optional[List[str]] = None,
    environment: Optional[str] = None,
    metric_collection: Optional[str] = None,
    trace_metric_collection: Optional[str] = None,
    llm_metric_collection: Optional[str] = None,
    agent_metric_collection: Optional[str] = None,
    tool_metric_collection_map: Optional[dict] = None,
    confident_prompt: Optional[Prompt] = None,
    test_case_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    is_test_mode: bool = False,
    agent_metrics: Optional[List[BaseMetric]] = None,
) -> None:
    with capture_tracing_integration("openinference"):
        _require_opentelemetry()

        if not api_key:
            api_key = get_confident_api_key()
            if not api_key:
                raise ValueError(
                    "CONFIDENT_API_KEY is not set. "
                    "Pass it directly or set the environment variable."
                )

        from .instrumentator import (
            OpenInferenceInstrumentationSettings,
            OpenInferenceSpanInterceptor,
        )

        openinference_settings = OpenInferenceInstrumentationSettings(
            api_key=api_key,
            name=name,
            thread_id=thread_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            environment=environment,
            metric_collection=metric_collection,
            trace_metric_collection=trace_metric_collection,
            llm_metric_collection=llm_metric_collection,
            agent_metric_collection=agent_metric_collection,
            tool_metric_collection_map=tool_metric_collection_map,
            confident_prompt=confident_prompt,
            test_case_id=test_case_id,
            turn_id=turn_id,
            is_test_mode=is_test_mode,
            agent_metrics=agent_metrics,
        )

        current_provider = trace.get_tracer_provider()

        # Initialize a real TracerProvider if a dummy one is currently active
        if type(current_provider).__name__ in (
            "ProxyTracerProvider",
            "NoOpTracerProvider",
        ):
            tracer_provider = TracerProvider()
            try:
                trace.set_tracer_provider(tracer_provider)
                logger.debug("Created and registered a new TracerProvider.")
            except Exception as exc:
                logger.warning("Could not set global tracer provider: %s", exc)
            current_provider = trace.get_tracer_provider()

        if not hasattr(current_provider, "add_span_processor"):
            logger.warning(
                "The active TracerProvider (%s) does not support "
                "add_span_processor. OpenInference telemetry cannot be attached.",
                type(current_provider).__name__,
            )
            return

        # 1. Attach our custom OpenInference Interceptor
        span_interceptor = OpenInferenceSpanInterceptor(openinference_settings)
        current_provider.add_span_processor(span_interceptor)

        # 2. Attach the appropriate exporter based on the environment/mode
        if is_test_mode:
            current_provider.add_span_processor(
                SimpleSpanProcessor(ConfidentSpanExporter())
            )
        else:
            current_provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(
                        endpoint=OTLP_ENDPOINT,
                        headers={"x-confident-api-key": api_key},
                    )
                )
            )

        logger.info(
            "Confident AI OpenInference telemetry attached (env=%s, test_mode=%s).",
            openinference_settings.environment,
            is_test_mode,
        )
