from __future__ import annotations

import logging
from typing import List, Optional

from deepeval.confident.api import get_confident_api_key
from deepeval.metrics.base_metric import BaseMetric
from deepeval.prompt import Prompt
from deepeval.telemetry import capture_tracing_integration

logger = logging.getLogger(__name__)


def _require_google_adk_instrumentor():
    try:
        from openinference.instrumentation.google_adk import (
            GoogleADKInstrumentor,
        )

        return GoogleADKInstrumentor
    except ImportError as exc:
        raise ImportError(
            "openinference-instrumentation-google-adk is not installed. "
            "Install it with: "
            "`pip install google-adk openinference-instrumentation-google-adk`."
        ) from exc


def instrument_google_adk(
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
    """Instrument Google ADK agents and ship traces to Confident AI.

    Wraps the community-maintained ``openinference-instrumentation-google-adk``
    package: every ADK agent, model call, and tool invocation emits an OTel
    span tagged with OpenInference semantic conventions, which deepeval's
    OpenInference span interceptor translates into ``confident.span.*``
    attributes before exporting via OTLP.

    Pair with ``@observe`` / ``with trace(...)`` to mix native deepeval spans
    with ADK-emitted OTel spans on the same trace.
    """

    with capture_tracing_integration("google_adk"):
        if not api_key:
            api_key = get_confident_api_key()
            if not api_key:
                raise ValueError(
                    "CONFIDENT_API_KEY is not set. "
                    "Pass it directly or set the environment variable."
                )

        GoogleADKInstrumentor = _require_google_adk_instrumentor()
        GoogleADKInstrumentor().instrument()

        from deepeval.integrations.openinference import (
            instrument_openinference,
        )

        instrument_openinference(
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

        logger.info(
            "Confident AI Google ADK telemetry attached (env=%s, test_mode=%s).",
            environment,
            is_test_mode,
        )
