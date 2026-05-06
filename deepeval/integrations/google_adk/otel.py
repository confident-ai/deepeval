from __future__ import annotations

import logging
from typing import List, Optional

from deepeval.confident.api import get_confident_api_key
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
    test_case_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    **removed_kwargs,
) -> None:
    """Instrument Google ADK agents and ship traces to Confident AI.

    Wraps the community-maintained ``openinference-instrumentation-google-adk``
    package: every ADK agent, model call, and tool invocation emits an OTel
    span tagged with OpenInference semantic conventions, which deepeval's
    OpenInference span interceptor translates into ``confident.span.*``
    attributes.

    Routing follows the Pydantic AI POC pattern: REST when a deepeval trace
    context is active (``@observe`` / ``with trace(...)``) or
    ``trace_manager.is_evaluating`` is True; OTLP otherwise. Pair with
    ``@observe`` / ``with trace(...)`` to mix native deepeval spans with
    ADK-emitted OTel spans on the same trace.

    All kwargs are optional and trace-level; span-level fields go on
    ``with next_*_span(...)`` / ``update_current_span(...)``.
    """
    if removed_kwargs:
        offending = ", ".join(sorted(removed_kwargs))
        raise TypeError(
            f"instrument_google_adk: unexpected keyword argument(s) "
            f"{offending}. Span-level kwargs were removed in the OTel POC "
            "migration; use ``with next_*_span(...)`` or "
            "``update_current_span(...)``. "
            "See deepeval/integrations/README.md."
        )

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
            test_case_id=test_case_id,
            turn_id=turn_id,
        )

        logger.info("Confident AI Google ADK telemetry attached.")
