"""Shared helpers for the Google ADK integration test suite.

Defines a ``trace_test(schema_name)`` decorator factory that resolves
schema files relative to ``schemas/`` next to this conftest, dispatching
to ``generate_trace_json`` (when ``GENERATE_SCHEMAS=true``) or
``assert_trace_json``. Mirrors the per-file definition in the AgentCore
suite, lifted into conftest so the four test modules don't duplicate
the same five lines.

The Google ADK test suite is split into:
  - ``test_span_interceptor.py`` — synthetic OTel-span unit tests for
    ``OpenInferenceSpanInterceptor`` (no live ADK / Gemini calls).
  - ``test_sync.py`` / ``test_async.py`` — end-to-end traces via real
    ADK agents, schema-asserted. Skipped without ``GOOGLE_API_KEY``.
  - ``test_evaluate_agent.py`` — component-level evals through
    ``dataset.evals_iterator``. Skipped without ``GOOGLE_API_KEY``
    + ``OPENAI_API_KEY`` (the metric scorer).

Skip markers live on the integration test modules themselves
(``pytestmark = pytest.mark.skipif(...)``) — defining them here would
also skip the synthetic interceptor tests, which don't need any keys.
"""

import os

from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
    is_generate_mode,
)


_current_dir = os.path.dirname(os.path.abspath(__file__))
_schemas_dir = os.path.join(_current_dir, "schemas")


def trace_test(schema_name: str):
    """Resolve to ``generate_trace_json`` or ``assert_trace_json``."""
    schema_path = os.path.join(_schemas_dir, schema_name)
    if is_generate_mode():
        return generate_trace_json(schema_path)
    else:
        return assert_trace_json(schema_path)
