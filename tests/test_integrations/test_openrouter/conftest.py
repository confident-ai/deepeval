import os

from tests.test_integrations.utils import (
    assert_trace_json,
    generate_trace_json,
)

_FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures"
)

GENERATE_MODE = os.environ.get("GENERATE_SCHEMAS", "").lower() in (
    "true",
    "1",
    "yes",
)


def trace_test(fixture_name: str):
    """Assert the trace against its fixture, or regenerate it.

    Regenerate with:
        GENERATE_SCHEMAS=true pytest tests/test_integrations/test_openrouter/
    """
    fixture_path = os.path.join(_FIXTURES_DIR, fixture_name)
    if GENERATE_MODE:
        return generate_trace_json(fixture_path)
    return assert_trace_json(fixture_path)
