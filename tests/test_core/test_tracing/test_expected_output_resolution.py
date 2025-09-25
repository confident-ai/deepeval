from deepeval.contextvars import set_current_golden, reset_current_golden
from deepeval.tracing import context as tracing_ctx
from deepeval.utils import is_missing


class _GoldenStub:
    def __init__(self, expected_output=None):
        self.expected_output = expected_output


def test_is_missing_helper_covers_whitespace():
    assert is_missing(None) is True
    assert is_missing("") is True
    assert is_missing("   ") is True
    assert is_missing("\n\t ") is True
    assert is_missing("x") is False
    assert is_missing("  x  ") is False


def test_resolve_expected_output_from_context_when_missing():
    # active golden has an expected_output
    tok = set_current_golden(
        _GoldenStub(expected_output="EXPECTED_FROM_DATASET")
    )
    try:
        # missing should resolve from CURRENT_GOLDEN
        assert (
            tracing_ctx._resolve_expected_output_from_context(None)
            == "EXPECTED_FROM_DATASET"
        )
        assert (
            tracing_ctx._resolve_expected_output_from_context("")
            == "EXPECTED_FROM_DATASET"
        )
        assert (
            tracing_ctx._resolve_expected_output_from_context("   ")
            == "EXPECTED_FROM_DATASET"
        )
    finally:
        reset_current_golden(tok)


def test_resolve_expected_output_from_context_respects_explicit_value():
    tok = set_current_golden(
        _GoldenStub(expected_output="EXPECTED_FROM_DATASET")
    )
    try:
        # non-missing should pass through unchanged
        assert (
            tracing_ctx._resolve_expected_output_from_context("USER_VALUE")
            == "USER_VALUE"
        )
        assert (
            tracing_ctx._resolve_expected_output_from_context("  USER_VALUE  ")
            == "  USER_VALUE  "
        )
    finally:
        reset_current_golden(tok)


def test_resolve_expected_output_from_context_when_no_golden_set():
    # no CURRENT_GOLDEN will resolve to the original value. Which is missing in this case
    assert tracing_ctx._resolve_expected_output_from_context(None) is None
    assert tracing_ctx._resolve_expected_output_from_context("") == ""
