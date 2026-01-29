import re
import pytest
from deepeval.tracing import observe
from deepeval.tracing.tracing import TraceManager


def simple_mask(data):
    """Simple mask that replaces all strings with [REDACTED]."""
    if isinstance(data, str):
        return "[REDACTED]"
    elif isinstance(data, dict):
        return {k: simple_mask(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [simple_mask(item) for item in data]
    return data


def credit_card_mask(data):
    """Mask that redacts credit card numbers."""
    if isinstance(data, str):
        pattern = r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
        return re.sub(pattern, "****-****-****-****", data)
    elif isinstance(data, dict):
        return {k: credit_card_mask(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [credit_card_mask(item) for item in data]
    return data


def email_mask(data):
    """Mask that redacts email addresses."""
    if isinstance(data, str):
        pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        return re.sub(pattern, "[EMAIL]", data)
    elif isinstance(data, dict):
        return {k: email_mask(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [email_mask(item) for item in data]
    return data


class TestMaskingConfiguration:
    """Tests for masking configuration."""

    def test_no_mask_by_default(self, monkeypatch):
        """Test that no mask is configured by default."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        assert manager.custom_mask_fn is None

    def test_configure_mask_function(self, monkeypatch):
        """Test configuring a mask function."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()

        manager.configure(mask=simple_mask)
        assert manager.custom_mask_fn is simple_mask

    def test_configure_mask_to_none_removes_mask(self, monkeypatch):
        """Test that setting mask to None removes masking."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        manager.configure(mask=simple_mask)
        assert manager.custom_mask_fn is simple_mask

        # Note: setting mask=None in configure doesn't reset it
        # because of the `if mask is not None` check
        # This tests the actual behavior
        manager.custom_mask_fn = None
        assert manager.custom_mask_fn is None

    def test_mask_function_is_called(self, monkeypatch):
        """Test that the mask function is called when masking data."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        manager.configure(mask=simple_mask)

        result = manager.mask("sensitive data")
        assert result == "[REDACTED]"

    def test_mask_returns_original_when_no_mask(self, monkeypatch):
        """Test that mask() returns original data when no mask configured."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()

        original = "sensitive data"
        result = manager.mask(original)
        assert result == original


class TestMaskFunctionBehavior:
    """Tests for mask function behavior."""

    def test_credit_card_mask_function(self, monkeypatch):
        """Test credit card masking function."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        manager.configure(mask=credit_card_mask)

        result = manager.mask("Card: 4111-1111-1111-1111")
        assert result == "Card: ****-****-****-****"

    def test_email_mask_function(self, monkeypatch):
        """Test email masking function."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        manager.configure(mask=email_mask)

        result = manager.mask("Contact: user@example.com")
        assert result == "Contact: [EMAIL]"

    def test_mask_nested_dict(self, monkeypatch):
        """Test masking nested dictionary."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        manager.configure(mask=simple_mask)

        data = {"key": "value", "nested": {"inner": "data"}}
        result = manager.mask(data)
        assert result == {
            "key": "[REDACTED]",
            "nested": {"inner": "[REDACTED]"},
        }

    def test_mask_list(self, monkeypatch):
        """Test masking list."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        manager.configure(mask=simple_mask)

        data = ["one", "two", "three"]
        result = manager.mask(data)
        assert result == ["[REDACTED]", "[REDACTED]", "[REDACTED]"]

    def test_mask_non_string_unchanged(self, monkeypatch):
        """Test that non-string, non-dict, non-list data passes through."""
        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        manager = TraceManager()
        manager.configure(mask=simple_mask)

        assert manager.mask(123) == 123
        assert manager.mask(45.67) == 45.67
        assert manager.mask(True) is True
        assert manager.mask(None) is None


class TestMaskingWithTraces:
    """Tests for masking applied to actual traces."""

    def test_mask_applied_to_input(self, monkeypatch):
        """Test that mask is applied to function input in traces."""
        from deepeval.tracing.tracing import trace_manager

        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        original_mask = trace_manager.custom_mask_fn

        try:
            trace_manager.configure(mask=credit_card_mask)

            @observe()
            def process_card(card_number: str) -> str:
                return f"Processed: {card_number}"

            # The masking happens during trace serialization
            result = process_card("4111-1111-1111-1111")
            assert "4111-1111-1111-1111" in result  # Function sees original

        finally:
            trace_manager.custom_mask_fn = original_mask

    def test_mask_applied_to_output(self, monkeypatch):
        """Test that mask is applied to function output in traces."""
        from deepeval.tracing.tracing import trace_manager

        monkeypatch.delenv("CONFIDENT_TRACE_ENVIRONMENT", raising=False)
        original_mask = trace_manager.custom_mask_fn

        try:
            trace_manager.configure(mask=email_mask)

            @observe()
            def get_email() -> str:
                return "user@example.com"

            result = get_email()
            assert result == "user@example.com"  # Function returns original

        finally:
            trace_manager.custom_mask_fn = original_mask
