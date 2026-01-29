import re
import pytest
from deepeval.tracing import observe, trace_manager


def mask_credit_cards(data):
    if isinstance(data, str):
        pattern = r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
        return re.sub(pattern, "****-****-****-****", data)
    elif isinstance(data, dict):
        return {k: mask_credit_cards(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [mask_credit_cards(item) for item in data]
    return data


def mask_emails(data):
    if isinstance(data, str):
        pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        return re.sub(pattern, "***@***.***", data)
    elif isinstance(data, dict):
        return {k: mask_emails(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [mask_emails(item) for item in data]
    return data


def comprehensive_mask(data):
    data = mask_credit_cards(data)
    data = mask_emails(data)
    return data


@observe()
def process_with_credit_card(user_input: str) -> str:
    return f"Processed: {user_input}"


@observe()
def process_with_email(user_input: str) -> str:
    return f"Email processed: {user_input}"


@observe()
def process_sensitive_data(data: dict) -> dict:
    return {"result": "processed", "original": data}


class TestMasking:

    def test_credit_card_masking(self):
        trace_manager.configure(mask=mask_credit_cards)
        try:
            result = process_with_credit_card("My card is 4111-1111-1111-1111")
            assert result == "Processed: My card is 4111-1111-1111-1111"
        finally:
            trace_manager.configure(mask=None)

    def test_email_masking(self):
        trace_manager.configure(mask=mask_emails)
        try:
            result = process_with_email("Contact: user@example.com")
            assert result == "Email processed: Contact: user@example.com"
        finally:
            trace_manager.configure(mask=None)

    def test_comprehensive_masking(self):
        trace_manager.configure(mask=comprehensive_mask)
        try:
            data = {
                "email": "user@example.com",
                "card": "4111-1111-1111-1111",
                "name": "John Doe",
            }
            result = process_sensitive_data(data)
            assert result["result"] == "processed"
        finally:
            trace_manager.configure(mask=None)

    def test_no_masking_by_default(self):
        trace_manager.configure(mask=None)
        result = process_with_credit_card("Card: 1234-5678-9012-3456")
        assert "1234-5678-9012-3456" in result

    def test_masking_function_helper(self):
        assert mask_credit_cards("4111-1111-1111-1111") == "****-****-****-****"
        assert mask_credit_cards("4111111111111111") == "****-****-****-****"
        assert mask_emails("test@example.com") == "***@***.***"

        data = {"cc": "4111-1111-1111-1111", "nested": {"email": "a@b.com"}}
        masked = comprehensive_mask(data)
        assert "****" in str(masked)
