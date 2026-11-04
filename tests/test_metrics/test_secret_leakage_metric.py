import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.community import SecretLeakageMetric
from deepeval.test_case import LLMTestCase

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
OPENAI_KEY = "sk-" + "a" * 32
GITHUB_TOKEN = "ghp_" + "b" * 36
SLACK_TOKEN = "xoxb-123456789012-abcdefghij"
PRIVATE_KEY = "-----BEGIN RSA PRIVATE KEY-----"
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0." + "c" * 16
ENTROPY_TOKEN = "aB3dE5fG7hJ9kL2mN4pQ6rS8tU0vW"


def _test_case(actual_output):
    return LLMTestCase(
        input="What is my key?",
        actual_output=actual_output,
    )


class TestSecretLeakageMetric:
    """SecretLeakageMetric is deterministic, so these run without a key."""

    def test_clean_output_passes(self):
        metric = SecretLeakageMetric()
        metric.measure(_test_case("The sky is blue and roses are red."))
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_aws_key_detected(self):
        metric = SecretLeakageMetric()
        metric.measure(_test_case(f"My key is {AWS_KEY}."))
        assert metric.score == 0.0
        assert metric.is_successful() is False
        assert "aws_key (1)" in metric.reason
        assert AWS_KEY not in metric.reason

    def test_openai_key_detected(self):
        metric = SecretLeakageMetric()
        metric.measure(_test_case(f"Key: {OPENAI_KEY}"))
        assert metric.score == 0.0
        assert "openai_key (1)" in metric.reason

    def test_github_token_detected(self):
        metric = SecretLeakageMetric()
        metric.measure(_test_case(f"Token {GITHUB_TOKEN} here."))
        assert metric.score == 0.0
        assert "github_token (1)" in metric.reason

    def test_slack_token_detected(self):
        metric = SecretLeakageMetric()
        metric.measure(_test_case(f"Token {SLACK_TOKEN} here."))
        assert metric.score == 0.0
        assert "slack_token (1)" in metric.reason

    def test_private_key_detected(self):
        metric = SecretLeakageMetric()
        metric.measure(_test_case(f"Key:\n{PRIVATE_KEY}\nMII..."))
        assert metric.score == 0.0
        assert "private_key (1)" in metric.reason

    def test_jwt_detected(self):
        metric = SecretLeakageMetric()
        metric.measure(_test_case(f"Bearer {JWT}"))
        assert metric.score == 0.0
        assert "jwt (1)" in metric.reason

    def test_values_never_leak_into_logs(self):
        metric = SecretLeakageMetric(verbose_mode=True)
        metric.measure(_test_case(f"Key: {OPENAI_KEY}"))
        assert OPENAI_KEY not in (metric.verbose_logs or "")

    def test_entities_subset(self):
        metric = SecretLeakageMetric(entities=["aws_key"])
        metric.measure(_test_case(f"Key: {OPENAI_KEY}"))
        assert metric.score == 1.0
        metric.measure(_test_case(f"Key: {AWS_KEY}"))
        assert metric.score == 0.0

    def test_custom_patterns(self):
        metric = SecretLeakageMetric(
            entities=[], custom_patterns={"internal": r"INT-[0-9]{6}"}
        )
        metric.measure(_test_case("Ticket INT-123456 closed."))
        assert metric.score == 0.0
        assert "internal (1)" in metric.reason

    def test_entropy_check_off_by_default(self):
        metric = SecretLeakageMetric()
        metric.measure(_test_case(f"Code {ENTROPY_TOKEN} done."))
        assert metric.score == 1.0

    def test_entropy_check_flags_random_strings(self):
        metric = SecretLeakageMetric(check_entropy=True)
        metric.measure(_test_case(f"Code {ENTROPY_TOKEN} done."))
        assert metric.score == 0.0
        assert "high_entropy_secret (1)" in metric.reason

    def test_unknown_entity_rejected(self):
        with pytest.raises(ValueError):
            SecretLeakageMetric(entities=["nope"])

    def test_empty_check_rejected(self):
        with pytest.raises(ValueError):
            SecretLeakageMetric(entities=[])

    def test_include_reason_false(self):
        metric = SecretLeakageMetric(include_reason=False)
        metric.measure(_test_case(f"Key: {AWS_KEY}"))
        assert metric.reason is None

    def test_missing_actual_output_raises(self):
        metric = SecretLeakageMetric()
        with pytest.raises(MissingTestCaseParamsError):
            metric.measure(LLMTestCase(input="What is my key?"))

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = SecretLeakageMetric()
        score = await metric.a_measure(_test_case(f"Key: {AWS_KEY}"))
        assert score == 0.0
