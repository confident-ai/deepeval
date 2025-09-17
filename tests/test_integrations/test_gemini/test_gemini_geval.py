import json
from unittest import mock

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.evaluate import assert_test
from deepeval.models import GeminiModel


class _TopLogProb:
    def __init__(self, token: str, logprob: float):
        self.token = token
        self.logprob = logprob


class _TokenEntry:
    def __init__(self, token: str, top_logprobs):
        self.token = token
        self.top_logprobs = top_logprobs


class _Logprobs:
    def __init__(self, content):
        self.content = content


class _Message:
    def __init__(self, content, logprobs):
        self.content = content
        self.logprobs = logprobs


class _Choice:
    def __init__(self, message, logprobs):
        self.message = message
        self.logprobs = logprobs


class _ChatCompletion:
    def __init__(self, choices):
        self.choices = choices


def _fake_generate_steps(self, prompt: str, schema=None):
    # Return deterministic JSON for steps vs scoring prompts
    if "generate 3-4 concise evaluation steps" in prompt:
        return (
            json.dumps(
                {
                    "steps": [
                        "Check relevance",
                        "Check correctness",
                        "Check completeness",
                    ]
                }
            ),
            0,
        )
    # Scoring/evaluation path
    return (
        json.dumps(
            {
                "score": 8,
                "reason": "Response is relevant, correct, and sufficiently complete.",
            }
        ),
        0,
    )


def _fake_generate_raw_response(self, prompt: str, top_logprobs: int = 20):
    # Return score 8 with logprobs: P(8)=0.9, P(7)=0.1 -> weighted sum 7.9
    token_entry = _TokenEntry(
        token="8",
        top_logprobs=[
            _TopLogProb(token="8", logprob=-0.10536051565782628),  # ~0.9
            _TopLogProb(token="7", logprob=-2.302585092994046),  # 0.1
        ],
    )
    logprobs = _Logprobs(content=[token_entry])
    message = _Message(
        content=json.dumps({"score": 8, "reason": "Weighted by logprobs"}),
        logprobs=logprobs,
    )
    completion = _ChatCompletion(
        choices=[_Choice(message=message, logprobs=logprobs)]
    )
    return completion, 0


def test_stubbed_gemini_geval_generate():
    """Test that the stubbed gemini model can generate a sampled score when falling back to the generate function."""
    with (
        mock.patch.object(
            GeminiModel, "load_model", lambda self, *a, **k: None
        ),
        mock.patch.object(GeminiModel, "generate", _fake_generate_steps),
    ):
        model = GeminiModel()

        metric = GEval(
            name="Validity",
            criteria="The response should directly answer the user question accurately.",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            model=model,
            threshold=0.8,
            verbose_mode=True,
        )

        test_case = LLMTestCase(
            input="What is the capital of France?",
            actual_output="Paris is the capital of France.",
            expected_output="Paris",
        )

        # Score 8 -> normalized 0.8, should pass
        assert_test(test_case=test_case, metrics=[metric], run_async=False)


def test_stubbed_gemini_geval_generate_raw_response_weighted_sum():
    """Test that the stubbed gemini model can generate a weighted summed score when using the generate_raw_response function."""
    with (
        mock.patch.object(
            GeminiModel, "load_model", lambda self, *a, **k: None
        ),
        mock.patch.object(GeminiModel, "generate", _fake_generate_steps),
        mock.patch.object(
            GeminiModel, "generate_raw_response", _fake_generate_raw_response
        ),
    ):
        model = GeminiModel()

        metric = GEval(
            name="Validity",
            criteria="The response should directly answer the user question accurately.",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            model=model,
            threshold=0.79,
            verbose_mode=True,
        )

        test_case = LLMTestCase(
            input="What is the capital of France?",
            actual_output="Paris is the capital of France.",
            expected_output="Paris",
        )

        # Weighted sum is 7.9 -> normalized 0.79, should pass
        assert_test(test_case=test_case, metrics=[metric], run_async=False)


##############################################
# Test Everything
##############################################

if __name__ == "__main__":
    test_stubbed_gemini_geval_generate()
    test_stubbed_gemini_geval_generate_raw_response_weighted_sum()
