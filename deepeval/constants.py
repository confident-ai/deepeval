from enum import Enum
from typing import Union

KEY_FILE: str = ".deepeval"
HIDDEN_DIR: str = ".deepeval"
PYTEST_RUN_TEST_NAME: str = "CONFIDENT_AI_RUN_TEST_NAME"
LOGIN_PROMPT = "\n✨👀 Looking for a place for your LLM test data to live 🏡❤️ ? Use [rgb(106,0,255)]Confident AI[/rgb(106,0,255)] to get & share testing reports, experiment with models/prompts, and catch regressions for your LLM system. Just run [cyan]'deepeval login'[/cyan] in the CLI."


CONFIDENT_TRACE_VERBOSE = "CONFIDENT_TRACE_VERBOSE"
CONFIDENT_TRACE_FLUSH = "CONFIDENT_TRACE_FLUSH"
CONFIDENT_TRACE_SAMPLE_RATE = "CONFIDENT_TRACE_SAMPLE_RATE"
CONFIDENT_TRACE_ENVIRONMENT = "CONFIDENT_TRACE_ENVIRONMENT"
CONFIDENT_TRACING_ENABLED = "CONFIDENT_TRACING_ENABLED"

CONFIDENT_METRIC_LOGGING_VERBOSE = "CONFIDENT_METRIC_LOGGING_VERBOSE"
CONFIDENT_METRIC_LOGGING_FLUSH = "CONFIDENT_METRIC_LOGGING_FLUSH"
CONFIDENT_METRIC_LOGGING_SAMPLE_RATE = "CONFIDENT_METRIC_LOGGING_SAMPLE_RATE"
CONFIDENT_METRIC_LOGGING_ENABLED = "CONFIDENT_METRIC_LOGGING_ENABLED"


CONFIDENT_OPEN_BROWSER = "CONFIDENT_OPEN_BROWSER"
CONFIDENT_TEST_CASE_BATCH_SIZE = "CONFIDENT_TEST_CASE_BATCH_SIZE"


class ProviderSlug(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    DEEPSEEK = "deepseek"
    GOOGLE = "google"
    GROK = "grok"
    KIMI = "kimi"
    LITELLM = "litellm"
    LOCAL = "local"
    OLLAMA = "ollama"


def slugify(value: Union[str, ProviderSlug]) -> str:
    return (
        value.value
        if isinstance(value, ProviderSlug)
        else str(value).strip().lower()
    )


SUPPORTED_PROVIDER_SLUGS = frozenset(s.value for s in ProviderSlug)
