import logging
from typing import Optional

import deepeval
from deepeval.config.settings import get_settings
from deepeval.telemetry import capture_tracing_integration

logger = logging.getLogger(__name__)

try:
    from autogen import ConversableAgent  # noqa: F401
    from autogen.oai.client import OpenAIWrapper  # noqa: F401

    ag2_installed = True
except ImportError as e:
    if get_settings().DEEPEVAL_VERBOSE_MODE:
        if isinstance(e, ModuleNotFoundError):
            logger.warning(
                "Optional ag2 dependency not installed: %s",
                e.name,
                stacklevel=2,
            )
        else:
            logger.warning(
                "Optional ag2 import failed: %s",
                e,
                stacklevel=2,
            )
    ag2_installed = False

IS_WRAPPED_ALL = False


def is_ag2_installed():
    if not ag2_installed:
        raise ImportError(
            "AG2 is not installed. Please install it with `pip install ag2[openai]`."
        )


def instrument_ag2(api_key: Optional[str] = None):
    """Instrument AG2 agents to capture traces for DeepEval evaluation.

    Call this before running any AG2 conversations. It patches
    ConversableAgent methods to capture LLM calls, tool executions,
    and agent interactions as DeepEval trace spans.

    Args:
        api_key: Optional Confident AI API key for cloud tracing.
            If not provided, uses DEEPEVAL_API_KEY env var.

    Example:
        from deepeval.integrations.ag2 import instrument_ag2
        instrument_ag2()

        # Now run your AG2 agents as usual - traces are captured automatically
        executor.run(assistant, message="...").process()
    """
    is_ag2_installed()

    with capture_tracing_integration("ag2"):
        if api_key:
            deepeval.login(api_key)
        wrap_all()


def reset_ag2_instrumentation():
    """Remove AG2 instrumentation and restore original methods."""
    global IS_WRAPPED_ALL

    if not IS_WRAPPED_ALL:
        return

    from deepeval.integrations.ag2.wrapper import (
        unwrap_all,
    )

    unwrap_all()
    IS_WRAPPED_ALL = False


def wrap_all():
    global IS_WRAPPED_ALL

    if not IS_WRAPPED_ALL:
        from deepeval.integrations.ag2.wrapper import (
            wrap_generate_reply,
            wrap_a_generate_reply,
            wrap_execute_function,
            wrap_a_execute_function,
            wrap_openai_wrapper_create,
        )

        wrap_generate_reply()
        wrap_a_generate_reply()
        wrap_execute_function()
        wrap_a_execute_function()
        wrap_openai_wrapper_create()

        IS_WRAPPED_ALL = True
