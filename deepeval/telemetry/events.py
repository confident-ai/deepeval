"""The canonical event surface.

Every event name that can reach an analytics backend is a member of `Event`.
`capture()` accepts nothing else, which is what stops dynamic names -- metric
class names, integration module paths -- from forking the taxonomy again.
"""

from enum import Enum

# Bumped whenever the event or property vocabulary changes incompatibly, so a
# mixed fleet of SDK versions never blends two schemas in one insight.
TELEMETRY_SCHEMA_VERSION = 2

# Carries one run's id into child processes. The environment is the only
# channel that reaches an xdist worker, which is a fresh interpreter rather
# than a fork, so nothing in memory survives the crossing.
TELEMETRY_RUN_ID_ENV_VAR = "DEEPEVAL_TELEMETRY_RUN_ID"


class Event(str, Enum):
    """One member per user action, not per code path."""

    EVALUATION = "Evaluation"
    SYNTHESIZER = "Synthesizer"
    CONVERSATION_SIMULATOR = "Conversation Simulator"
    BENCHMARK = "Benchmark"
    INTEGRATION_INSTALLED = "Integration Installed"
    CLI_COMMAND = "CLI Command"
    LOGIN_PROMPT_SHOWN = "Login Prompt Shown"
    LOGIN = "Login"


class Entrypoint(str, Enum):
    """How the user got into an evaluation.

    A dimension on `Event.EVALUATION` rather than five separate event names, so
    "how much evaluation happened" stays one query as entrypoints are added.
    """

    EVALUATE = "evaluate"
    EVALS_ITERATOR = "evals_iterator"
    PYTEST = "pytest"
    # The TypeScript SDK's test runner. Never emitted from here.
    VITEST = "vitest"
    COMPARE = "compare"
    STANDALONE = "standalone"


class Feature(str, Enum):
    REDTEAMING = "redteaming"
    SYNTHESIZER = "synthesizer"
    EVALUATION = "evaluation"
    COMPONENT_EVALUATION = "component_evaluation"
    BENCHMARK = "benchmark"
    CONVERSATION_SIMULATOR = "conversation_simulator"
    TRACING_INTEGRATION = "tracing_integration"
    UNKNOWN = "unknown"
