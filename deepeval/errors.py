class DeepEvalError(Exception):
    """Base class for framework-originated errors.
    If raised and not handled, it will abort the current operation.
    We may also stringify instances of this class and attach them to traces or spans to surface
    non-fatal diagnostics while allowing the run to continue.
    """


class UserAppError(Exception):
    """Represents exceptions thrown by user LLM apps/tools.
    We record these on traces or spans and keep the overall evaluation run alive.
    """


class SpeechHTTPError(DeepEvalError):
    """A speech (TTS/STT) provider returned a non-2xx response.

    Defined here rather than beside the transport that raises it so
    `deepeval.models.retry_policy` can classify it without importing
    `deepeval.models.speech`, whose providers import the retry policy.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_label: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.provider_label = provider_label


class SpeechAuthError(SpeechHTTPError):
    """Speech provider credentials were rejected (401/403). Never retryable."""


class SpeechRateLimitError(SpeechHTTPError):
    """A speech provider is rate limiting (429). Usually worth retrying."""


class MissingTestCaseParamsError(DeepEvalError):
    """Required test case fields are missing."""


class MismatchedTestCaseInputsError(DeepEvalError):
    """Inputs provided to a metric or test case are inconsistent or invalid."""


class NoMetricsError(DeepEvalError):
    """An evaluation run was started with no metric sources at any level.

    Raised by the ``evals_iterator`` executor when, after iteration completes,
    we can prove that no metrics were declared via:
      - ``evals_iterator(metrics=[...])`` (top-level / trace-level metrics)
      - ``@observe(metrics=[...])`` or ``@observe(metric_collection=...)``
        on any span (span-level metrics)
      - ``update_current_trace(metrics=[...])`` inside the traced function
        (trace-level metrics, set at runtime)

    Without this check, the user would silently get a misleading
    ``"All metrics errored for all test cases, please try again."`` print
    at the end of a run that quietly did nothing.
    """


class IncompatibleTestRunsError(DeepEvalError, ValueError):
    """Raised when comparing historical test runs that are incompatible.

    Common causes:
      - Runs evaluated a different number of test cases.
      - Test case inputs or definitions do not match.
      - One or both test runs contain no test cases.
    """


class JudgeEvaluationError(DeepEvalError, RuntimeError):
    """Raised when an LLM judge fails during test run comparison.

    Common causes:
      - The judge model raises an unhandled exception or network error.
      - The judge returns unparseable or incomplete output after retry.
    """
