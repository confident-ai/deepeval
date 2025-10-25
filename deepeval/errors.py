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


class MissingTestCaseParamsError(DeepEvalError):
    """Required test case fields are missing."""

    pass


class MismatchedTestCaseInputsError(DeepEvalError):
    """Inputs provided to a metric or test case are inconsistent or invalid."""

    pass
