from enum import Enum


class SIMBAStrategy(str, Enum):
    """
    Edit strategies used by SIMBA-style optimization.

    - APPEND_DEMO: append one or more input/output demos distilled from the
      current minibatch, similar in spirit to DSPy's `append_a_demo`.
    - APPEND_RULE: append a concise natural-language rule distilled from
      feedback, similar in spirit to DSPy's `append_a_rule`.
    """

    APPEND_DEMO = "append_demo"
    APPEND_RULE = "append_rule"
