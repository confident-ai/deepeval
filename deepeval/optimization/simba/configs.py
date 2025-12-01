from __future__ import annotations
from pydantic import Field, PositiveInt, conint

from deepeval.optimization.copro.configs import COPROConfig


class SIMBAConfig(COPROConfig):
    """
    Configuration for SIMBA style cooperative prompt optimization.

    Extends `COPROConfig` with strategy specific controls:

      - How many minibatch examples are surfaced as demos and how long
        those snippets can be (`max_demos_per_proposal`,
        `demo_input_max_chars`).
    """

    max_demos_per_proposal: conint(ge=0) = Field(
        default=3,
        description=(
            "Maximum number of goldens from the current minibatch that are "
            "converted into concrete input/output demos when using the "
            "APPEND_DEMO strategy."
        ),
    )

    demo_input_max_chars: PositiveInt = Field(
        default=256,
        description=(
            "Maximum number of characters taken from the golden input and "
            "expected output when constructing demo snippets for APPEND_DEMO."
        ),
    )
