from __future__ import annotations
from pydantic import Field, conint

from deepeval.optimization.miprov2.configs import MIPROConfig


class COPROConfig(MIPROConfig):
    """
    Configuration for COPRO style cooperative prompt optimization.

    This extends MIPROConfig with settings that control the cooperative
    sampling behavior.

    The core MIPROConfig fields behave exactly the same as in MIPROv2.
    """

    population_size: conint(ge=1) = Field(
        default=4,
        description=(
            "Maximum number of prompt candidates maintained in the active pool. "
            "Once this limit is exceeded, lower scoring candidates are pruned."
        ),
    )

    proposals_per_step: conint(ge=1) = Field(
        default=4,
        description=(
            "Number of child prompts proposed cooperatively from the same "
            "parent in each optimization iteration."
        ),
    )
