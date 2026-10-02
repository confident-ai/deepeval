from typing import Any, List, Optional, TYPE_CHECKING, Union

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    field_serializer,
)
from pydantic_core import to_jsonable_python
from deepeval.config.eval_mode import ClassifierEvalModeName

if TYPE_CHECKING:
    from deepeval.models import DeepEvalBaseLLM

    ExpectationModel = Optional[Union[str, DeepEvalBaseLLM]]
else:
    # Importing models while test_case is loading would form an import cycle.
    ExpectationModel = Any


class Expectations(BaseModel):
    """Required and prohibited behavior for a case or entire conversation."""

    model_config = ConfigDict(extra="forbid")

    model: ExpectationModel = Field(default=None, repr=False)
    eval_mode: Optional[ClassifierEvalModeName] = Field(default=None)

    @field_serializer("model")
    def serialize_model(self, model) -> Optional[str]:
        # Model names are portable; live clients and their credentials are not.
        return model if isinstance(model, str) else None

    @field_validator("model")
    @classmethod
    def validate_model(cls, model):
        from deepeval.models import DeepEvalBaseLLM

        if model is not None and not isinstance(model, (str, DeepEvalBaseLLM)):
            raise ValueError(
                "model must be a model name or DeepEvalBaseLLM instance"
            )
        return model

    must: List[str] = Field(default_factory=list)
    must_not: List[str] = Field(
        default_factory=list,
        serialization_alias="mustNot",
        validation_alias=AliasChoices("must_not", "mustNot"),
    )

    @field_validator("must", "must_not")
    @classmethod
    def validate_conditions(cls, conditions: List[str]) -> List[str]:
        if any(not condition.strip() for condition in conditions):
            raise ValueError("Expectations must contain non-empty conditions.")
        return [condition.strip() for condition in conditions]

    def __bool__(self) -> bool:
        return bool(self.must or self.must_not)


def expectation_evidence(test_case) -> dict:
    """The exact observations used by the judge and its cache key."""
    evidence = test_case.model_dump(
        mode="json",
        exclude_none=True,
        include={
            "input",
            "actual_output",
            "context",
            "retrieval_context",
            "tools_called",
            "mcp_tools_called",
            "mcp_resources_called",
            "mcp_prompts_called",
            "turns",
            "scenario",
            "chatbot_role",
        },
    )

    trace = getattr(test_case, "_trace_dict", None)
    if trace is not None:
        evidence["trace"] = to_jsonable_python(trace, fallback=str)
    return evidence
