from typing import List

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from pydantic_core import to_jsonable_python


class Expectations(BaseModel):
    """Required and prohibited behavior for a case or entire conversation."""

    model_config = ConfigDict(extra="forbid")

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
