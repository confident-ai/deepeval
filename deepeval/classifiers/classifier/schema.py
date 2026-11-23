from typing import Any, List, Optional, Type

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class ClassificationResult(BaseModel):
    label: str
    reason: str


class EvidenceItem(BaseModel):
    field: str
    quote: str
    note: str = ""
    span_index: Optional[int] = None
    turn_index: Optional[int] = None


class SuggestedClassificationResult(ClassificationResult):
    new_label_description: Optional[str]


class EvidencedClassificationResult(ClassificationResult):
    evidence: List[EvidenceItem] = Field(default_factory=list)

    @field_validator("evidence", mode="before")
    @classmethod
    def _drop_invalid(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return []
        return [item for item in value if _is_evidence(item)]


class SuggestedEvidencedClassificationResult(
    SuggestedClassificationResult, EvidencedClassificationResult
):
    pass


def _is_evidence(item: Any) -> bool:
    return _is_valid(EvidenceItem, item)


def _is_valid(model: Type[BaseModel], item: Any) -> bool:
    try:
        model.model_validate(item)
    except ValidationError:
        return False
    return True


class LabelMergeResult(BaseModel):
    same_as: Optional[str] = None
    reason: str = ""


class LabelReview(BaseModel):
    summary: str = ""
    confidence: float = Field(ge=0, le=1)
    reason: str = ""


def _label_list(item: Type[ClassificationResult]) -> Type[BaseModel]:
    class LabelList(BaseModel):
        model_config = ConfigDict(title=f"{item.__name__}List")
        labels: List[item]

        @model_validator(mode="before")
        @classmethod
        def _lenient(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                return data
            if "labels" not in data and "label" in data:
                return {"labels": [data]}
            if isinstance(data.get("labels"), list):
                return {
                    **data,
                    "labels": [i for i in data["labels"] if _is_valid(item, i)],
                }
            return data

    return LabelList


LABEL_LIST_SCHEMAS = {
    item: _label_list(item)
    for item in (
        ClassificationResult,
        SuggestedClassificationResult,
        EvidencedClassificationResult,
        SuggestedEvidencedClassificationResult,
    )
}
