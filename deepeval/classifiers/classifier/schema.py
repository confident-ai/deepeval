from typing import Any, List, Optional, Type, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

###############################################
# Classifications
###############################################


def _to_text(value: Any) -> str:
    return "" if value is None else str(value)


class ClassificationResult(BaseModel):
    label: str
    reason: str = ""

    @field_validator("reason", mode="before")
    @classmethod
    def _coerce_reason(cls, value: Any) -> str:
        return _to_text(value)


def _is_valid(model: Type[BaseModel], item: Any) -> bool:
    try:
        model.model_validate(item)
    except ValidationError:
        return False
    return True


###############################################
# Evidence
###############################################


class EvidenceItem(BaseModel):
    field: str
    quote: str
    note: str = ""
    span_index: Optional[int] = None
    turn_index: Optional[int] = None

    @field_validator("note", mode="before")
    @classmethod
    def _coerce_note(cls, value: Any) -> str:
        return _to_text(value)


class EvidencedClassificationResult(ClassificationResult):
    evidence: List[EvidenceItem] = Field(default_factory=list)

    @field_validator("evidence", mode="before")
    @classmethod
    def _drop_invalid(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return []
        return [item for item in value if _is_valid(EvidenceItem, item)]


###############################################
# Label Discovery
###############################################


class SuggestedClassificationResult(ClassificationResult):
    new_label_description: Optional[str] = None

    @field_validator("new_label_description", mode="before")
    @classmethod
    def _drop_empty_description(cls, value: Any) -> Optional[str]:
        text = _to_text(value).strip()
        return None if text.lower() in ("", "null", "none") else text


class SuggestedEvidencedClassificationResult(
    SuggestedClassificationResult, EvidencedClassificationResult
):
    pass


class LabelMergeResult(BaseModel):
    same_as: Optional[str] = None
    reason: str = ""


###############################################
# Review
###############################################


class LabelReview(BaseModel):
    summary: str = ""
    confidence: float = Field(ge=0, le=1)
    reason: str = ""

    def explanation(self) -> str:
        return self.reason


###############################################
# Multi-Label
###############################################


class ClassificationResults(BaseModel):
    labels: List[ClassificationResult]


def as_classification_results(
    response: Union[ClassificationResult, ClassificationResults],
) -> List[ClassificationResult]:
    if isinstance(response, ClassificationResults):
        return response.labels
    return [response]


def _label_list(
    item: Type[ClassificationResult],
) -> Type[ClassificationResults]:
    class LabelList(ClassificationResults):
        model_config = ConfigDict(title=f"{item.__name__}List")
        labels: List[item]

        @model_validator(mode="before")
        @classmethod
        def _lenient(cls, data: Any) -> Any:
            if not isinstance(data, dict):
                return data
            if "labels" not in data and "label" in data:
                return {"labels": [data] if _is_valid(item, data) else []}
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
