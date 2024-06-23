from pydantic import BaseModel, Field
from typing import Optional, List

from deepeval.dataset.golden import Golden, ConversationalGolden


def to_snake_case(string: str) -> str:
    return "".join(
        ["_" + i.lower() if i.isupper() else i for i in string]
    ).lstrip("_")


class APIDataset(BaseModel):
    alias: str
    overwrite: bool
    goldens: Optional[List[Golden]] = Field(default=[])
    conversational_goldens: Optional[List[ConversationalGolden]] = Field(
        default=[], alias="conversationalGoldens"
    )


class CreateDatasetHttpResponse(BaseModel):
    link: str


class DatasetHttpResponse(BaseModel):
    goldens: List[Golden] = Field(alias="goldens")
    conversational_goldens: List[ConversationalGolden] = Field(
        alias="conversationalGoldens"
    )
    datasetId: str
