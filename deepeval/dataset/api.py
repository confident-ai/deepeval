from pydantic import BaseModel, Field
from typing import Optional, List

from deepeval.dataset.golden import Golden, ConversationalGolden


class APIDataset(BaseModel):
    alias: str
    overwrite: bool
    goldens: Optional[List[Golden]] = Field(default=None)


class CreateDatasetHttpResponse(BaseModel):
    link: str


class DatasetHttpResponse(BaseModel):
    goldens: List[Golden]
    conversational_goldens: List[ConversationalGolden] = Field(
        alias="conversationalGoldens"
    )
    datasetId: str
