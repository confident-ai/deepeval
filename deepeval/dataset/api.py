from pydantic import BaseModel, Field
from typing import Optional, List

from deepeval.dataset.golden import Golden, ConversationalGolden


class APIDataset(BaseModel):
    alias: str
    overwrite: bool
    goldens: Optional[List[Golden]] = Field(None)
    conversational_goldens: Optional[List[ConversationalGolden]] = Field(
        None, alias="conversationalGoldens"
    )


class APIQueueDataset(BaseModel):
    alias: str
    goldens: Optional[List[Golden]] = Field(None)
    conversational_goldens: Optional[List[ConversationalGolden]] = Field(
        None, alias="conversationalGoldens"
    )


class CreateDatasetHttpResponse(BaseModel):
    link: str


class DatasetHttpResponse(BaseModel):
    goldens: Optional[List[Golden]] = Field(None, alias="goldens")
    conversational_goldens: Optional[List[ConversationalGolden]] = Field(
        None, alias="conversationalGoldens"
    )
    datasetId: str
