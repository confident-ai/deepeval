from typing import List, Optional

from pydantic import BaseModel, Field

from deepeval.dataset.golden import ConversationalGolden, Golden


class APIDataset(BaseModel):
    alias: str
    overwrite: Optional[bool] = None
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
