from pydantic import BaseModel, Field, model_validator
from typing import Optional, List

from deepeval.dataset.golden import Golden, ConversationalGolden


class APIDataset(BaseModel):
    finalized: bool
    version: Optional[str] = None
    goldens: Optional[List[Golden]] = Field(None)
    conversational_goldens: Optional[List[ConversationalGolden]] = Field(
        None, alias="conversationalGoldens"
    )

    @model_validator(mode="after")
    def prepare_goldens_for_api(self):
        for golden in self.goldens or []:
            golden._prepare_for_api()
        for golden in self.conversational_goldens or []:
            golden._prepare_for_api()
        return self


class APIQueueDataset(BaseModel):
    alias: str
    goldens: Optional[List[Golden]] = Field(None)
    conversational_goldens: Optional[List[ConversationalGolden]] = Field(
        None, alias="conversationalGoldens"
    )

    @model_validator(mode="after")
    def prepare_goldens_for_api(self):
        for golden in self.goldens or []:
            golden._prepare_for_api()
        for golden in self.conversational_goldens or []:
            golden._prepare_for_api()
        return self


class DatasetHttpResponse(BaseModel):
    id: str
    version: Optional[str] = None
    goldens: Optional[List[Golden]] = Field(None, alias="goldens")
    conversational_goldens: Optional[List[ConversationalGolden]] = Field(
        None, alias="conversationalGoldens"
    )


class DatasetVersion(BaseModel):
    id: str
    version: str
    created_at: Optional[str] = Field(None, alias="createdAt")


class DatasetVersionsHttpResponse(BaseModel):
    versions: List[DatasetVersion]


class CreateDatasetVersionHttpResponse(BaseModel):
    id: str
    version: str
