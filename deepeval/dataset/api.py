from pydantic import BaseModel, Field, model_validator
from typing import Optional, List

from deepeval.dataset.golden import Golden, ConversationalGolden


class APIDataset(BaseModel):
    finalized: bool
    goldens: Optional[List[Golden]] = Field(None)
    conversational_goldens: Optional[List[ConversationalGolden]] = Field(
        None, alias="conversationalGoldens"
    )

    @model_validator(mode="after")
    def prepare_goldens_for_api(self):
        if self.goldens:
            for golden in self.goldens:
                golden.name = None
                golden.images_mapping = golden._get_images_mapping()
        if self.conversational_goldens:
            for golden in self.conversational_goldens:
                golden.name = None
                golden.images_mapping = golden._get_images_mapping()

        return self


class APIQueueDataset(BaseModel):
    alias: str
    goldens: Optional[List[Golden]] = Field(None)
    conversational_goldens: Optional[List[ConversationalGolden]] = Field(
        None, alias="conversationalGoldens"
    )

    @model_validator(mode="after")
    def prepare_goldens_for_api(self):
        if self.goldens:
            for golden in self.goldens:
                golden.name = None
                golden.images_mapping = golden._get_images_mapping()
        if self.conversational_goldens:
            for golden in self.conversational_goldens:
                golden.name = None
                golden.images_mapping = golden._get_images_mapping()

        return self


class DatasetHttpResponse(BaseModel):
    id: str
    goldens: Optional[List[Golden]] = Field(None, alias="goldens")
    conversational_goldens: Optional[List[ConversationalGolden]] = Field(
        None, alias="conversationalGoldens"
    )
