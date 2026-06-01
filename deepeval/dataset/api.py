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
        if self.goldens:
            for golden in self.goldens:
                golden.name = None
                golden.images_mapping = golden._get_images_mapping()
                if golden.retrieval_context:
                    golden.retrieval_context = [
                        rc.context if hasattr(rc, "context") else rc
                        for rc in golden.retrieval_context
                    ]
        if self.conversational_goldens:
            for golden in self.conversational_goldens:
                golden.name = None
                golden.images_mapping = golden._get_images_mapping()
                if golden.turns:
                    for turn in golden.turns:
                        if turn.retrieval_context:
                            turn.retrieval_context = [
                                rc.context if hasattr(rc, "context") else rc
                                for rc in turn.retrieval_context
                            ]

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
                if golden.retrieval_context:
                    golden.retrieval_context = [
                        rc.context if hasattr(rc, "context") else rc
                        for rc in golden.retrieval_context
                    ]
        if self.conversational_goldens:
            for golden in self.conversational_goldens:
                golden.name = None
                golden.images_mapping = golden._get_images_mapping()
                if golden.turns:
                    for turn in golden.turns:
                        if turn.retrieval_context:
                            turn.retrieval_context = [
                                rc.context if hasattr(rc, "context") else rc
                                for rc in turn.retrieval_context
                            ]

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
