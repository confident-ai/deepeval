from pydantic import BaseModel, Field


class PromptHttpResponse(BaseModel):
    promptVersionId: str
    template: str


class PromptApi(BaseModel):
    promptVersionId: str
    template: str = Field(serialization_alias="value")
