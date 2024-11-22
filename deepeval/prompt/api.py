from pydantic import BaseModel


class PromptHttpResponse(BaseModel):
    value: str
    version: str
