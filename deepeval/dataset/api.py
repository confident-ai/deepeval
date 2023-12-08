from pydantic import BaseModel, Field
from typing import Optional, List, Union


class Golden(BaseModel):
    input: str
    actual_output: Optional[str] = Field(None, alias="actualOutput")
    expected_output: Optional[str] = Field(None, alias="expectedOutput")
    context: Optional[list] = Field(None)


class APIDataset(BaseModel):
    alias: str
    overwrite: bool
    goldens: Optional[List[Golden]] = Field(default=None)


class CreateDatasetHttpResponse(BaseModel):
    link: str


class DatasetHttpResponse(BaseModel):
    goldens: List[Golden]
