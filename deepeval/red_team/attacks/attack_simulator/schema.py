from pydantic import BaseModel
from typing import List


class SyntheticData(BaseModel):
    input: str


class SyntheticDataList(BaseModel):
    data: List[SyntheticData]
