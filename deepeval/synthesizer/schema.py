from typing import List
from pydantic import BaseModel


class SyntheticData(BaseModel):
    input: str


class SyntheticDataList(BaseModel):
    data: List[SyntheticData]


class SQLData(BaseModel):
    sql: str


class ComplianceData(BaseModel):
    non_compliant: bool


class Response(BaseModel):
    response: str
