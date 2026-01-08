from __future__ import annotations
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from deepeval.metrics.g_eval.utils import ApiGEval

class ApiMetric(BaseModel):
    name: str = Field(alias="name")

class ApiBinaryJudgementNode(BaseModel):
    name: str = Field(alias="name", default="binaryJudegementNode")
    criteria: str = Field(alias="criteria")
    label: Optional[str] = Field(alias="label")
    evaluation_params: Optional[List[str]] = Field(alias="evaluationParams")
    children: List[ApiVerdictNode] = Field(alias="children")

class ApiNonBinaryJudgementNode(BaseModel):
    name: str = Field(alias="name", default="nonBinaryJudgementNode")
    criteria: str = Field(alias="criteria")
    label: Optional[str] = Field(alias="label")
    evaluation_params: Optional[List[str]] = Field(alias="evaluationParams")
    children: List[ApiVerdictNode] = Field(alias="children")

class ApiTaskNode(BaseModel):
    name: str = Field(alias="name", default="taskNode")
    instructions: str = Field(alias="instructions")
    label: Optional[str] = Field(alias="label")
    output_label: str = Field(alias="outputLabel")
    evaluation_params: List[str] = Field(alias="evaluationParams")
    children: List[Union[ApiBinaryJudgementNode, ApiNonBinaryJudgementNode, ApiTaskNode]] = Field(alias="children")

class ApiVerdictNode(BaseModel):
    name: str = Field(alias="name", default="verdictNode")
    verdict: Union[str, bool] = Field(alias="verdict")
    score: Optional[int] = Field(alias="score")
    child: Optional[Union[ApiTaskNode, ApiBinaryJudgementNode, ApiNonBinaryJudgementNode, ApiMetric, ApiGEval]] = Field(alias="child")

class ApiDAG(BaseModel):
    root_nodes: List[Union[
        ApiTaskNode, ApiBinaryJudgementNode, ApiNonBinaryJudgementNode, ApiVerdictNode
    ]] = Field(alias="rootNodes")

class ApiDAGMetric(BaseModel):
    name: str = Field(alias="name")
    multi_turn: bool = Field(alias="multiTurn")
    is_dag: bool = Field(alias="isDag", default=True)
    dag: ApiDAG = Field(alias="dag")