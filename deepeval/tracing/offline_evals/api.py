from typing import Optional
from pydantic import BaseModel, Field


class EvaluateThreadRequestBody(BaseModel):
    metric_collection: str = Field(alias="metricCollection")
    overwrite_metrics: bool = Field(alias="overwriteMetrics")
    chatbot_role: Optional[str] = Field(default=None, alias="chatbotRole")


class EvaluateTraceRequestBody(BaseModel):
    metric_collection: str = Field(alias="metricCollection")
    overwrite_metrics: bool = Field(alias="overwriteMetrics")


class EvaluateSpanRequestBody(BaseModel):
    metric_collection: str = Field(alias="metricCollection")
    overwrite_metrics: bool = Field(alias="overwriteMetrics")
