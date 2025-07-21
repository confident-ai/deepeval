from pydantic import BaseModel, Field


class EvaluateThreadRequestBody(BaseModel):
    metric_collection: str = Field(alias="metricCollection")


class EvaluateTraceRequestBody(BaseModel):
    trace_uuid: str = Field(alias="traceUuid")
    metric_collection: str = Field(alias="metricCollection")


class EvaluateSpanRequestBody(BaseModel):
    span_uuid: str = Field(alias="spanUuid")
    metric_collection: str = Field(alias="metricCollection")
