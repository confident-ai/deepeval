from typing import Dict, Optional, Union, List
from pydantic import BaseModel, ConfigDict
from deepeval.metrics.base_metric import YesNo


class Knowledge(BaseModel):
    # Each fact’s value is either a string or a list of strings
    # data: Dict[str, Union[str, List[str]]]
    data: Optional[Dict[str, Union[str, List[str]]]] = None
    # Forbid extra top-level fields to satisfy OpenAI’s schema requirements
    model_config = ConfigDict(extra="forbid")


class KnowledgeRetentionVerdict(BaseModel):
    verdict: YesNo
    reason: Optional[str] = None
    model_config = ConfigDict(extra="forbid")


class KnowledgeRetentionScoreReason(BaseModel):
    reason: str
    model_config = ConfigDict(extra="forbid")
