from typing import Optional, List, Dict
from pydantic import BaseModel, Field

class APIEvent(BaseModel):
    name: str = Field(..., alias="name")
    model: str
    input: str
    output: str
    retrieval_context: Optional[List[str]] = Field(
        None, alias="retrievalContext"
    )
    completion_time: Optional[float] = Field(None, alias="completionTime")
    token_usage: Optional[float] = Field(None, alias="tokenUsage")
    token_cost: Optional[float] = Field(None, alias="tokenCost")
    distinct_id: Optional[str] = Field(None, alias="distinctId")
    conversation_id: Optional[str] = Field(None, alias="conversationId")
    additional_data: Optional[Dict] = Field(None, alias="additionalData")
