from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Any

class BatchFinding(BaseModel):
    spanUuid: str = Field(
        ...,
        description="The UUID of the highest attributable span within the batch where the vulnerability emerged."
    )
    vulnerability: str = Field(
        ...,
        description="The overarching vulnerability category (e.g., 'Bias', 'PII Leakage')."
    )
    vulnerabilityType: str = Field(
        ...,
        description="The specific subtype of the vulnerability (e.g., 'gender', 'race')."
    )
    reason: str = Field(
        ...,
        description="Detailed explanation of why this span is considered a breach, or if a child's breach was materialized."
    )
    outcome: Literal["materialized", "mitigated", "attempted"] = Field(
        ...,
        description="Resolution: 'materialized' (leaked to user), 'mitigated' (caught by guardrail), or 'attempted' (no breach)."
    )

class BatchFindingsList(BaseModel):
    findings: List[BatchFinding]

class SpanNode(BaseModel):
    """Strictly types the extracted span data before it goes into the LLM batch."""
    spanUuid: str
    parentUuid: Optional[str] = None
    type: str
    name: Optional[str] = None
    status: Optional[str] = None
    
    # Core I/O
    input: Optional[Any] = None
    output: Optional[Any] = None
    error: Optional[str] = None
    
    # Context & Tools
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None
    tools_called: Optional[List[Any]] = None
    expected_output: Optional[str] = None
    
    # Subclass-specific fields (Agent, LLM, Retriever, Tool)
    model: Optional[str] = None
    available_tools: Optional[List[str]] = None
    agent_handoffs: Optional[List[str]] = None
    embedder: Optional[str] = None
    description: Optional[str] = None
    
    # Bottom-up propagation
    child_findings: Optional[List[BatchFinding]] = None

    previous_findings: Optional[List[BatchFinding]] = None