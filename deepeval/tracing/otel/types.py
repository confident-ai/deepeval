from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator

class ConfidentSpanType(Enum):
    CHAT = "chat" # gen_ai.operation.name
    GENERATE_CONTENT = "generate_content" # gen_ai.operation.name
    TEXT_COMPLETION = "text_completion" # gen_ai.operation.name
    TOOL = "tool" # confident.tool.name
    AGENT = "agent" # confident.agent.name
    RETRIEVER = "retriever" # confident.retriever.name

class BaseConfidentSpan(BaseModel):
    # Standard OTel Span Info
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    name: str
    start_time_unix_nano: Optional[int] = None
    end_time_unix_nano: Optional[int] = None
    status_code: str # e.g., "OK", "ERROR"
    status_message: Optional[str] = None
    
class ConfidentLlmInputMessage(BaseModel):
    gen_ai_system_message: Optional[str] = Field(default=None) # event(gen_ai.system.message).attributes -> convert to strigified json
    gen_ai_user_message: Optional[str] = Field(default=None) # event(gen_ai.user.message).attributes -> convert to strigified json
    gen_ai_assistant_message: Optional[str] = Field(default=None) # event(gen_ai.assistant.message).attributes -> convert to strigified json
    gen_ai_tool_message: Optional[str] = Field(default=None) # event(gen_ai.tool.message).attributes -> convert to strigified json
    gen_ai_choice: Optional[str] = Field(default=None) # event(gen_ai.choice).attributes -> convert to strigified json
    
    @model_validator(mode='after')
    def validate_exactly_one_message(self) -> 'ConfidentLlmInputMessage':
        fields = [
            self.gen_ai_system_message,
            self.gen_ai_user_message,
            self.gen_ai_assistant_message, 
            self.gen_ai_tool_message,
            self.gen_ai_choice
        ]
        
        filled_fields = [field for field in fields if field is not None]
        
        if len(filled_fields) != 1:
            raise ValueError("Exactly one of gen_ai_system_message, gen_ai_user_message, or gen_ai_assistant_message must be provided")
            
        return self

class ConfidentLlmOutput(BaseModel):
    gen_ai_assistant_message: Optional[str] = Field(default=None) # event(gen_ai.assistant.message).attributes -> convert to strigified json
    gen_ai_tool_message: Optional[str] = Field(default=None) # event(gen_ai.tool.message).attributes -> convert to strigified json
    gen_ai_choice: Optional[str] = Field(default=None) # event(gen_ai.choice).attributes -> convert to strigified json

    @model_validator(mode='after')
    def validate_exactly_one_message(self) -> 'ConfidentLlmOutput':
        fields = [
            self.gen_ai_assistant_message,
            self.gen_ai_tool_message,
            self.gen_ai_choice
        ]
        filled_fields = [field for field in fields if field is not None]
        if len(filled_fields) != 1:
            raise ValueError("Exactly one of gen_ai_assistant_message, gen_ai_tool_message, or gen_ai_choice must be provided")
        return self

class ConfidentLlmSpan(BaseConfidentSpan):
    # Includes only fields relavant to LlmSpan
    gen_ai_request_model: Optional[str] = Field(default=None) # gen_ai.request.model
    gen_ai_llm_input_messages: Optional[Union[ConfidentLlmInputMessage, List[ConfidentLlmInputMessage]]] = None # iterate over events
    gen_ai_llm_output: Optional[ConfidentLlmOutput] = None # if last this is the last assistant event
    gen_ai_usage_input_tokens: Optional[int] = Field(default=None) # gen_ai.usage.input_tokens
    gen_ai_usage_output_tokens: Optional[int] = Field(default=None) # gen_ai.usage.output_tokens

# Types for Confident specific signals
class ConfidentToolSpan(BaseConfidentSpan):
    # Includes only fields relavant to ToolSpan
    name: Optional[str] = Field(default=None) # confident.tool.name (custom attribute)
    description: Optional[str] = Field(default=None) # confident.tool.description (custom attribute)
    input: Optional[str] = Field(default=None) # confident.tool.input (custom attribute)
    output: Optional[str] = Field(default=None) # confident.tool.output (custom attribute)

class ConfidentAgentSpan(BaseConfidentSpan):
    # Includes only fields relavant to AgentSpan
    name: Optional[str] = Field(default=None) # confident.agent.name (custom attribute)
    input: Optional[str] = Field(default=None) # confident.agent.input (custom attribute)
    output: Optional[str] = Field(default=None) # confident.agent.output (custom attribute)
    available_tools: Optional[List[str]] = Field(default=None) # confident.agent.available_tools (custom attribute)
    agent_handoffs: Optional[List[str]] = Field(default=None) # confident.agent.agent_handoffs (custom attribute)


class ConfidentRetrieverSpan(BaseConfidentSpan):
    # Includes only fields relavant to RetrieverSpan
    name: Optional[str] = Field(default=None) # confident.retriever.name (custom attribute)
    input: Optional[str] = Field(default=None) # confident.retriever.input (custom attribute)``
    embedder: Optional[str] = Field(default=None) # gen_ai.request.model
    top_k: Optional[int] = Field(default=None) # confident.retriever.top_k (custom attribute)
    chunk_size: Optional[int] = Field(default=None) # confident.retriever.chunk_size (custom attribute)
        

    