from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator

class ConfidentLlmInputMessage(BaseModel):
    confident_system_message: Optional[str] = Field(default=None, alias="gen_ai.system.message")
    confident_user_message: Optional[str] = Field(default=None, alias="gen_ai.user.message")
    confident_assistant_message: Optional[str] = Field(default=None, alias="gen_ai.assistant.message")
    
    @model_validator(mode='after')
    def validate_exactly_one_message(self) -> 'ConfidentLlmInputMessage':
        fields = [
            self.confident_system_message,
            self.confident_user_message,
            self.confident_assistant_message
        ]
        
        filled_fields = [field for field in fields if field is not None]
        
        if len(filled_fields) != 1:
            raise ValueError("Exactly one of gen_ai_system_message, gen_ai_user_message, or gen_ai_assistant_message must be provided")
            
        return self

class ConfidentLlmOutput(BaseModel):
    confident_assistant_message: Optional[str] = Field(default=None, alias="gen_ai.assistant.message")

class BaseConfidentGenAiOperationSpan(BaseModel):
    # Standard OTel Span Info
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    name: str
    start_time_unix_nano: int
    end_time_unix_nano: int
    status_code: str # e.g., "OK", "ERROR"
    status_message: Optional[str] = None
    span_type: str

class ConfidentLlmSpan(BaseConfidentGenAiOperationSpan):
    # Includes only fields relavant to LlmSpan
    # ref: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
    confident_request_model: Optional[str] = Field(default=None, alias="gen_ai.request.model")
    confident_llm_input_messages: Optional[Union[ConfidentLlmInputMessage, List[ConfidentLlmInputMessage]]] = None
    confident_llm_output: Optional[ConfidentLlmOutput] = None
    confident_usage_input_tokens: Optional[int] = Field(default=None, alias="gen_ai.usage.input_tokens")
    confident_usage_output_tokens: Optional[int] = Field(default=None, alias="gen_ai.usage.output_tokens")
    # unable to find any cost related fields in the semantic conventions
    #TODO: add other model configs to LlmAttributes.model_config





    