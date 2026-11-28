from enum import Enum
from typing import Literal


class IndirectInstructionType(Enum):
    RAG_INJECTION = "rag_injection"
    TOOL_OUTPUT_INJECTION = "tool_output_injection"
    DOCUMENT_EMBEDDED_INSTRUCTIONS = "document_embedded_instructions"
    CROSS_CONTENT_INJECTION = "cross_context_injection"


IndirectInstructionTypes = Literal[
    IndirectInstructionType.RAG_INJECTION.value,
    IndirectInstructionType.TOOL_OUTPUT_INJECTION.value,
    IndirectInstructionType.DOCUMENT_EMBEDDED_INSTRUCTIONS.value,
    IndirectInstructionType.CROSS_CONTENT_INJECTION.value,
]
