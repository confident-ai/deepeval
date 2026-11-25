import os
from typing import List, Type

from deepeval.red_team.vulnerabilities import (
    BFLA,
    BOLA,
    RBAC,
    SSRF,
    DebugAccess,
    ExcessiveAgency,
    IndirectInstruction,
    PIILeakage,
    PromptLeakage,
    SQLInjection,
    ShellInjection,
    ToolMetadataPoisoning,
    ToolOrchestrationAbuse,
    UnexpectedCodeExecution,
)
from deepeval.red_team.vulnerabilities.base_vulnerability import (
    BaseVulnerability,
)

CODE_CONTEXT_LIMIT = int(os.getenv("CODE_CONTEXT_LIMIT", "40000"))

# Default categories the code scanner looks for: a curated subset of the red teaming
# taxonomy whose risk is visible in source code.
DEFAULT_CODE_SCAN_VULNERABILITIES: List[Type[BaseVulnerability]] = [
    UnexpectedCodeExecution,
    ShellInjection,
    SQLInjection,
    SSRF,
    ExcessiveAgency,
    IndirectInstruction,
    ToolMetadataPoisoning,
    ToolOrchestrationAbuse,
    BFLA,
    BOLA,
    RBAC,
    DebugAccess,
    PromptLeakage,
    PIILeakage,
]
