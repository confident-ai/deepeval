from typing import Dict, List
from pydantic import BaseModel

from .base_vulnerability import BaseVulnerability
from .bias.bias import Bias
from .toxicity.toxicity import Toxicity
from .misinformation.misinformation import Misinformation
from .illegal_activity.illegal_activity import IllegalActivity
from .prompt_leakage.prompt_leakage import PromptLeakage
from .pii_leakage.pii_leakage import PIILeakage
from .bfla.bfla import BFLA
from .bola.bola import BOLA
from .rbac.rbac import RBAC
from .debug_access.debug_access import DebugAccess
from .shell_injection.shell_injection import ShellInjection
from .sql_injection.sql_injection import SQLInjection
from .ssrf.ssrf import SSRF
from .intellectual_property.intellectual_property import IntellectualProperty
from .competition.competition import Competition
from .graphic_content.graphic_content import GraphicContent
from .personal_safety.personal_safety import PersonalSafety
from .goal_theft.goal_theft import GoalTheft
from .recursive_hijacking.recursive_hijacking import RecursiveHijacking
from .robustness.robustness import Robustness
from .excessive_agency.excessive_agency import ExcessiveAgency
from .indirect_instruction.indirect_instruction import IndirectInstruction
from .tool_orchestration_abuse.tool_orchestration_abuse import (
    ToolOrchestrationAbuse,
)
from .agent_identity_abuse.agent_identity_abuse import AgentIdentityAbuse
from .tool_metadata_poisoning.tool_metadata_poisoning import (
    ToolMetadataPoisoning,
)
from .unexpected_code_execution.unexpected_code_execution import (
    UnexpectedCodeExecution,
)
from .insecure_inter_agent_communication.insecure_inter_agent_communication import (
    InsecureInterAgentCommunication,
)
from .autonomous_agent_drift.autonomous_agent_drift import AutonomousAgentDrift
from .exploit_tool_agent.exploit_tool_agent import ExploitToolAgent
from .external_system_abuse.external_system_abuse import ExternalSystemAbuse
from .cross_context_retrieval.cross_context_retrieval import (
    CrossContextRetrieval,
)
from .system_reconnaissance.system_reconnaissance import SystemReconnaissance
from .child_protection.child_protection import ChildProtection
from .fairness.fairness import Fairness
from .ethics.ethics import Ethics

# Import types
from .bias.types import BiasType
from .toxicity.types import ToxicityType
from .misinformation.types import MisinformationType
from .illegal_activity.types import IllegalActivityType
from .prompt_leakage.types import PromptLeakageType
from .pii_leakage.types import PIILeakageType
from .bfla.types import BFLAType
from .bola.types import BOLAType
from .rbac.types import RBACType
from .debug_access.types import DebugAccessType
from .shell_injection.types import ShellInjectionType
from .sql_injection.types import SQLInjectionType
from .ssrf.types import SSRFType
from .intellectual_property.types import IntellectualPropertyType
from .competition.types import CompetitionType
from .graphic_content.types import GraphicContentType
from .personal_safety.types import PersonalSafetyType
from .goal_theft.types import GoalTheftType
from .recursive_hijacking.types import RecursiveHijackingType
from .robustness.types import RobustnessType
from .excessive_agency.types import ExcessiveAgencyType
from .indirect_instruction.indirect_instruction import IndirectInstructionType
from .tool_orchestration_abuse.tool_orchestration_abuse import (
    ToolOrchestrationAbuseType,
)
from .agent_identity_abuse.agent_identity_abuse import AgentIdentityAbuseType
from .tool_metadata_poisoning.tool_metadata_poisoning import (
    ToolMetadataPoisoningType,
)
from .unexpected_code_execution.unexpected_code_execution import (
    UnexpectedCodeExecutionType,
)
from .insecure_inter_agent_communication.insecure_inter_agent_communication import (
    InsecureInterAgentCommunicationType,
)
from .autonomous_agent_drift.autonomous_agent_drift import (
    AutonomousAgentDriftType,
)
from .exploit_tool_agent.types import ExploitToolAgentType
from .external_system_abuse.types import ExternalSystemAbuseType
from .cross_context_retrieval.types import CrossContextRetrievalType
from .system_reconnaissance.types import SystemReconnaissanceType
from .hallucination.hallucination import Hallucination
from .hallucination.types import HallucinationType
from .child_protection.types import ChildProtectionType
from .fairness.types import FairnessType
from .ethics.types import EthicsType

VULNERABILITY_CLASSES_MAP: Dict[str, BaseVulnerability] = {
    v.name: v
    for v in [
        Bias,
        Toxicity,
        ChildProtection,
        Ethics,
        Fairness,
        Misinformation,
        IllegalActivity,
        PromptLeakage,
        PIILeakage,
        BFLA,
        BOLA,
        RBAC,
        DebugAccess,
        ShellInjection,
        SQLInjection,
        SSRF,
        IntellectualProperty,
        Competition,
        GraphicContent,
        PersonalSafety,
        GoalTheft,
        RecursiveHijacking,
        Robustness,
        ExcessiveAgency,
        IndirectInstruction,
        ToolOrchestrationAbuse,
        AgentIdentityAbuse,
        ToolMetadataPoisoning,
        UnexpectedCodeExecution,
        InsecureInterAgentCommunication,
        AutonomousAgentDrift,
        ExploitToolAgent,
        ExternalSystemAbuse,
        CrossContextRetrieval,
        SystemReconnaissance,
        Hallucination,
    ]
}


class VulnerabilityInfo(BaseModel):
    description: str
    allowed_types: List[str]
    category: str


VULNERABILITY_INFO_MAP: Dict[str, VulnerabilityInfo] = {
    name: VulnerabilityInfo(
        description=vuln_class.description,
        allowed_types=vuln_class.ALLOWED_TYPES,
        category=vuln_class.category,
    )
    for name, vuln_class in VULNERABILITY_CLASSES_MAP.items()
}
