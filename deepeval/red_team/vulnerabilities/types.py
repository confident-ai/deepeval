from typing import Union

from deepeval.red_team._metrics.intellectual_property.template import (
    IntellectualPropertyTemplate,
)
from deepeval.red_team.vulnerabilities.bias.template import BiasTemplate
from deepeval.red_team.vulnerabilities.competition.template import (
    CompetitionTemplate,
)
from deepeval.red_team.vulnerabilities.graphic_content.template import (
    GraphicContentTemplate,
)
from deepeval.red_team.vulnerabilities.illegal_activity.template import (
    IllegalActivityTemplate,
)
from deepeval.red_team.vulnerabilities.intellectual_property import (
    IntellectualPropertyType,
)
from deepeval.red_team.vulnerabilities.misinformation.template import (
    MisinformationTemplate,
)
from deepeval.red_team.vulnerabilities.personal_safety.template import (
    PersonalSafetyTemplate,
)
from deepeval.red_team.vulnerabilities.pii_leakage.template import (
    PIILeakageTemplate,
)
from deepeval.red_team.vulnerabilities.prompt_leakage.template import (
    PromptLeakageTemplate,
)
from deepeval.red_team.vulnerabilities.toxicity.template import ToxicityTemplate
from deepeval.red_team.vulnerabilities.illegal_activity import (
    IllegalActivityType,
)
from deepeval.red_team.vulnerabilities.personal_safety import PersonalSafetyType
from deepeval.red_team.vulnerabilities.graphic_content import GraphicContentType
from deepeval.red_team.vulnerabilities.misinformation import MisinformationType
from deepeval.red_team.vulnerabilities.prompt_leakage import PromptLeakageType
from deepeval.red_team.vulnerabilities.competition import CompetitionType
from deepeval.red_team.vulnerabilities.pii_leakage import PIILeakageType
from deepeval.red_team.vulnerabilities.toxicity import ToxicityType
from deepeval.red_team.vulnerabilities.bias import BiasType
from deepeval.red_team.vulnerabilities.rbac import RBACType
from deepeval.red_team.vulnerabilities.bola.types import BOLAType
from deepeval.red_team.vulnerabilities.bfla.types import BFLAType
from deepeval.red_team.vulnerabilities.ssrf.types import SSRFType
from deepeval.red_team.vulnerabilities.debug_access.types import DebugAccessType
from deepeval.red_team.vulnerabilities.shell_injection.types import (
    ShellInjectionType,
)
from deepeval.red_team.vulnerabilities.sql_injection.types import (
    SQLInjectionType,
)
from deepeval.red_team.vulnerabilities.xss.types import XSSType
from deepeval.red_team.vulnerabilities.path_traversal.types import (
    PathTraversalType,
)
from deepeval.red_team.vulnerabilities.rbac.template import RBACTemplate
from deepeval.red_team.vulnerabilities.bola.template import BOLATemplate
from deepeval.red_team.vulnerabilities.bfla.template import BFLATemplate
from deepeval.red_team.vulnerabilities.ssrf.template import SSRFTemplate
from deepeval.red_team.vulnerabilities.xss.template import XSSTemplate
from deepeval.red_team.vulnerabilities.path_traversal.template import (
    PathTraversalTemplate,
)
from deepeval.red_team.vulnerabilities.debug_access.template import (
    DebugAccessTemplate,
)
from deepeval.red_team.vulnerabilities.shell_injection.template import (
    ShellInjectionTemplate,
)
from deepeval.red_team.vulnerabilities.sql_injection.template import (
    SQLInjectionTemplate,
)
from deepeval.red_team.vulnerabilities.robustness import RobustnessType
from deepeval.red_team.vulnerabilities.robustness.template import (
    RobustnessTemplate,
)
from deepeval.red_team.vulnerabilities.excessive_agency import (
    ExcessiveAgencyType,
)
from deepeval.red_team.vulnerabilities.excessive_agency.template import (
    ExcessiveAgencyTemplate,
)
from deepeval.red_team.vulnerabilities.exploit_tool_agent.types import (
    ExploitToolAgentType,
)
from deepeval.red_team.vulnerabilities.exploit_tool_agent.template import (
    ExploitToolAgentTemplate,
)
from deepeval.red_team.vulnerabilities.external_system_abuse.types import (
    ExternalSystemAbuseType,
)
from deepeval.red_team.vulnerabilities.external_system_abuse.template import (
    ExternalSystemAbuseTemplate,
)
from deepeval.red_team.vulnerabilities.cross_context_retrieval.types import (
    CrossContextRetrievalType,
)
from deepeval.red_team.vulnerabilities.cross_context_retrieval.template import (
    CrossContextRetrievalTemplate,
)
from deepeval.red_team.vulnerabilities.system_reconnaissance.types import (
    SystemReconnaissanceType,
)
from deepeval.red_team.vulnerabilities.system_reconnaissance.template import (
    SystemReconnaissanceTemplate,
)
from deepeval.red_team.vulnerabilities.mcp_server_trust_exploitation.types import (
    MCPServerTrustExploitationType,
)
from deepeval.red_team.vulnerabilities.mcp_server_trust_exploitation.template import (
    MCPServerTrustExploitationTemplate,
)

# Import agentic vulnerability types
from deepeval.red_team.vulnerabilities.goal_theft.types import GoalTheftType
from deepeval.red_team.vulnerabilities.recursive_hijacking.types import (
    RecursiveHijackingType,
)
from deepeval.red_team.vulnerabilities.goal_theft.template import (
    GoalTheftTemplate,
)
from deepeval.red_team.vulnerabilities.recursive_hijacking.template import (
    RecursiveHijackingTemplate,
)

VulnerabilityType = Union[
    IllegalActivityType,
    PersonalSafetyType,
    GraphicContentType,
    MisinformationType,
    PromptLeakageType,
    PromptLeakageType,
    CompetitionType,
    PIILeakageType,
    ToxicityType,
    BiasType,
    IntellectualPropertyType,
    IntellectualPropertyType,
    IntellectualPropertyType,
    RBACType,
    BOLAType,
    BFLAType,
    SSRFType,
    PathTraversalType,
    DebugAccessType,
    ShellInjectionType,
    SQLInjectionType,
    XSSType,
    ExploitToolAgentType,
    ExternalSystemAbuseType,
    CrossContextRetrievalType,
    SystemReconnaissanceType,
    MCPServerTrustExploitationType,
    # Restored vulnerability types
    RobustnessType,
    ExcessiveAgencyType,
    # Agentic vulnerability types
    GoalTheftType,
    RecursiveHijackingType,
]

TemplateType = Union[
    BiasTemplate,
    CompetitionTemplate,
    GraphicContentTemplate,
    IllegalActivityTemplate,
    IntellectualPropertyTemplate,
    MisinformationTemplate,
    PersonalSafetyTemplate,
    PIILeakageTemplate,
    PromptLeakageTemplate,
    ToxicityTemplate,
    RBACTemplate,
    BOLATemplate,
    BFLATemplate,
    SSRFTemplate,
    PathTraversalTemplate,
    DebugAccessTemplate,
    ShellInjectionTemplate,
    SQLInjectionTemplate,
    XSSTemplate,
    RobustnessTemplate,
    ExcessiveAgencyTemplate,
    GoalTheftTemplate,
    RecursiveHijackingTemplate,
    ExploitToolAgentTemplate,
    ExternalSystemAbuseTemplate,
    CrossContextRetrievalTemplate,
    SystemReconnaissanceTemplate,
    MCPServerTrustExploitationTemplate,
]
