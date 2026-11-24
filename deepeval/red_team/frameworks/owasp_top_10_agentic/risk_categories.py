from typing import List
from deepteam.frameworks.risk_category import RiskCategory
from deepteam.vulnerabilities import (
    Ethics,
    Misinformation,
    PromptLeakage,
    BFLA,
    BOLA,
    RBAC,
    DebugAccess,
    ShellInjection,
    SQLInjection,
    IntellectualProperty,
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
)
from deepteam.attacks.single_turn import (
    Base64,
    PromptInjection,
    PromptProbing,
    Roleplay,
    ROT13,
    ContextPoisoning,
)
from deepteam.attacks.multi_turn import (
    CrescendoJailbreaking,
    LinearJailbreaking,
    TreeJailbreaking,
)

OWASP_ASI_CATEGORIES: List[RiskCategory] = [
    RiskCategory(
        name="ASI_01",
        _display_name="ASI01:2026 Agent Goal Hijack",
        description="Attackers manipulate agent goals, plans, or decision paths through direct or indirect instruction injection, causing agents to pursue unintended or malicious objectives.",
        attacks=[
            PromptInjection(weight=3),
            LinearJailbreaking(weight=2),
            CrescendoJailbreaking(weight=2),
            TreeJailbreaking(weight=1),
            Roleplay(
                weight=2,
                role="System planner",
                persona="Authorized orchestrator",
            ),
        ],
        vulnerabilities=[
            GoalTheft(types=["escalating_probing", "cooperative_dialogue"]),
            ExcessiveAgency(types=["autonomy", "functionality"]),
            Robustness(types=["hijacking"]),
            RecursiveHijacking(
                types=["goal_propagation_attacks", "self_modifying_goals"]
            ),
            IndirectInstruction(
                types=[
                    "cross_context_injection",
                    "document_embedded_instructions",
                    "rag_injection",
                    "tool_output_injection",
                ]
            ),
        ],
    ),
    RiskCategory(
        name="ASI_02",
        _display_name="ASI02:2026 Tool Misuse & Exploitation",
        description="Agents misuse or abuse tools through unsafe composition, recursion, or excessive execution, causing harmful side effects despite valid permissions.",
        attacks=[
            PromptInjection(weight=2),
            Roleplay(
                weight=2, role="Automation agent", persona="Agent orchestrator"
            ),
        ],
        vulnerabilities=[
            ExcessiveAgency(types=["functionality", "permissions"]),
            BFLA(types=["function_bypass", "authorization_bypass"]),
            ToolOrchestrationAbuse(
                types=[
                    "recursive_tool_calls",
                    "unsafe_tool_composition",
                    "tool_budget_exhaustion",
                    "cross_tool_state_leakage",
                ]
            ),
        ],
    ),
    RiskCategory(
        name="ASI_03",
        _display_name="ASI03:2026 Agent Identity & Privilege Abuse",
        description="Abuse of delegated authority, ambiguous agent identity, or trust assumptions leading to unauthorized actions.",
        attacks=[
            Roleplay(
                weight=3,
                role="Privileged agent",
                persona="Authorized priority agent",
            ),
            PromptProbing(weight=2),
        ],
        vulnerabilities=[
            BOLA(
                types=[
                    "object_access_bypass",
                    "unauthorized_object_manipulation",
                ]
            ),
            RBAC(types=["role_bypass", "unauthorized_role_assumption"]),
            PromptLeakage(
                types=[
                    "guard_exposure",
                    "permissions_and_roles",
                    "secrets_and_credentials",
                ]
            ),
            AgentIdentityAbuse(
                types=[
                    "agent_impersonation",
                    "cross_agent_trust_abuse",
                    "identity_inheritance",
                ]
            ),
        ],
    ),
    RiskCategory(
        name="ASI_04",
        _display_name="ASI04:2026 Agentic Supply Chain Compromise",
        description="Compromise of external agents, tools, schemas, or prompts that agents dynamically trust or import.",
        attacks=[
            PromptInjection(weight=2),
        ],
        vulnerabilities=[
            IntellectualProperty(types=["imitation"]),
            ToolMetadataPoisoning(
                types=[
                    "schema_manipulation",
                    "description_deception",
                    "permission_misrepresentation",
                    "registry_poisoning",
                ]
            ),
        ],
    ),
    RiskCategory(
        name="ASI_05",
        _display_name="ASI05:2026 Unexpected Code Execution",
        description="Agent-generated or agent-triggered code execution without sufficient validation or isolation.",
        attacks=[
            PromptInjection(weight=2),
            Base64(weight=2),
            ROT13(weight=2),
        ],
        vulnerabilities=[
            ShellInjection(
                types=["command_injection", "system_command_execution"]
            ),
            SQLInjection(types=["blind_sql_injection"]),
            DebugAccess(
                types=["administrative_interface_exposure", "debug_mode_bypass"]
            ),
            UnexpectedCodeExecution(
                types=[
                    "unauthorized_code_execution",
                    "shell_command_execution",
                    "eval_usage",
                ]
            ),
        ],
    ),
    RiskCategory(
        name="ASI_06",
        _display_name="ASI06:2026 Memory & Context Poisoning",
        description="Injection or leakage of agent memory or contextual state that influences future reasoning or actions.",
        attacks=[
            PromptInjection(weight=2),
            CrescendoJailbreaking(weight=1),
            ContextPoisoning(weight=3),
        ],
        vulnerabilities=[
            Misinformation(
                types=["expertize_misrepresentation", "factual_errors"]
            ),
            Robustness(types=["input_overreliance", "hijacking"]),
        ],
    ),
    RiskCategory(
        name="ASI_07",
        _display_name="ASI07:2026 Insecure Inter-Agent Communication",
        description="Manipulation of messages exchanged between agents, planners, and executors.",
        attacks=[
            Roleplay(
                weight=2, role="Planner", persona="Critical Thinking Agent"
            ),
            LinearJailbreaking(weight=2),
            CrescendoJailbreaking(weight=1),
        ],
        vulnerabilities=[
            InsecureInterAgentCommunication(
                types=[
                    "agent_in_the_middle",
                    "message_injection",
                    "message_spoofing",
                ]
            ),
        ],
    ),
    RiskCategory(
        name="ASI_08",
        _display_name="ASI08:2026 Cascading Agent Failures",
        description="Small agent failures propagate through connected systems causing large-scale impact.",
        attacks=[
            PromptInjection(weight=2),
            LinearJailbreaking(weight=1),
            CrescendoJailbreaking(weight=1),
        ],
        vulnerabilities=[
            Robustness(types=["hijacking"]),
            ExcessiveAgency(types=["autonomy"]),
        ],
    ),
    RiskCategory(
        name="ASI_09",
        _display_name="ASI09:2026 Human-Agent Trust Exploitation",
        description="Exploitation of human over-reliance on agents through misleading explanations or authority framing.",
        attacks=[
            Roleplay(
                weight=3, persona="Authoritative expert", role="Human Agent"
            ),
        ],
        vulnerabilities=[
            Ethics(types=["moral_integrity", "responsible_transparency"]),
            Misinformation(
                types=["expertize_misrepresentation", "unsupported_claims"]
            ),
        ],
    ),
    RiskCategory(
        name="ASI_10",
        _display_name="ASI10:2026 Rogue Agents",
        description="Agents acting beyond intended objectives due to goal drift, collusion, or emergent behavior.",
        attacks=[
            CrescendoJailbreaking(weight=1),
            LinearJailbreaking(weight=2),
        ],
        vulnerabilities=[
            RecursiveHijacking(
                types=["goal_propagation_attacks", "self_modifying_goals"]
            ),
            AutonomousAgentDrift(
                types=[
                    "agent_collusion",
                    "goal_drift",
                    "reward_hacking",
                    "runaway_autonomy",
                ]
            ),
        ],
    ),
]
