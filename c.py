from deepeval.guardrails import (
    Guardrails,
    PrivacyGuard,
    CybersecurityGuard,
    GuardType,
)
from deepeval.guardrails.cybersecurity_guard import CyberattackType

purpose = "Customer Support Chatbot"

# Initialize the CybersecurityGuard
cybersecurity_guard = CybersecurityGuard(
    purpose=purpose,
    guard_type=GuardType.INPUT,
    vulnerabilities=[
        CyberattackType.SQL_INJECTION,
        CyberattackType.SHELL_INJECTION,
    ],
)


guardrails = Guardrails(guards=[PrivacyGuard(), cybersecurity_guard])


user_input = "Hi my name is alex and I live on Maple Street 123"
output = "I'm sorry but I can't answer this"

guard_results = guardrails.guard_input(user_input)
print(guard_results)
