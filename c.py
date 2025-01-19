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


guardrails = Guardrails(guards=[PrivacyGuard()])


user_input = "Hi I'm here to return an order."
output = "Sure! What do you want returned?"

guard_results = guardrails.guard_input(user_input)
print(guard_results)
