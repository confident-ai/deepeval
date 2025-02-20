from deepeval.guardrails import Guardrails
from deepeval.guardrails import PrivacyGuard

from deepeval.guardrails import CybersecurityGuard
from deepeval.guardrails.cybersecurity_guard import CyberattackCategory


guardrails = Guardrails(guards=[PrivacyGuard()])
guard_result = guardrails.guard_input(input="Hey doesn't t his look nice?")
