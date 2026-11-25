import re
from typing import get_args

from deepeval.red_team.guardrails.guards.schema import SafetyLevelSchema
from deepeval.red_team.guardrails.guards.prompt_injection_guard.template import (
    PromptInjectionGuardTemplate,
)
from deepeval.red_team.guardrails.guards.toxicity_guard.template import (
    ToxicityGuardTemplate,
)
from deepeval.red_team.guardrails.guards.illegal_guard.template import (
    IllegalGuardTemplate,
)
from deepeval.red_team.guardrails.guards.cybersecurity_guard.template import (
    CybersecurityGuardTemplate,
)
from deepeval.red_team.guardrails.guards.hallucination_guard.template import (
    HallucinationGuardTemplate,
)
from deepeval.red_team.guardrails.guards.privacy_guard.template import (
    PrivacyGuardTemplate,
)
from deepeval.red_team.guardrails.guards.topical_guard.template import (
    TopicalGuardTemplate,
)


# Allowed safety levels are derived from the schema the guards parse into,
# so this test stays in sync if SafetyLevelSchema ever changes.
ALLOWED_LEVELS = set(
    get_args(SafetyLevelSchema.model_fields["safety_level"].annotation)
)

# Every rendered guard prompt instructs the model to answer in the form
# {"safety_level": "safe"/"unsafe"/"borderline", ...}. This captures that
# slash-separated list of quoted levels.
_LEVELS_PATTERN = re.compile(r'"safety_level":\s*((?:"[a-z]+"/?)+)')


def _instructed_levels(prompt: str) -> set:
    levels = set()
    for group in _LEVELS_PATTERN.findall(prompt):
        levels.update(re.findall(r'"([a-z]+)"', group))
    return levels


# (name, rendered input prompt, rendered output prompt) for every guard.
RENDERED_PROMPTS = [
    (
        "prompt_injection",
        PromptInjectionGuardTemplate.judge_input_prompt("hello"),
        PromptInjectionGuardTemplate.judge_output_prompt("hello", "world"),
    ),
    (
        "toxicity",
        ToxicityGuardTemplate.judge_input_prompt("hello"),
        ToxicityGuardTemplate.judge_output_prompt("hello", "world"),
    ),
    (
        "illegal",
        IllegalGuardTemplate.judge_input_prompt("hello"),
        IllegalGuardTemplate.judge_output_prompt("hello", "world"),
    ),
    (
        "hallucination",
        HallucinationGuardTemplate.judge_input_prompt("hello"),
        HallucinationGuardTemplate.judge_output_prompt("hello", "world"),
    ),
    (
        "privacy",
        PrivacyGuardTemplate.judge_input_prompt("hello"),
        PrivacyGuardTemplate.judge_output_prompt("hello", "world"),
    ),
    (
        "cybersecurity",
        CybersecurityGuardTemplate.judge_input_prompt("hello", ["sql"]),
        CybersecurityGuardTemplate.judge_output_prompt(
            "hello", "world", ["sql"]
        ),
    ),
    (
        "topical",
        TopicalGuardTemplate.judge_input_prompt("hello", ["weather"]),
        TopicalGuardTemplate.judge_output_prompt("hello", "world", ["weather"]),
    ),
]


class TestGuardTemplateSafetyLevels:

    def test_instructed_levels_match_schema(self):
        for name, input_prompt, output_prompt in RENDERED_PROMPTS:
            for prompt in (input_prompt, output_prompt):
                instructed = _instructed_levels(prompt)
                assert instructed, f"{name}: no safety_level format found"
                invalid = instructed - ALLOWED_LEVELS
                assert not invalid, (
                    f"{name}: prompt instructs levels {invalid} that are not "
                    f"in SafetyLevelSchema {ALLOWED_LEVELS}"
                )

    def test_no_uncertain_level(self):
        # "uncertain" was previously instructed by several guards but is not a
        # valid SafetyLevelSchema value, so the borderline tier was unreachable.
        for name, input_prompt, output_prompt in RENDERED_PROMPTS:
            assert "uncertain" not in input_prompt, name
            assert "uncertain" not in output_prompt, name
