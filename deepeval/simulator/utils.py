import inspect
from typing import TYPE_CHECKING, List, Type

from deepeval.test_case import Turn
from deepeval.utils import serialize_to_json

if TYPE_CHECKING:
    from deepeval.simulator.template import SimulationTemplate


def serialize_turns_for_prompt(turns: List[Turn]) -> str:
    """Serialize conversation state without embedding voice audio bytes."""
    prompt_turns = [turn.model_dump_for_prompt() for turn in turns]
    return serialize_to_json(prompt_turns, indent=4, ensure_ascii=False)


def validate_simulation_template(
    simulation_template: Type["SimulationTemplate"],
):
    from deepeval.simulator.template import SimulationTemplate

    if not issubclass(simulation_template, SimulationTemplate):
        raise TypeError(
            "simulation_template must inherit from " "SimulationTemplate."
        )

    expected_signatures = {
        "simulate_first_user_turn": {
            "args": ["golden", "language"],
            "signature": (
                "simulate_first_user_turn("
                "golden: ConversationalGolden, language: str"
                ") -> str"
            ),
        },
        "simulate_user_turn": {
            "args": ["golden", "turns", "language"],
            "signature": (
                "simulate_user_turn("
                "golden: ConversationalGolden, turns: List[Turn], "
                "language: str"
                ") -> str"
            ),
        },
    }

    for method_name, expected_signature in expected_signatures.items():
        expected_args = expected_signature["args"]
        expected_signature_text = expected_signature["signature"]
        method = getattr(simulation_template, method_name, None)
        if method is None:
            raise TypeError(
                "simulation_template must define "
                f"`{expected_signature_text}`."
            )

        parameters = list(inspect.signature(method).parameters.values())
        positional_parameters = [
            parameter
            for parameter in parameters
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        actual_args = [
            parameter.name
            for parameter in positional_parameters[: len(expected_args)]
        ]
        if actual_args != expected_args:
            raise TypeError(
                f"simulation_template `{method_name}` must accept the "
                f"correct arguments: `{expected_signature_text}`."
            )
