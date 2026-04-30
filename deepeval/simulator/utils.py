import inspect
from typing import Type

from deepeval.simulator.template import ConversationSimulatorTemplate


def validate_simulation_template(
    simulation_template: Type[ConversationSimulatorTemplate],
):
    if not issubclass(simulation_template, ConversationSimulatorTemplate):
        raise TypeError(
            "simulation_template must inherit from "
            "ConversationSimulatorTemplate."
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
