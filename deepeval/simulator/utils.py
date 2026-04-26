import inspect
from typing import Type

from deepeval.dataset import ConversationalGolden
from deepeval.simulator.template import ConversationSimulatorTemplate
from deepeval.test_case import Turn


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


def dump_conversational_golden(golden: ConversationalGolden):
    new_golden = ConversationalGolden(
        scenario=golden.scenario,
        expected_outcome=golden.expected_outcome,
        user_description=golden.user_description,
        context=golden.context,
        turns=(
            [
                Turn(
                    role=turn.role,
                    content=turn.content,
                    user_id=turn.user_id,
                    retrieval_context=turn.retrieval_context,
                    tools_called=turn.tools_called,
                )
                for turn in golden.turns
            ]
            if golden.turns is not None
            else None
        ),
    )
    try:
        body = new_golden.model_dump(
            by_alias=True,
            exclude_none=True,
            exclude={"turns": {"__all__": {"_mcp_interaction"}}},
        )
    except AttributeError:
        body = new_golden.dict(
            by_alias=True,
            exclude_none=True,
            exclude={"turns": {"__all__": {"_mcp_interaction"}}},
        )
    return {"conversationalGolden": body}
