from typing import List

from deepeval.dataset import ConversationalGolden
from deepeval.simulator.utils import serialize_turns_for_prompt
from deepeval.test_case import Turn
from deepeval.templates import resolve_template

_TEMPLATE_CLASS = "SimulatorInterruptTemplate"
_FEATURE = "simulator"


class SimulatorInterruptTemplate:
    @staticmethod
    def write_interruption(
        *,
        golden: ConversationalGolden,
        turns: List[Turn],
        language: str,
        frustrated: bool = False,
    ) -> str:
        frustration_block = ""
        if frustrated:
            frustration_block = resolve_template(
                _FEATURE,
                _TEMPLATE_CLASS,
                "interruption_frustration",
            ).strip()

        return resolve_template(
            _FEATURE,
            _TEMPLATE_CLASS,
            "write_interruption",
            frustrated=frustrated,
            frustration_block=frustration_block,
            language=language,
            persona=(
                golden.persona.prompt_block()
                if golden.persona is not None
                else ""
            ),
            scenario=golden.scenario or "",
            previous_conversation=serialize_turns_for_prompt(turns),
        )
