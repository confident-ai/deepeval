from typing import Dict, List

from deepeval.dataset import ConversationalGolden
from deepeval.simulator.utils import serialize_turns_for_prompt
from deepeval.test_case import Turn
from deepeval.templates import SimulatorTemplateMethod, resolve_template
from deepeval.voice.interruption import InterruptionLevel

_TEMPLATE_CLASS = "SimulatorInterruptTemplate"
_FEATURE = "simulator"

_BIAS_METHODS: Dict[InterruptionLevel, SimulatorTemplateMethod] = {
    "rare": "interruption_bias_rare",
    "normal": "interruption_bias_normal",
    "frequent": "interruption_bias_frequent",
}


class SimulatorInterruptTemplate:
    @staticmethod
    def interruption_bias(level: InterruptionLevel) -> str:
        return resolve_template(
            _FEATURE,
            _TEMPLATE_CLASS,
            _BIAS_METHODS[level],
        ).strip()

    @staticmethod
    def decide_interrupt(
        *,
        golden: ConversationalGolden,
        turns: List[Turn],
        partial_agent_transcript: str,
        interruption_level: InterruptionLevel,
        language: str,
        frustrated: bool = False,
    ) -> str:
        previous_conversation = serialize_turns_for_prompt(turns)
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
            "decide_interrupt",
            interruption_level=interruption_level,
            prompt_bias=SimulatorInterruptTemplate.interruption_bias(
                interruption_level
            ),
            frustrated=frustrated,
            frustration_block=frustration_block,
            language=language,
            persona=(
                golden.persona.prompt_block()
                if golden.persona is not None
                else ""
            ),
            scenario=golden.scenario or "",
            previous_conversation=previous_conversation,
            partial_agent_transcript=partial_agent_transcript,
        )
