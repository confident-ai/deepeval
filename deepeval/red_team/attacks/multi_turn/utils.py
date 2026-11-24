import inspect

from deepeval.models import DeepEvalBaseLLM

from deepteam.attacks import BaseAttack


def enhance_attack(
    attack: BaseAttack, current_attack: str, simulator_model: DeepEvalBaseLLM
):
    sig = inspect.signature(attack.enhance)
    try:
        if "simulator_model" in sig.parameters:
            return attack.enhance(
                attack=current_attack, simulator_model=simulator_model
            )
        return attack.enhance(attack=current_attack)
    except:
        return current_attack


async def a_enhance_attack(
    attack: BaseAttack, current_attack: str, simulator_model: DeepEvalBaseLLM
):
    sig = inspect.signature(attack.enhance)
    try:
        if "simulator_model" in sig.parameters:
            return await attack.a_enhance(
                attack=current_attack, simulator_model=simulator_model
            )
        return await attack.a_enhance(attack=current_attack)
    except:
        return current_attack
