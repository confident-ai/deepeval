from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

from deepeval.red_team.orchestration.strategy.anti_refusal import (
    AntiRefusalStrategy,
)
from deepeval.red_team.orchestration.strategy.encoding import (
    Base64Strategy,
    LeetspeakStrategy,
    ROT13Strategy,
)
from deepeval.red_team.orchestration.strategy.environmental_switch import (
    EnvironmentalSwitchStrategy,
)
from deepeval.red_team.orchestration.strategy.gradual_escalation import (
    GradualEscalationStrategy,
)
from deepeval.red_team.orchestration.strategy.impersonation import (
    ImpersonationStrategy,
)
from deepeval.red_team.orchestration.strategy.strategy import (
    Effectiveness,
    Strategy,
)

STRATEGIES: List[Type[Strategy]] = [
    AntiRefusalStrategy,
    Base64Strategy,
    ROT13Strategy,
    LeetspeakStrategy,
    EnvironmentalSwitchStrategy,
    ImpersonationStrategy,
    GradualEscalationStrategy,
]

_INSTANCES = [cls() for cls in STRATEGIES]
_REGISTRY: Dict[str, Type[Strategy]] = {
    instance.name: cls for instance, cls in zip(_INSTANCES, STRATEGIES)
}


def strategy_catalog(
    exclude: Optional[Iterable[Tuple[str, str]]] = None,
) -> List[Dict[str, Any]]:
    excluded = set(exclude or [])
    levels = [level.value for level in Effectiveness]
    catalog = []
    for instance in _INSTANCES:
        remaining = [
            level for level in levels if (instance.name, level) not in excluded
        ]
        if remaining:
            catalog.append(
                {
                    "name": instance.name,
                    "description": instance.description,
                    "effectiveness": remaining,
                }
            )
    return catalog


def build_strategy(
    name: Optional[str], effectiveness: Optional[str]
) -> Optional[Strategy]:
    cls = _REGISTRY.get(name)
    if cls is None:
        return None
    try:
        level = Effectiveness(effectiveness)
    except ValueError:
        level = Effectiveness.MEDIUM
    return cls(effectiveness=level)
