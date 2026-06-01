from typing import Optional, Type

from deepeval.simulator.simulation_graph.node import SimulationNode
from deepeval.simulator.template import SimulationTemplate
from deepeval.simulator.utils import validate_simulation_template


def default_simulation_node(
    *,
    template: Optional[Type[SimulationTemplate]] = None,
    terminal: bool = False,
    max_visits: Optional[int] = None,
    name: str = "default",
) -> SimulationNode:
    """Returns a fresh `SimulationNode` whose action calls today's
    `simulator_model` + `SimulationTemplate` path.

    Args:
        template: Optional subclass of `SimulationTemplate` used
            to render the user-turn prompt. When omitted, the built-in
            `SimulationTemplate` is used. The template is
            validated at construction time.
        terminal: If True, the simulation ends immediately after this node
            emits a user turn and the assistant replies.
        max_visits: Optional emission cap (see `SimulationNode`).
        name: Optional debug name.

    Use cases:
    - As the implicit root when no `simulation_graph` is passed
      to `ConversationSimulator` (constructed internally).
    - As a composable building block inside a custom graph, e.g.
      `my_root.add_node(default_simulation_node(), when="The assistant asked a
      clarifying question")` to delegate one branch to the LLM.
    - To customize the user-turn prompt:
      `default_simulation_node(template=MyTemplate)`.
    """
    if template is not None:
        validate_simulation_template(template)
    effective_template: Type[SimulationTemplate] = (
        template or SimulationTemplate
    )

    async def _default_user_action(simulator, turns, golden):
        if len(turns) == 0:
            return await simulator.a_generate_first_user_input(
                golden, template=effective_template
            )
        return await simulator.a_generate_next_user_input(
            golden, turns, template=effective_template
        )

    return SimulationNode(
        action=_default_user_action,
        terminal=terminal,
        max_visits=max_visits,
        name=name,
    )
