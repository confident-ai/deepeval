from typing import List, Optional

from deepeval.dataset.golden import ConversationalGolden
from deepeval.simulator import (
    ConversationSimulator,
    SimulationGroup,
    SimulationNode,
)
from tests.test_core.test_simulator.helpers import (
    StaticSimulatorModel,
    static_callback,
)


def _says(line: str, **kwargs) -> SimulationNode:
    return SimulationNode(action=lambda: line, name=line, **kwargs)


def _run(
    root: SimulationNode, edge_indices: List[Optional[int]], turns: int = 8
):
    simulator = ConversationSimulator(
        model_callback=static_callback,
        simulator_model=StaticSimulatorModel(edge_indices=edge_indices),
        simulation_graph=root,
        async_mode=False,
    )
    golden = ConversationalGolden(scenario="A scripted call")
    case = simulator.simulate([golden], max_user_simulations=turns)[0]
    said = [turn.content for turn in case.turns if turn.role == "user"]
    return said, case.metadata


def test_an_ending_makes_a_node_terminal_and_is_recorded():
    root = _says("open")
    done = _says("done", ending="resolved")
    root.add_node(done, when="the agent helps")

    said, metadata = _run(root, [1])

    assert done.terminal
    assert said == ["open", "done"]
    assert metadata["Simulation path"] == ["open", "done"]
    assert metadata["Simulation ending"] == "resolved"
    assert metadata["Simulation failed"] is False
    assert metadata["Shortest path"] == 2


def test_a_failing_node_marks_the_conversation_failed():
    root = _says("open")
    gave_up = _says("forget it", ending="gave_up", fails=True)
    done = _says("done", ending="resolved")
    root.add_node(done, when="the agent helps")
    root.add_node(gave_up, when="the agent refuses")

    said, metadata = _run(root, [2])

    assert said == ["open", "forget it"]
    assert metadata["Simulation ending"] == "gave_up"
    assert metadata["Simulation failed"] is True
    assert metadata["Failed nodes"] == ["forget it"]
    assert metadata["Shortest path"] == 2


def test_group_nodes_are_said_in_the_order_asked_and_only_once():
    root = _says("open")
    name = _says("name")
    born = _says("date of birth")
    done = _says("done", ending="resolved")
    group = root.add_group(
        SimulationGroup(
            [(name, "asks for the name"), (born, "asks for the date of birth")]
        )
    )
    group.add_node(done, when="confirms the caller")

    # From open: [name, born]. From born: name was not said, so [name, done].
    # From name: born was said, so only [done].
    said, metadata = _run(root, [2, 1, 1])

    assert said == ["open", "date of birth", "name", "done"]
    assert metadata["Shortest path"] == 4


def test_a_reaction_is_said_then_the_call_resumes_where_it_was():
    root = _says("open")
    done = _says("done", ending="resolved")
    gave_up = _says("forget it", ending="gave_up", fails=True)
    root.add_node(done, when="the agent helps")
    root.add_reaction(
        _says("I already told you", max_visits=1),
        when="the agent asks again",
        then=gave_up,
    )

    # Choices from open: 1 = done, 2 = the reaction.
    said, metadata = _run(root, [2, 1])

    assert said == ["open", "I already told you", "done"]
    assert metadata["Simulation ending"] == "resolved"


def test_a_used_up_reaction_moves_to_its_then_node():
    root = _says("open")
    done = _says("done", ending="resolved")
    gave_up = _says("forget it", ending="gave_up", fails=True)
    root.add_node(done, when="the agent helps")
    root.add_reaction(
        _says("I already told you", max_visits=1),
        when="the agent asks again",
        then=gave_up,
    )

    said, metadata = _run(root, [2, 2])

    assert said == ["open", "I already told you", "forget it"]
    assert metadata["Simulation ending"] == "gave_up"
    assert metadata["Simulation failed"] is True


def test_off_script_line_is_said_then_the_call_ends_off_script():
    root = _says("open")
    done = _says("done", ending="resolved")
    root.add_node(done, when="the agent helps")
    root.off_script(_says("Sorry, can you say that again?", max_visits=2))

    said, metadata = _run(root, [None])

    assert said == [
        "open",
        "Sorry, can you say that again?",
        "Sorry, can you say that again?",
    ]
    assert metadata["Simulation ending"] == "off_script"
    assert metadata["Simulation failed"] is True
