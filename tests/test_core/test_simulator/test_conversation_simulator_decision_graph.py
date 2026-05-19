from typing import List, Optional

import pytest

from deepeval.dataset.golden import ConversationalGolden
from deepeval.simulator import (
    ConversationSimulator,
    SimulationNode,
    default_simulation_node,
)
from deepeval.test_case.conversational_test_case import Turn
from tests.test_core.test_simulator.helpers import (
    StaticSimulatorModel,
    async_static_callback,
    static_callback,
)


# ---------------------------------------------------------------------------
# Builder / validation tests
# ---------------------------------------------------------------------------


def test_add_node_rejects_empty_when():
    node = SimulationNode(action=lambda: "hi", name="root")
    other = SimulationNode(action=lambda: "child", name="other")

    with pytest.raises(ValueError):
        node.add_node(other, when="")

    with pytest.raises(ValueError):
        node.add_node(other, when="   ")


def test_add_node_returns_child_for_chaining():
    root = SimulationNode(action=lambda: "a", name="root")
    child = SimulationNode(action=lambda: "b", name="child")

    returned = root.add_node(child, when="moves on")

    assert returned is child


def test_add_node_rejects_non_string_when():
    node = SimulationNode(action=lambda: "hi", name="root")
    other = SimulationNode(action=lambda: "child", name="other")

    with pytest.raises(ValueError):
        node.add_node(other, when=None)  # type: ignore[arg-type]


def test_max_visits_must_be_positive():
    with pytest.raises(ValueError):
        SimulationNode(action=lambda: "x", max_visits=0)


# ---------------------------------------------------------------------------
# Default behavior (no graph passed) regression
# ---------------------------------------------------------------------------


def test_default_graph_does_not_classify_edges():
    """When no `simulation_graph` is passed the default node has no
    edges, so the runner must never invoke the EdgeChoice classifier."""
    golden = ConversationalGolden(
        scenario="Default behavior",
        expected_outcome=None,
        user_description="Default User",
    )
    simulator_model = StaticSimulatorModel()
    simulator = ConversationSimulator(
        model_callback=static_callback,
        simulator_model=simulator_model,
        async_mode=False,
    )

    cases = simulator.simulate([golden], max_user_simulations=2)

    tc = cases[0]
    assert len(tc.turns) == 4
    assert tc.turns[0].role == "user"
    assert tc.turns[1].role == "assistant"
    assert "EdgeChoice" not in simulator_model.schema_calls
    # User turns are produced via the existing SimulatedInput path.
    assert simulator_model.schema_calls.count("SimulatedInput") == 2


# ---------------------------------------------------------------------------
# Terminal nodes
# ---------------------------------------------------------------------------


def test_terminal_node_ends_after_assistant_reply():
    golden = ConversationalGolden(
        scenario="Single terminal",
        expected_outcome=None,
        user_description="Terminal User",
    )

    root = SimulationNode(action=lambda: "hello", terminal=True, name="root")

    simulator = ConversationSimulator(
        model_callback=static_callback,
        simulator_model=StaticSimulatorModel(),
        async_mode=False,
        simulation_graph=root,
    )

    cases = simulator.simulate([golden], max_user_simulations=5)

    tc = cases[0]
    # One user turn + one assistant turn, then the simulation ends.
    assert len(tc.turns) == 2
    assert tc.turns[0] == Turn(role="user", content="hello")
    assert tc.turns[1].role == "assistant"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_two_node_graph_routes_to_child_on_match():
    golden = ConversationalGolden(
        scenario="Two-node routing",
        expected_outcome=None,
        user_description="Routing User",
    )

    root = SimulationNode(action=lambda: "I need a refund", name="root")
    child = SimulationNode(
        action=lambda: "Thanks!", terminal=True, name="child"
    )
    root.add_node(child, when="The assistant approved the refund")

    # Classifier returns 1 -> route to child.
    simulator_model = StaticSimulatorModel(edge_indices=[1])
    simulator = ConversationSimulator(
        model_callback=static_callback,
        simulator_model=simulator_model,
        async_mode=False,
        simulation_graph=root,
    )

    cases = simulator.simulate([golden], max_user_simulations=5)
    tc = cases[0]

    assert [t.content for t in tc.turns if t.role == "user"] == [
        "I need a refund",
        "Thanks!",
    ]
    assert simulator_model.schema_calls.count("EdgeChoice") == 1


def test_none_of_the_above_stays_on_current_node():
    golden = ConversationalGolden(
        scenario="No-match routing",
        expected_outcome=None,
        user_description="Stay User",
    )

    visits: List[int] = []

    def root_action():
        visits.append(len(visits))
        return "still complaining"

    root = SimulationNode(action=root_action, name="root")
    other = SimulationNode(action=lambda: "moved on", name="other")
    root.add_node(other, when="The assistant solved it")

    # Classifier always returns None -> stay on `root`.
    simulator_model = StaticSimulatorModel(edge_indices=[None])
    simulator = ConversationSimulator(
        model_callback=static_callback,
        simulator_model=simulator_model,
        async_mode=False,
        simulation_graph=root,
    )

    cases = simulator.simulate([golden], max_user_simulations=3)
    tc = cases[0]

    # All 3 user turns come from root, never moved to `other`.
    user_contents = [t.content for t in tc.turns if t.role == "user"]
    assert user_contents == ["still complaining"] * 3
    assert simulator_model.schema_calls.count("EdgeChoice") == 3


# ---------------------------------------------------------------------------
# max_visits exhaustion
# ---------------------------------------------------------------------------


def test_self_loop_with_max_visits_caps_emissions():
    golden = ConversationalGolden(
        scenario="Self-loop cap",
        expected_outcome=None,
        user_description="Loop User",
    )

    push = SimulationNode(action=lambda: "push", max_visits=3, name="push")
    push.add_node(push, when="The assistant still refused")

    # Classifier always routes to push (self-loop). max_visits=3 means 3
    # emissions, then the 4th entry attempt skips and ends.
    simulator_model = StaticSimulatorModel(edge_indices=[1])
    simulator = ConversationSimulator(
        model_callback=static_callback,
        simulator_model=simulator_model,
        async_mode=False,
        simulation_graph=push,
    )

    cases = simulator.simulate([golden], max_user_simulations=10)
    tc = cases[0]

    user_turns = [t for t in tc.turns if t.role == "user"]
    assistant_turns = [t for t in tc.turns if t.role == "assistant"]
    assert len(user_turns) == 3
    assert len(assistant_turns) == 3


# ---------------------------------------------------------------------------
# Action return-value normalization
# ---------------------------------------------------------------------------


def test_action_returning_str_is_wrapped_as_user_turn():
    golden = ConversationalGolden(
        scenario="Str wrap",
        expected_outcome=None,
        user_description="User",
    )

    root = SimulationNode(action=lambda: "raw text", terminal=True, name="root")

    simulator = ConversationSimulator(
        model_callback=static_callback,
        simulator_model=StaticSimulatorModel(),
        async_mode=False,
        simulation_graph=root,
    )
    cases = simulator.simulate([golden], max_user_simulations=1)
    assert cases[0].turns[0] == Turn(role="user", content="raw text")


def test_action_returning_non_user_turn_raises_type_error():
    golden = ConversationalGolden(
        scenario="Bad role",
        expected_outcome=None,
        user_description="User",
    )

    root = SimulationNode(
        action=lambda: Turn(role="assistant", content="oops"),
        terminal=True,
        name="root",
    )
    simulator = ConversationSimulator(
        model_callback=static_callback,
        simulator_model=StaticSimulatorModel(),
        async_mode=False,
        simulation_graph=root,
    )

    with pytest.raises(TypeError):
        simulator.simulate([golden], max_user_simulations=1)


# ---------------------------------------------------------------------------
# Kwargs filtering — `simulator` is exposed to actions
# ---------------------------------------------------------------------------


def test_node_action_receives_simulator_when_declared():
    golden = ConversationalGolden(
        scenario="Simulator kwarg",
        expected_outcome=None,
        user_description="User",
    )

    received: List[Optional[object]] = []

    def action(simulator, turns, golden):
        received.append(simulator)
        return f"have-simulator={simulator is not None}; turns={len(turns)}"

    root = SimulationNode(action=action, terminal=True, name="root")

    simulator = ConversationSimulator(
        model_callback=static_callback,
        simulator_model=StaticSimulatorModel(),
        async_mode=False,
        simulation_graph=root,
    )

    cases = simulator.simulate([golden], max_user_simulations=1)

    assert len(received) == 1
    assert received[0] is simulator
    assert cases[0].turns[0].content == "have-simulator=True; turns=0"


# ---------------------------------------------------------------------------
# Composition: drop default_simulation_node into a custom graph
# ---------------------------------------------------------------------------


def test_default_simulation_node_usable_as_child():
    golden = ConversationalGolden(
        scenario="Composition",
        expected_outcome=None,
        user_description="Compose User",
    )

    root = SimulationNode(action=lambda: "scripted opener", name="root")
    root.add_node(
        default_simulation_node(terminal=True),
        when="The assistant responded",
    )

    # Classifier picks edge 1 -> default_simulation_node.
    simulator_model = StaticSimulatorModel(edge_indices=[1])
    simulator = ConversationSimulator(
        model_callback=static_callback,
        simulator_model=simulator_model,
        async_mode=False,
        simulation_graph=root,
    )

    cases = simulator.simulate([golden], max_user_simulations=5)
    tc = cases[0]

    user_contents = [t.content for t in tc.turns if t.role == "user"]
    assert user_contents[0] == "scripted opener"
    # The second user turn comes from the SimulatedInput path used by default_simulation_node.
    assert user_contents[1].startswith("simulated user input")
    # Then default_simulation_node is terminal -> ends after assistant reply.
    assert len(tc.turns) == 4
    assert simulator_model.schema_calls.count("EdgeChoice") == 1


# ---------------------------------------------------------------------------
# Async action support
# ---------------------------------------------------------------------------


def test_async_action_in_async_mode():
    golden = ConversationalGolden(
        scenario="Async action",
        expected_outcome=None,
        user_description="Async User",
    )

    async def aaction():
        return "async user input"

    root = SimulationNode(action=aaction, terminal=True, name="root")

    simulator = ConversationSimulator(
        model_callback=async_static_callback,
        simulator_model=StaticSimulatorModel(),
        async_mode=True,
        simulation_graph=root,
    )
    cases = simulator.simulate([golden], max_user_simulations=1)
    assert cases[0].turns[0].content == "async user input"


# ---------------------------------------------------------------------------
# Deprecation: controller= alias still works
# ---------------------------------------------------------------------------


def test_controller_kwarg_is_deprecated_alias():
    golden = ConversationalGolden(
        scenario="Deprecated alias",
        expected_outcome=None,
        user_description="Legacy User",
    )

    def controller(simulated_user_turns):
        return None  # falls back to proceed()

    with pytest.warns(DeprecationWarning, match="controller"):
        simulator = ConversationSimulator(
            model_callback=static_callback,
            simulator_model=StaticSimulatorModel(),
            async_mode=False,
            controller=controller,
        )

    cases = simulator.simulate([golden], max_user_simulations=1)
    assert len(cases[0].turns) == 2


def test_controller_and_stopping_controller_both_raise():
    def controller(simulated_user_turns):
        return None

    with pytest.raises(TypeError, match="not both"):
        ConversationSimulator(
            model_callback=static_callback,
            simulator_model=StaticSimulatorModel(),
            async_mode=False,
            controller=controller,
            stopping_controller=controller,
        )
