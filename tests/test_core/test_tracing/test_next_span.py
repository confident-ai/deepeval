"""``next_*_span(...)`` staging and ``apply_pending_to_span(...)``, independent
of any integration that consumes them."""

from deepeval.tracing.context import (
    apply_pending_to_span,
    next_agent_span,
    next_llm_span,
    next_retriever_span,
    next_span,
    next_tool_span,
    pop_pending_for,
)
from deepeval.tracing.types import AgentSpan, BaseSpan, TraceSpanStatus


class TestNextSpanPureContextAPI:
    def test_pop_outside_with_returns_empty(self):
        """No pending slot → popping returns an empty dict, never None."""
        assert pop_pending_for("agent") == {}
        assert pop_pending_for(None) == {}

    def test_next_agent_span_one_shot_consumption(self):
        """First pop drains; second pop returns empty for the same scope."""
        with next_agent_span(metric_collection="A", available_tools=["x"]):
            first = pop_pending_for("agent")
            assert first == {
                "metric_collection": "A",
                "available_tools": ["x"],
            }

            second = pop_pending_for("agent")
            assert second == {}

    def test_scope_exit_restores_prior_value(self):
        """Token-based reset: leaving the ``with`` block puts the slot
        back to ``None`` (not just empty-dict)."""
        with next_agent_span(metric_collection="A"):
            pass

        # After exit, popping yields nothing — slot is back to None.
        assert pop_pending_for("agent") == {}

    def test_stacked_typed_slots_are_independent(self):
        """``with next_agent_span(...), next_llm_span(...):`` keeps each
        slot separate; popping one does not drain the other."""
        with next_agent_span(metric_collection="A"), next_llm_span(
            model="gpt-4"
        ):
            agent_payload = pop_pending_for("agent")
            assert agent_payload == {"metric_collection": "A"}

            llm_payload = pop_pending_for("llm")
            assert llm_payload == {"model": "gpt-4"}

    def test_base_slot_consumed_by_first_typed_pop(self):
        """``next_span`` is "next of any type"; the first ``pop_pending_for``
        call inside the scope drains it regardless of typed slot match."""
        with next_span(metadata={"k": "v"}):
            first = pop_pending_for("agent")
            assert first == {"metadata": {"k": "v"}}

            # Subsequent pops see no base slot.
            assert pop_pending_for("llm") == {}

    def test_typed_overrides_base_on_key_overlap(self):
        """When base + typed both set the same key, the typed slot wins
        (more specific wins)."""
        with next_span(metric_collection="base"), next_agent_span(
            metric_collection="typed"
        ):
            payload = pop_pending_for("agent")
            assert payload["metric_collection"] == "typed"

    def test_pop_for_mismatched_type_drains_only_base(self):
        """``next_agent_span(...)`` is NOT consumed by
        ``pop_pending_for('llm')``. Base slot still goes (it's
        any-type)."""
        with next_span(metadata={"k": "v"}), next_agent_span(
            metric_collection="A"
        ):
            llm_payload = pop_pending_for("llm")
            # Base flowed through, agent slot untouched.
            assert llm_payload == {"metadata": {"k": "v"}}

            agent_payload = pop_pending_for("agent")
            assert agent_payload == {"metric_collection": "A"}

    def test_nested_same_type_innermost_wins(self):
        """Nested ``with next_agent_span(...)`` blocks: inner overrides
        for its scope; on exit, outer is restored."""
        with next_agent_span(metric_collection="outer"):
            with next_agent_span(metric_collection="inner"):
                assert pop_pending_for("agent") == {
                    "metric_collection": "inner"
                }

            # Outer scope's value is back, ready to be consumed once.
            assert pop_pending_for("agent") == {"metric_collection": "outer"}

    def test_drops_none_kwargs(self):
        """Slots store only kwargs the user actually passed; ``None``
        kwargs are stripped so consumers don't have to re-check."""
        with next_agent_span(metric_collection="A"):
            payload = pop_pending_for("agent")
            assert "available_tools" not in payload
            assert "name" not in payload
            assert "metadata" not in payload

    def test_unconsumed_payload_does_not_leak_across_scopes(self):
        """If no consumer pops inside the ``with``, the payload is
        discarded on exit — it never leaks to a sibling scope."""
        with next_agent_span(metric_collection="leaked"):
            pass  # nobody popped

        # Sibling scope: starts clean.
        with next_agent_span(metric_collection="fresh"):
            assert pop_pending_for("agent") == {"metric_collection": "fresh"}

    def test_drain_visible_across_asyncio_sub_context(self):
        """Regression: ``Agent.run_sync(...)`` calls ``asyncio.run(...)``
        which creates a new asyncio context that inherits a SNAPSHOT of
        contextvars. A naive ``ContextVar.set(None)`` from inside that
        snapshot would not propagate back, letting a second consumer in
        the parent context re-consume the same value.

        This test simulates the failure mode by running the consumer
        inside ``asyncio.run`` and verifying that the second consumer
        in the OUTER context sees the slot already drained.
        """
        import asyncio

        with next_agent_span(metric_collection="only-once"):

            async def _consume():
                return pop_pending_for("agent")

            # First consumer runs inside asyncio.run — same trick
            # ``Agent.run_sync`` plays internally.
            first = asyncio.run(_consume())
            assert first == {"metric_collection": "only-once"}

            # Second consumer in the outer ``with`` context. Must see
            # an empty dict because the asyncio sub-context's drain
            # mutated the shared ``_PendingSlot``.
            second = pop_pending_for("agent")
            assert second == {}

    def test_other_typed_helpers_each_use_their_own_slot(self):
        """Smoke test that ``next_tool_span`` / ``next_retriever_span``
        wire up to their respective slots (not the base/agent/llm
        slots)."""
        with next_tool_span(description="foo"):
            assert pop_pending_for("tool") == {"description": "foo"}
            assert pop_pending_for("agent") == {}

        with next_retriever_span(top_k=3, embedder="ada-002"):
            assert pop_pending_for("retriever") == {
                "top_k": 3,
                "embedder": "ada-002",
            }


# ---------------------------------------------------------------------------
# apply_pending_to_span — placeholder mutation behavior.
# ---------------------------------------------------------------------------


def _make_placeholder(cls=BaseSpan, **kw) -> BaseSpan:
    """Helper to build a minimal placeholder for applier tests."""
    base_kwargs = {
        "uuid": "u-1",
        "trace_uuid": "t-1",
        "status": TraceSpanStatus.IN_PROGRESS,
        "start_time": 0.0,
    }
    if cls is AgentSpan:
        base_kwargs.setdefault("name", "agent")
    base_kwargs.update(kw)
    return cls(**base_kwargs)


class TestApplyPendingToSpan:
    def test_empty_payload_is_noop(self):
        span = _make_placeholder()
        apply_pending_to_span(span, {})
        # Nothing changed — sanity.
        assert span.metric_collection is None
        assert span.metadata is None

    def test_base_field_setattr(self):
        span = _make_placeholder()
        apply_pending_to_span(
            span,
            {"metric_collection": "mc", "metadata": {"k": "v"}, "name": "n"},
        )
        assert span.metric_collection == "mc"
        assert span.metadata == {"k": "v"}
        assert span.name == "n"

    def test_agent_specific_fields_apply_only_to_agent_span(self):
        agent = _make_placeholder(AgentSpan)
        apply_pending_to_span(
            agent,
            {"available_tools": ["a", "b"], "agent_handoffs": ["h1"]},
        )
        assert agent.available_tools == ["a", "b"]
        assert agent.agent_handoffs == ["h1"]

    def test_cross_type_keys_silently_dropped(self):
        """Applier is hasattr-guarded: typed kwargs that don't apply to
        the placeholder's class are silently skipped instead of raising."""
        base = _make_placeholder()  # plain BaseSpan, no model/embedder/etc.
        apply_pending_to_span(
            base,
            {
                "model": "gpt-4",  # llm-only
                "embedder": "ada",  # retriever-only
                "available_tools": ["x"],  # agent-only
                "metric_collection": "shared",  # base — should land
            },
        )
        # Only the shared base field landed.
        assert base.metric_collection == "shared"
        # Cross-type keys did not raise and did not phantom-attribute.
        assert not hasattr(base, "model")
        assert not hasattr(base, "embedder")

    def test_test_case_unpacking_then_individual_fields_override(self):
        """``test_case`` is unpacked first, then individual base fields
        applied — so individual kwargs override the test_case's
        equivalent fields. Mirrors ``update_current_span(...)``'s order
        of operations (test_case first, then ``input``/``output``/etc.).
        Asserting this so the contract doesn't quietly flip."""
        from deepeval.test_case.llm_test_case import LLMTestCase

        span = _make_placeholder()
        tc = LLMTestCase(
            input="tc-input",
            actual_output="tc-output",
            expected_output="tc-expected",
        )
        apply_pending_to_span(
            span,
            {
                "test_case": tc,
                "input": "individual-input",  # overrides tc.input
                "output": "individual-output",  # overrides tc.actual_output
                # expected_output not overridden — falls through to tc.
            },
        )
        assert span.input == "individual-input"
        assert span.output == "individual-output"
        assert span.expected_output == "tc-expected"
