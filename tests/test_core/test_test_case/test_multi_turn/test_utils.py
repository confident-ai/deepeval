from deepeval.test_case import Turn
from deepeval.metrics.utils import get_unit_interactions


def make_turns(seq):
    """Helper to create Turn objects from list of (role, content)."""
    return [Turn(role=role, content=content) for role, content in seq]


class TestGetUnitInteractions:
    def test_consecutive_users(self):
        seq = [("user", "u1"), ("user", "u2"), ("assistant", "a1")]
        expected = [[("user", "u1"), ("user", "u2"), ("assistant", "a1")]]
        result = get_unit_interactions(make_turns(seq))
        assert [
            [(t.role, t.content) for t in unit] for unit in result
        ] == expected

    def test_consecutive_assistants(self):
        seq = [("user", "u1"), ("assistant", "a1"), ("assistant", "a2")]
        expected = [[("user", "u1"), ("assistant", "a1"), ("assistant", "a2")]]
        result = get_unit_interactions(make_turns(seq))
        assert [
            [(t.role, t.content) for t in unit] for unit in result
        ] == expected

    def test_user_assistant_user(self):
        seq = [("user", "u1"), ("assistant", "a1"), ("user", "u2")]
        expected = [[("user", "u1"), ("assistant", "a1")]]  # last user ignored
        result = get_unit_interactions(make_turns(seq))
        assert [
            [(t.role, t.content) for t in unit] for unit in result
        ] == expected

    def test_starts_with_assistant(self):
        seq = [("assistant", "a1"), ("user", "u1"), ("assistant", "a2")]
        expected = [[("assistant", "a1"), ("user", "u1"), ("assistant", "a2")]]
        result = get_unit_interactions(make_turns(seq))
        assert [
            [(t.role, t.content) for t in unit] for unit in result
        ] == expected

    def test_assistant_only_start_then_end_with_user(self):
        seq = [("assistant", "a1"), ("assistant", "a2"), ("user", "u1")]
        expected = []  # ends with user -> ignored
        result = get_unit_interactions(make_turns(seq))
        assert [
            [(t.role, t.content) for t in unit] for unit in result
        ] == expected

    def test_multiple_units(self):
        seq = [
            ("user", "u1"),
            ("assistant", "a1"),
            ("user", "u2"),
            ("assistant", "a2"),
        ]
        expected = [
            [("user", "u1"), ("assistant", "a1")],
            [("user", "u2"), ("assistant", "a2")],
        ]
        result = get_unit_interactions(make_turns(seq))
        assert [
            [(t.role, t.content) for t in unit] for unit in result
        ] == expected

    def test_empty_input(self):
        seq = []
        expected = []
        result = get_unit_interactions(make_turns(seq))
        assert [
            [(t.role, t.content) for t in unit] for unit in result
        ] == expected
