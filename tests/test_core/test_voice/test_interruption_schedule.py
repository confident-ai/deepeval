import pytest

from deepeval.dataset import InterruptionBehavior, Persona
from deepeval.voice.interruption import InterruptionSchedule


def _picks(schedule: InterruptionSchedule, replies: int):
    picked = []
    for _ in range(replies):
        due = schedule.on_agent_reply_start()
        picked.append(due)
        if due:
            schedule.on_interruption_landed()
    return picked


def test_a_fixed_cadence_interrupts_every_nth_reply():
    assert _picks(InterruptionSchedule(3), 9) == [
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]


def test_every_one_interrupts_each_reply():
    assert _picks(InterruptionSchedule(1), 4) == [True, True, True, True]


def test_a_range_keeps_every_gap_inside_it():
    picked = _picks(InterruptionSchedule((2, 4), seed=7), 200)
    hits = [index for index, due in enumerate(picked) if due]
    gaps = [hits[0] + 1] + [b - a for a, b in zip(hits, hits[1:])]

    assert set(gaps) <= {2, 3, 4}
    assert len(set(gaps)) > 1


def test_the_same_seed_gives_the_same_schedule():
    assert _picks(InterruptionSchedule((1, 5), seed=3), 50) == _picks(
        InterruptionSchedule((1, 5), seed=3), 50
    )


def test_a_reply_that_was_not_interrupted_carries_it_to_the_next():
    schedule = InterruptionSchedule(2)

    assert schedule.on_agent_reply_start() is False
    assert schedule.on_agent_reply_start() is True
    assert schedule.on_agent_reply_start() is True
    schedule.on_interruption_landed()
    assert schedule.on_agent_reply_start() is False


def test_the_peek_matches_the_next_reply_without_counting_it():
    schedule = InterruptionSchedule(2)

    assert schedule.interrupts_next_reply is False
    schedule.on_agent_reply_start()
    assert schedule.interrupts_next_reply is True
    assert schedule.interrupts_next_reply is True
    assert schedule.on_agent_reply_start() is True


@pytest.mark.parametrize("every", [0, -1, (3, 2), (0, 2), 1.5, True, (1, 2, 3)])
def test_an_invalid_cadence_is_rejected(every):
    with pytest.raises(ValueError, match="Invalid interruption cadence"):
        InterruptionBehavior(every=every)


def test_a_persona_accepts_the_cadence_as_data():
    for raw, expected in (({"every": [2, 3]}, (2, 3)), ({"every": 3}, (3, 3))):
        persona = Persona(
            characteristics="Impatient", interruption_behavior=raw
        )
        assert persona.interruption_behavior.every == expected
