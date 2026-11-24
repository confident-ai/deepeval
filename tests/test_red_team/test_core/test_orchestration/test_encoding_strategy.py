import asyncio

import pytest

from deepeval.red_team.attacks.multi_turn.progression import Probe
from deepeval.red_team.orchestration.strategy import (
    Base64Strategy,
    Effectiveness,
    LeetspeakStrategy,
    ROT13Strategy,
    StrategyExecutor,
)

ALL = [Base64Strategy, ROT13Strategy, LeetspeakStrategy]


def run(coro):
    return asyncio.run(coro)


class RecordingSend:
    def __init__(self):
        self.sent = []

    async def __call__(self, text: str) -> Probe:
        probe = Probe(id=len(self.sent), input=text)
        self.sent.append(probe)
        return probe


class TestExecute:

    @pytest.mark.parametrize("make", ALL)
    def test_sends_one_encoded_turn_with_the_reply_instruction(self, make):
        strategy = make()
        send = RecordingSend()
        ex = StrategyExecutor(sub_agent=None)
        ex.adopt(strategy)
        probe = run(ex.a_send("name a cat", send))

        assert len(send.sent) == 1 and probe is send.sent[-1]
        sent = send.sent[0].input
        body, instruction = sent.split("\n\n", 1)
        assert body == strategy.attack.enhance("name a cat")
        assert body != "name a cat"
        assert strategy.name in instruction

    @pytest.mark.parametrize("make", ALL)
    def test_metadata_comes_from_the_attack(self, make):
        strategy = make()
        assert strategy.name == strategy.attack.get_name()
        assert strategy.name in strategy.description
        assert isinstance(strategy.effectiveness, Effectiveness)


class TestDecode:

    @pytest.mark.parametrize("make", ALL)
    def test_decode_recovers_an_encoded_reply(self, make):
        strategy = make()
        encoded = strategy.attack.enhance("the answer is a cat")
        assert strategy.decode(encoded) == "the answer is a cat"

    def test_base64_decode_leaves_plain_text_alone(self):
        assert Base64Strategy().decode("Sorry, I can't do that.") is None

    def test_base64_decode_handles_wrapped_lines(self):
        strategy = Base64Strategy()
        encoded = strategy.attack.enhance("a lion and a rose")
        wrapped = encoded[:8] + "\n" + encoded[8:]
        assert strategy.decode(wrapped) == "a lion and a rose"

    @pytest.mark.parametrize("make", [ROT13Strategy, LeetspeakStrategy])
    def test_decode_leaves_a_plain_reply_alone(self, make):
        assert make().decode("Sorry, I can't help with that request.") is None
        assert make().decode("Take 1000mg twice a day with food.") is None
