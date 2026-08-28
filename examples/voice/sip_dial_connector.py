"""Dial a phone number into a voice simulation through your own LiveKit trunk.

deepeval ships no telephony: reaching a phone number needs a carrier, and that
account is yours. `LiveKitConnector` joins a room, runs `_after_join()`, then
waits for the agent's audio track. A subclass dials the callee inside that hook
and names the callee so its track, not some other participant's, is adopted.

Requires `pip install "deepeval[voice]"`, LIVEKIT_URL / LIVEKIT_API_KEY /
LIVEKIT_API_SECRET, and an outbound SIP trunk configured in LiveKit
(https://docs.livekit.io/sip/trunk-outbound/).

    python sip_dial_connector.py +14155550100 ST_xxxxxxxxxxxx
"""

import sys
from typing import ClassVar

from deepeval.dataset import ConversationalGolden, Persona
from deepeval.errors import DeepEvalError
from deepeval.simulator import ConversationSimulator
from deepeval.voice import LiveKitConnector, VoiceConfig, VoiceProtocol

CALLEE_IDENTITY = "callee"


class SipDialConnector(LiveKitConnector):

    protocol: ClassVar[VoiceProtocol] = VoiceProtocol.SIP

    def __init__(self, phone_number: str, sip_trunk_id: str, **kwargs):
        kwargs.setdefault("agent_identity", CALLEE_IDENTITY)
        kwargs.setdefault("connect_timeout_s", 60.0)
        kwargs.setdefault("turn_detection", "patient")
        super().__init__(**kwargs)
        self.phone_number = phone_number
        self.sip_trunk_id = sip_trunk_id

    async def _after_join(self) -> None:
        api = self._api
        client = api.LiveKitAPI(self.url, self.api_key, self.api_secret)
        try:
            await client.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    sip_trunk_id=self.sip_trunk_id,
                    sip_call_to=self.phone_number,
                    room_name=self.room_name,
                    participant_identity=CALLEE_IDENTITY,
                    participant_name=self.phone_number,
                    wait_until_answered=True,
                    krisp_enabled=True,
                )
            )
        except Exception as error:
            await self.disconnect()
            raise DeepEvalError(
                f"Dialing {self.phone_number} failed: {error}"
            ) from error
        finally:
            await client.aclose()


def main(phone_number: str, sip_trunk_id: str) -> None:
    simulator = ConversationSimulator(
        voice_config=VoiceConfig(
            connector=lambda: SipDialConnector(phone_number, sip_trunk_id),
        ),
        max_concurrent=2,
    )
    goldens = [
        ConversationalGolden(
            scenario="Move Thursday's appointment to next Tuesday morning.",
            expected_outcome="The appointment is rescheduled and confirmed.",
            persona=Persona(characteristics="A polite but hurried caller."),
        ),
        ConversationalGolden(
            scenario="Ask what documents to bring to a first visit.",
            expected_outcome="The caller is told what to bring.",
            persona=Persona(characteristics="A first-time patient."),
        ),
    ]
    for case in simulator.simulate(goldens, max_user_simulations=4):
        print(f"\n{case.scenario}")
        for turn in case.turns:
            latency = f" ({turn.latency_ms:.0f} ms)" if turn.latency_ms else ""
            print(f"  {turn.role:>9}{latency}: {turn.content}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
