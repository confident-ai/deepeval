import os
import json
import time
import base64
import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from deepeval.errors import DeepEvalError
from deepeval.test_case import Audio, ToolCall
from deepeval.voice.connectors import audio_utils
from deepeval.voice.connectors.types import AgentEvent, ConnectorTurn
from deepeval.voice.connectors.transports.websocket import (
    BaseWebSocketConnector,
    InboundEvent,
)

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict], Union[Any, Awaitable[Any]]]

REALTIME_URL = "wss://api.openai.com/v1/realtime"
_AUDIO_DELTAS = ("response.output_audio.delta", "response.audio.delta")
_TRANSCRIPT_DELTAS = (
    "response.output_audio_transcript.delta",
    "response.audio_transcript.delta",
)


class OpenAIRealtimeConnector(BaseWebSocketConnector):
    """Calls an OpenAI speech-to-speech model directly, as the agent.

    The model hears the caller's audio and answers in audio, with nothing in
    between. `tools` are function definitions in the Realtime API's shape and
    `tool_handlers` run them here; every call is recorded on the agent's turn
    as `tools_called`.

    Args:
        model: The Realtime model, such as `"gpt-realtime"`.
        instructions: The agent's system prompt.
        tools: Function tools the agent may call.
        tool_handlers: One handler per tool name, given the call's arguments.
        voice: The voice the agent speaks in.
        greets_first: Have the agent speak as soon as the call connects.
        transcription_model: Transcribes the caller for `provider_transcript`.
    """

    def __init__(
        self,
        model: str = "gpt-realtime",
        instructions: Optional[str] = None,
        tools: Optional[List[dict]] = None,
        tool_handlers: Optional[Dict[str, ToolHandler]] = None,
        voice: str = "marin",
        api_key: Optional[str] = None,
        greets_first: bool = True,
        transcription_model: str = "gpt-4o-mini-transcribe",
        tool_timeout_s: float = 30.0,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.model = model
        self.instructions = instructions
        self.tools = tools or []
        self.tool_handlers = tool_handlers or {}
        self.voice = voice
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.greets_first = greets_first
        self.transcription_model = transcription_model
        self.tool_timeout_s = tool_timeout_s
        self._spoken = ""
        self._continuing = False
        self._turn_tools: List[ToolCall] = []
        self._playout: Optional[asyncio.Queue] = None
        self._play_task: Optional[asyncio.Task] = None
        self._playing = False
        self._cut_off = False

    @property
    def signals_turn_complete(self) -> bool:
        # `response.done` closes every reply.
        return True

    @property
    def supports_duplex(self) -> bool:
        return True

    async def _open_session(self) -> str:
        if not self.api_key:
            raise DeepEvalError(
                "OpenAIRealtimeConnector needs an API key: pass `api_key` or "
                "set OPENAI_API_KEY."
            )
        return f"{REALTIME_URL}?model={self.model}"

    def _connect_headers(self) -> Optional[dict]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _initial_messages(self) -> List[Union[str, bytes]]:
        pcm = {"type": "audio/pcm", "rate": self.sample_rate}
        session: Dict[str, Any] = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": pcm,
                    "transcription": {"model": self.transcription_model},
                    "turn_detection": {"type": "server_vad"},
                },
                "output": {"format": pcm, "voice": self.voice},
            },
            "tools": self.tools,
            "tool_choice": "auto",
        }
        if self.instructions:
            session["instructions"] = self.instructions
        messages = [json.dumps({"type": "session.update", "session": session})]
        if self.greets_first:
            messages.append(json.dumps({"type": "response.create"}))
        return messages

    async def connect(self) -> None:
        self._spoken = ""
        self._turn_tools = []
        self._playout = asyncio.Queue()
        self._playing = False
        self._cut_off = False
        await super().connect()
        self._play_task = self._loop.create_task(self._play())

    async def disconnect(self) -> None:
        if self._play_task is not None:
            self._play_task.cancel()
            self._play_task = None
        await super().disconnect()

    async def _play(self) -> None:
        """Release the agent's reply at the speed it would be heard.

        The model sends a reply far faster than it takes to say, so passing
        it straight on would end the agent's turn seconds before a caller on
        a real line had heard it.
        """
        playhead = 0.0
        while True:
            event = await self._playout.get()
            if event.audio is None:
                # Text and the end of the turn follow the audio before them.
                await asyncio.sleep(max(0.0, playhead - time.perf_counter()))
                if event.turn_complete:
                    self._playing = False
                await self._inbound.put(event)
                continue
            self._playing = True
            for frame in audio_utils.iter_pcm16_frames(
                event.audio, self._recv_rate
            ):
                if self._cut_off:
                    break
                await asyncio.sleep(max(0.0, playhead - time.perf_counter()))
                now = time.perf_counter()
                await self._inbound.put(AgentEvent(audio=frame, received_at=now))
                playhead = max(playhead, now) + len(frame) / 2 / self._recv_rate

    def _stop_playing(self) -> None:
        """The caller spoke over the agent: drop what has not been heard."""
        if not self._playing and self._playout.empty():
            return
        self._cut_off = True
        while not self._playout.empty():
            self._playout.get_nowait()
        self._interrupted = True
        self._playout.put_nowait(AgentEvent(turn_complete=True))

    def _encode_outbound(self, pcm: bytes) -> str:
        return json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )

    def _decode_inbound(self, raw: Union[str, bytes]) -> Optional[InboundEvent]:
        try:
            message = json.loads(raw)
        except (ValueError, TypeError):
            return None
        kind = message.get("type")

        if kind == "session.updated":
            return InboundEvent(ready=True)

        if kind == "response.created":
            # A reply that follows its own tool calls is still the same turn.
            if not self._continuing:
                self._spoken = ""
            self._continuing = False
            self._cut_off = False
            return None

        if kind == "input_audio_buffer.speech_started":
            self._stop_playing()
            return None

        if kind in _AUDIO_DELTAS:
            if not self._cut_off:
                self._playout.put_nowait(
                    AgentEvent(audio=base64.b64decode(message["delta"]))
                )
            return None

        if kind in _TRANSCRIPT_DELTAS:
            # The reader wants the whole reply so far, not the next piece.
            self._spoken += message.get("delta", "")
            if not self._cut_off:
                self._playout.put_nowait(
                    AgentEvent(
                        transcript=self._spoken,
                        received_at=time.perf_counter(),
                    )
                )
            return None

        if kind == "conversation.item.input_audio_transcription.completed":
            heard = (message.get("transcript") or "").strip()
            return InboundEvent(provider_transcript=heard) if heard else None

        if kind == "response.done":
            response = message.get("response") or {}
            calls = [
                item
                for item in response.get("output") or []
                if item.get("type") == "function_call"
            ]
            if calls:
                # The reply carries on once the tools have answered.
                self._loop.create_task(self._run_tools(calls))
                return None
            # A reply the caller cut off has already been closed.
            if not self._cut_off:
                self._playout.put_nowait(AgentEvent(turn_complete=True))
            return None

        if kind == "error":
            logger.warning(
                "OpenAI Realtime error: %s",
                (message.get("error") or {}).get("message"),
            )
        return None

    def take_agent_speech_during_uplink(self) -> List[List[AgentEvent]]:
        waiting: List[AgentEvent] = []
        while not self._inbound.empty():
            waiting.append(self._inbound.get_nowait())
        finished = max(
            (
                index
                for index, event in enumerate(waiting)
                if event.turn_complete
            ),
            default=-1,
        )
        for event in waiting[finished + 1 :]:
            self._inbound.put_nowait(event)
        replies: List[List[AgentEvent]] = [[]]
        for event in waiting[: finished + 1]:
            if event.turn_complete:
                replies.append([])
            else:
                replies[-1].append(event)
        return [reply for reply in replies if reply]

    async def _run_tools(self, calls: List[dict]) -> None:
        """Answer the tool calls of one response, then let the agent go on."""
        began = time.perf_counter()
        self._hold_turn(True)
        try:
            for call in calls:
                name = call.get("name")
                try:
                    arguments = json.loads(call.get("arguments") or "{}")
                except ValueError:
                    arguments = {}
                result = await self._invoke(name, arguments)
                made = ToolCall(
                    name=name, input_parameters=arguments, output=result
                )
                self._turn_tools.append(made)
                await self._inbound.put(
                    AgentEvent(
                        tools_called=[made], received_at=time.perf_counter()
                    )
                )
                await self._send(
                    json.dumps(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": call.get("call_id"),
                                "output": json.dumps(result, default=str),
                            },
                        }
                    )
                )
            if self._spoken:
                self._spoken += " "
            self._continuing = True
            await self._send(json.dumps({"type": "response.create"}))
        except Exception:
            logger.exception("Failed to answer an OpenAI Realtime tool call.")
            await self._inbound.put(AgentEvent(turn_complete=True))
        finally:
            self._tool_spans.append((began, time.perf_counter()))
            self._hold_turn(False)

    async def _invoke(self, name: Optional[str], arguments: dict) -> Any:
        handler = self.tool_handlers.get(name)
        if handler is None:
            return {"error": f"No tool named {name!r}."}
        try:
            if inspect.iscoroutinefunction(handler):
                return await asyncio.wait_for(
                    handler(arguments), timeout=self.tool_timeout_s
                )
            return await asyncio.wait_for(
                self._loop.run_in_executor(None, handler, arguments),
                timeout=self.tool_timeout_s,
            )
        except asyncio.TimeoutError:
            return {"error": f"{name} did not answer in time."}
        except Exception as error:
            return {"error": f"{type(error).__name__}: {error}"}

    async def exchange_turn(self, audio: Audio) -> ConnectorTurn:
        self._spoken = ""
        self._turn_tools = []
        turn = await super().exchange_turn(audio)
        turn.tools_called = list(self._turn_tools) or None
        return turn
