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
from deepeval.voice.connectors.types import AgentEvent, ConnectorTurn
from deepeval.voice.connectors.transports.websocket import (
    BaseWebSocketConnector,
    InboundEvent,
)

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict], Union[Any, Awaitable[Any]]]

PHONIC_URL = "wss://api.phonic.ai/v1/sts/ws"
# Phonic says a few words, stops, then calls a tool. The turn is only over if
# no tool call follows within this long of it going quiet.
TOOL_GRACE_S = 1.5


class PhonicConnector(BaseWebSocketConnector):
    """Calls a Phonic speech-to-speech model directly, as the agent.

    `tools` are function definitions (`name`, `description`, `parameters`)
    and `tool_handlers` run them here; every call is recorded on the agent's
    turn as `tools_called`.

    Args:
        model: The Phonic model, such as `"phonic-1"`.
        instructions: The agent's system prompt.
        tools: Function tools the agent may call.
        tool_handlers: One handler per tool name, given the call's arguments.
        voice: The voice the agent speaks in.
        intelligence_level: `"standard"` or `"high"`.
        additional_languages: Languages besides English the agent should
            recognise and speak, as ISO 639-1 codes. The agent changes to
            one when the caller asks it to.
        greets_first: Have the agent speak as soon as the call connects.
    """

    def __init__(
        self,
        model: str = "phonic-1",
        instructions: Optional[str] = None,
        tools: Optional[List[dict]] = None,
        tool_handlers: Optional[Dict[str, ToolHandler]] = None,
        voice: str = "sabrina",
        api_key: Optional[str] = None,
        intelligence_level: str = "high",
        additional_languages: Optional[List[str]] = None,
        greets_first: bool = True,
        tool_timeout_s: float = 30.0,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.model = model
        self.instructions = instructions
        self.tools = tools or []
        self.tool_handlers = tool_handlers or {}
        self.voice = voice
        self.api_key = api_key or os.getenv("PHONIC_API_KEY")
        self.intelligence_level = intelligence_level
        self.additional_languages = additional_languages or []
        self.greets_first = greets_first
        self.tool_timeout_s = tool_timeout_s
        # Seconds of agent audio, which is what Phonic bills on. Kept in one
        # dict so a clone made for the call reports into the same place.
        self.usage: Dict[str, float] = {}
        self._spoken = ""
        self._closing: Optional[asyncio.Task] = None

    @property
    def signals_turn_complete(self) -> bool:
        # `assistant_finished_speaking` closes every reply.
        return True

    @property
    def supports_duplex(self) -> bool:
        return True

    async def _open_session(self) -> str:
        if not self.api_key:
            raise DeepEvalError(
                "PhonicConnector needs an API key: pass `api_key` or set "
                "PHONIC_API_KEY."
            )
        return PHONIC_URL

    def _connect_headers(self) -> Optional[dict]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _initial_messages(self) -> List[Union[str, bytes]]:
        rate = f"pcm_{self.sample_rate}"
        config: Dict[str, Any] = {
            "type": "config",
            "model": self.model,
            "intelligence_level": self.intelligence_level,
            "voice_id": self.voice,
            "input_format": rate,
            "output_format": rate,
            "generate_welcome_message": self.greets_first,
            "tools": [self._tool(tool) for tool in self.tools],
        }
        if self.additional_languages:
            # "request": the agent changes language when the caller asks it
            # to. Phonic's automatic mode allows only two extra languages.
            config["additional_languages"] = self.additional_languages
            config["multilingual_mode"] = "request"
        if self.instructions:
            config["system_prompt"] = self.instructions
        return [json.dumps(config)]

    def _tool(self, tool: dict) -> dict:
        parameters = dict(tool.get("parameters") or {"type": "object"})
        properties = parameters.get("properties") or {}
        # Inline tools are strict: every property required, nothing extra.
        parameters.update(
            properties=properties,
            required=list(properties),
            additionalProperties=False,
        )
        return {
            "type": "custom_websocket",
            "tool_schema": {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") or tool["name"],
                    "parameters": parameters,
                    "strict": True,
                },
            },
            "tool_call_output_timeout_ms": int(self.tool_timeout_s * 1000),
        }

    async def connect(self) -> None:
        self._spoken = ""
        self._closing = None
        await super().connect()

    async def _send(self, message: Union[str, bytes]) -> None:
        try:
            await super()._send(message)
        except ConnectionResetError:
            # The call is closing; a frame still on its way out is dropped.
            logger.debug("Phonic socket closed while sending.")

    def _encode_outbound(self, pcm: bytes) -> str:
        return json.dumps(
            {
                "type": "audio_chunk",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )

    def _decode_inbound(self, raw: Union[str, bytes]) -> Optional[InboundEvent]:
        try:
            message = json.loads(raw)
        except (ValueError, TypeError):
            return None
        kind = message.get("type")

        if kind == "ready_to_start_conversation":
            return InboundEvent(ready=True)

        if kind == "audio_chunk":
            audio = base64.b64decode(message.get("audio") or "")
            self.usage["agent_audio_seconds"] = self.usage.get(
                "agent_audio_seconds", 0.0
            ) + len(audio) / 2 / self.sample_rate
            # The reader wants the whole reply so far, not the next piece.
            self._spoken += message.get("text") or ""
            # Quiet frames keep arriving between replies; only new words mean
            # the agent has more to say.
            if message.get("text"):
                self._keep_turn_open()
            return InboundEvent(
                audio=audio or None, transcript=self._spoken or None
            )

        if kind == "input_text":
            heard = (message.get("text") or "").strip()
            return InboundEvent(provider_transcript=heard) if heard else None

        if kind == "interrupted_response":
            self._interrupted = True
            self._keep_turn_open()
            self._spoken = ""
            return InboundEvent(turn_complete=True)

        if kind == "assistant_finished_speaking":
            self._keep_turn_open()
            self._closing = self._loop.create_task(self._close_turn())
            return None

        if kind == "tool_call":
            self._keep_turn_open()
            if self._spoken:
                self._spoken += " "
            self._loop.create_task(self._run_tool(message))
            return None

        if kind in ("error", "warning"):
            logger.warning("Phonic %s: %s", kind, message.get(kind))
        return None

    def _keep_turn_open(self) -> None:
        if self._closing is not None:
            self._closing.cancel()
            self._closing = None

    async def _close_turn(self) -> None:
        """End the agent's turn unless a tool call or more speech follows."""
        await asyncio.sleep(TOOL_GRACE_S)
        while self._tools_running:
            await asyncio.sleep(0.1)
        self._closing = None
        self._spoken = ""
        await self._inbound.put(AgentEvent(turn_complete=True))

    async def _run_tool(self, call: dict) -> None:
        began = time.perf_counter()
        self._hold_turn(True)
        try:
            name, arguments = call.get("tool_name"), call.get("parameters") or {}
            result = await self._invoke(name, arguments)
            await self._inbound.put(
                AgentEvent(
                    tools_called=[
                        ToolCall(
                            name=name, input_parameters=arguments, output=result
                        )
                    ],
                    received_at=time.perf_counter(),
                )
            )
            await self._send(
                json.dumps(
                    {
                        "type": "tool_call_output",
                        "tool_call_id": call.get("tool_call_id"),
                        "output": result,
                    },
                    default=str,
                )
            )
        except Exception:
            logger.exception("Failed to answer a Phonic tool call.")
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
