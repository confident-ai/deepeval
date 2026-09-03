import os
import json
import time
import base64
import asyncio
import inspect
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from deepeval.errors import DeepEvalError
from deepeval.voice.connectors.transports.websocket import (
    BaseWebSocketConnector,
    InboundEvent,
)

logger = logging.getLogger(__name__)

ClientToolHandler = Callable[[dict], Union[Any, Awaitable[Any]]]

# Isolated data-residency environments are separate workspaces on their own
# hosts, and they do not follow one pattern: `us` is a plain subdomain while the
# rest sit under `residency`. Spelling them out is the only way to get them
# right.
# https://elevenlabs.io/docs/overview/administration/data-residency
DEFAULT_HOST = "api.elevenlabs.io"
REGION_HOSTS = {
    "us": "api.us.elevenlabs.io",
    "eu": "api.eu.residency.elevenlabs.io",
    "in": "api.in.residency.elevenlabs.io",
    "sg": "api.sg.residency.elevenlabs.io",
}


class ElevenLabsConnector(BaseWebSocketConnector):
    def __init__(
        self,
        agent_id: str,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        client_tools: Optional[Dict[str, ClientToolHandler]] = None,
        conversation_config_override: Optional[dict] = None,
        dynamic_variables: Optional[dict] = None,
        client_tool_timeout_s: float = 30.0,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.agent_id = agent_id
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.client_tools = client_tools or {}
        self.conversation_config_override = conversation_config_override
        self.dynamic_variables = dynamic_variables
        self.client_tool_timeout_s = client_tool_timeout_s
        self.region = region.lower() if region else None
        if self.region is not None and self.region not in REGION_HOSTS:
            raise DeepEvalError(
                f"Unknown ElevenLabs region {region!r}. Use one of "
                f"{', '.join(sorted(REGION_HOSTS))}, or leave it unset for the "
                "global environment."
            )

    def _host(self) -> str:
        if self.region is None:
            return DEFAULT_HOST
        return REGION_HOSTS[self.region]

    @property
    def signals_turn_complete(self) -> bool:
        # `agent_response_complete` closes every turn.
        return True

    async def _open_session(self) -> str:
        host = self._host()
        if self.api_key:
            # Private agents require a signed URL minted via the REST API.
            rest_url = (
                f"https://{host}/v1/convai/conversation/get-signed-url"
                f"?agent_id={self.agent_id}"
            )
            async with self._session.get(
                rest_url, headers={"xi-api-key": self.api_key}
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise DeepEvalError(
                        f"ElevenLabs signed-url request failed "
                        f"({resp.status}): {body}"
                    )
                data = await resp.json()
            signed_url = data.get("signed_url")
            if not signed_url:
                raise DeepEvalError(
                    "ElevenLabs signed-url response missing 'signed_url'."
                )
            return signed_url

        return f"wss://{host}/v1/convai/conversation?agent_id={self.agent_id}"

    def _initial_messages(self) -> List[Union[str, bytes]]:
        # Overrides have to be enabled per-field on the agent in ElevenLabs
        # before it will honour them; the values are supplied per conversation,
        # which is what lets one agent be simulated as many different callers.
        payload = {"type": "conversation_initiation_client_data"}
        if self.conversation_config_override:
            payload["conversation_config_override"] = (
                self.conversation_config_override
            )
        if self.dynamic_variables:
            payload["dynamic_variables"] = self.dynamic_variables
        return [json.dumps(payload)]

    def _encode_outbound(self, pcm: bytes) -> str:
        return json.dumps(
            {"user_audio_chunk": base64.b64encode(pcm).decode("ascii")}
        )

    def _decode_inbound(self, raw: Union[str, bytes]) -> Optional[InboundEvent]:
        try:
            message = json.loads(raw)
        except (ValueError, TypeError):
            return None
        msg_type = message.get("type")

        if msg_type == "conversation_initiation_metadata":
            meta = message.get("conversation_initiation_metadata_event", {})
            self._send_rate = self._parse_format_rate(
                meta.get("user_input_audio_format")
            )
            self._recv_rate = self._parse_format_rate(
                meta.get("agent_output_audio_format")
            )
            return InboundEvent(ready=True)

        if msg_type == "audio":
            audio_b64 = message.get("audio_event", {}).get("audio_base_64")
            if not audio_b64:
                return None
            return InboundEvent(audio=base64.b64decode(audio_b64))

        if msg_type == "agent_response":
            text = message.get("agent_response_event", {}).get("agent_response")
            return InboundEvent(transcript=text)

        if msg_type == "agent_response_complete":
            return InboundEvent(turn_complete=True)

        if msg_type == "ping":
            event_id = message.get("ping_event", {}).get("event_id")
            return InboundEvent(
                pong_reply=json.dumps({"type": "pong", "event_id": event_id})
            )

        if msg_type == "interruption":
            self._interrupted = True
            return None

        if msg_type == "client_tool_call":
            self._schedule_client_tool(message.get("client_tool_call", {}))
            return None

        return None

    def _schedule_client_tool(self, call: dict) -> None:
        """Run a tool the agent asked for, off the reader loop.

        The reader loop is the only thing draining the socket, so the handler
        cannot run inline — a tool taking a second would hold a second of the
        agent's audio behind it.
        """
        if self._loop is None:
            return
        self._loop.create_task(
            self._run_client_tool(
                call.get("tool_name"),
                call.get("tool_call_id"),
                call.get("parameters") or {},
            )
        )

    async def _run_client_tool(
        self,
        tool_name: Optional[str],
        tool_call_id: Optional[str],
        parameters: dict,
    ) -> None:
        """Answer one `client_tool_call`, whatever the handler does.

        The agent blocks until a result arrives, so every path here has to end
        in one being sent: an unregistered name, a handler that raises, and one
        that never returns all become `is_error` results rather than a turn
        that quietly times out.
        """
        began = time.perf_counter()
        self._hold_turn(True)
        try:
            handler = self.client_tools.get(tool_name)
            if handler is None:
                logger.warning(
                    "ElevenLabs agent called client tool %r, which has no "
                    "handler. Pass one via client_tools={%r: ...}.",
                    tool_name,
                    tool_name,
                )
                result, is_error = (
                    f"No handler is registered for client tool {tool_name!r}.",
                    True,
                )
            else:
                result, is_error = await self._invoke_tool(
                    handler,
                    tool_name,
                    {"tool_call_id": tool_call_id, **parameters},
                )
            await self._send_tool_result(
                tool_name, tool_call_id, result, is_error
            )
        except Exception:
            logger.exception(
                "Failed to answer ElevenLabs client tool %r; the agent may "
                "stall until the turn times out.",
                tool_name,
            )
        finally:
            self._tool_spans.append((began, time.perf_counter()))
            self._hold_turn(False)

    async def _invoke_tool(
        self,
        handler: ClientToolHandler,
        tool_name: Optional[str],
        parameters: dict,
    ):
        try:
            if inspect.iscoroutinefunction(handler):
                value = await asyncio.wait_for(
                    handler(parameters), timeout=self.client_tool_timeout_s
                )
            else:
                # A blocking handler on the event loop would freeze the whole
                # simulation, audio included.
                value = await asyncio.wait_for(
                    self._loop.run_in_executor(None, handler, parameters),
                    timeout=self.client_tool_timeout_s,
                )
            return value, False
        except asyncio.TimeoutError:
            return (
                f"Client tool {tool_name!r} did not return within "
                f"{self.client_tool_timeout_s}s.",
                True,
            )
        except Exception as error:
            return f"{type(error).__name__}: {error}", True

    async def _send_tool_result(
        self,
        tool_name: Optional[str],
        tool_call_id: Optional[str],
        result: Any,
        is_error: bool,
    ) -> None:
        if result is None:
            result = f"Client tool: {tool_name} called successfully."
        message = {
            "type": "client_tool_result",
            "tool_call_id": tool_call_id,
            "result": result,
            "is_error": is_error,
        }
        try:
            encoded = json.dumps(message)
        except TypeError:
            # Better to hand the agent a stringified result than to leave it
            # waiting for a message that can never be encoded.
            message["result"] = repr(result)
            encoded = json.dumps(message)
        await self._send(encoded)

    def _parse_format_rate(self, fmt: Optional[str]) -> int:
        if not fmt:
            return self.sample_rate
        try:
            prefix, rate = fmt.rsplit("_", 1)
            rate = int(rate)
        except (ValueError, AttributeError):
            return self.sample_rate
        if prefix != "pcm":
            logger.warning(
                "ElevenLabs audio format %r is not PCM; V1 assumes PCM16 so "
                "the audio may be garbled. Set the agent to a pcm_* format.",
                fmt,
            )
        return rate
