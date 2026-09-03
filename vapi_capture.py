"""Capture Vapi's WebSocket event vocabulary for one short conversation.

Run this once against a scratch assistant. It creates a `vapi.websocket` call,
speaks two utterances into it, and writes every inbound message to
`vapi_events.jsonl` — binary audio as a summary line, JSON control messages
verbatim. The log answers the four questions VapiConnector needs:

  1. Is there a per-turn completion event to trust instead of silence?
  2. Do final transcripts arrive, so STT can be skipped?
  3. Is there a ready/handshake message before audio is accepted?
  4. Does the assistant speak first, before we send anything?

Usage:
    VAPI_API_KEY=... VAPI_ASSISTANT_ID=... OPENAI_API_KEY=... python vapi_capture.py
"""

import asyncio
import json
import os
import sys
import time

import aiohttp

from deepeval.models import OpenAITTSModel
from deepeval.voice.connectors import audio_utils

API_BASE = os.getenv("VAPI_API_URL", "https://api.vapi.ai")
SAMPLE_RATE = 24000
CHUNK_MS = 20
LOG_PATH = "vapi_events.jsonl"

UTTERANCES = [
    "Hi, I'd like to book a table for two people tomorrow at seven pm.",
    "Actually, could you make that eight pm instead?",
]

# How long to listen after each utterance before speaking again.
LISTEN_SECONDS = 15.0
# How long to listen before the first utterance, to see whether the assistant
# opens the conversation on its own.
GREETING_SECONDS = 8.0


def log(record: dict, started: float) -> None:
    record = {"at_s": round(time.perf_counter() - started, 3), **record}
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    if record.get("kind") != "audio":
        print(json.dumps(record)[:400])


async def create_call(
    session: aiohttp.ClientSession, api_key: str, assistant_id: str
) -> str:
    payload = {
        "assistantId": assistant_id,
        "transport": {
            "provider": "vapi.websocket",
            "audioFormat": {
                "format": "pcm_s16le",
                "container": "raw",
                "sampleRate": SAMPLE_RATE,
            },
        },
    }
    async with session.post(
        f"{API_BASE}/call",
        headers={"authorization": f"Bearer {api_key}"},
        json=payload,
    ) as resp:
        body = await resp.text()
        if resp.status >= 300:
            raise SystemExit(
                f"Vapi call creation failed ({resp.status}): {body}"
            )
        data = json.loads(body)
    print(json.dumps({"created_call": data}, indent=2)[:2000])
    url = (data.get("transport") or {}).get("websocketCallUrl")
    if not url:
        raise SystemExit("Response had no transport.websocketCallUrl")
    return url


async def synthesize_pcm(tts: OpenAITTSModel, text: str) -> bytes:
    audio, _cost = await tts.a_synthesize(text)
    pcm, rate, channels = audio_utils.wav_bytes_to_pcm16(audio.get_bytes())
    pcm = audio_utils.downmix_to_mono(pcm, channels)
    if rate != SAMPLE_RATE:
        pcm = audio_utils.resample_pcm16(pcm, rate, SAMPLE_RATE)
    return pcm


async def send_pcm(ws: aiohttp.ClientWebSocketResponse, pcm: bytes) -> None:
    """Stream in real time, as a caller's microphone would."""
    frame_bytes = int(SAMPLE_RATE * 2 * CHUNK_MS / 1000)
    for offset in range(0, len(pcm), frame_bytes):
        await ws.send_bytes(pcm[offset : offset + frame_bytes])
        await asyncio.sleep(CHUNK_MS / 1000)


async def reader(ws: aiohttp.ClientWebSocketResponse, started: float) -> None:
    audio_bytes = 0
    audio_messages = 0
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.BINARY:
            audio_bytes += len(msg.data)
            audio_messages += 1
            # One line per second of audio keeps the log readable.
            if audio_messages % 50 == 0:
                log(
                    {
                        "kind": "audio",
                        "messages": audio_messages,
                        "total_bytes": audio_bytes,
                    },
                    started,
                )
        elif msg.type == aiohttp.WSMsgType.TEXT:
            try:
                parsed = json.loads(msg.data)
            except ValueError:
                log({"kind": "text_unparsed", "raw": msg.data[:2000]}, started)
                continue
            log({"kind": "event", "message": parsed}, started)
        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
            log({"kind": "socket_closed", "type": str(msg.type)}, started)
            return


async def main() -> None:
    api_key = os.getenv("VAPI_API_KEY")
    assistant_id = os.getenv("VAPI_ASSISTANT_ID")
    if not api_key or not assistant_id:
        raise SystemExit("Set VAPI_API_KEY and VAPI_ASSISTANT_ID")

    open(LOG_PATH, "w").close()
    tts = OpenAITTSModel()
    print("Synthesizing utterances...")
    utterance_pcm = [await synthesize_pcm(tts, text) for text in UTTERANCES]

    async with aiohttp.ClientSession() as session:
        url = await create_call(session, api_key, assistant_id)
        print(f"Connecting to {url}")
        async with session.ws_connect(url) as ws:
            started = time.perf_counter()
            read_task = asyncio.create_task(reader(ws, started))

            log(
                {
                    "kind": "note",
                    "note": "listening for an unprompted greeting",
                },
                started,
            )
            await asyncio.sleep(GREETING_SECONDS)

            for index, pcm in enumerate(utterance_pcm):
                log(
                    {
                        "kind": "note",
                        "note": f"sending utterance {index + 1}",
                        "text": UTTERANCES[index],
                        "pcm_bytes": len(pcm),
                    },
                    started,
                )
                await send_pcm(ws, pcm)
                log(
                    {"kind": "note", "note": f"utterance {index + 1} sent"},
                    started,
                )
                await asyncio.sleep(LISTEN_SECONDS)

            await ws.send_str(json.dumps({"type": "end-call"}))
            log({"kind": "note", "note": "sent end-call"}, started)
            await asyncio.sleep(2.0)
            await ws.close()
            await read_task

    print(f"\nWrote {LOG_PATH}")
    print("Paste it back, or grep it for distinct event types:")
    print(
        "  jq -r 'select(.kind==\"event\") | .message.type' "
        f"{LOG_PATH} | sort | uniq -c"
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(1)
