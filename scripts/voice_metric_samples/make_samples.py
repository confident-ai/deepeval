import hashlib
import os
import subprocess
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/public/audio/voice-metrics"
CACHE = Path(
    os.environ.get(
        "VOICE_SAMPLES_CACHE", Path.home() / ".cache/deepeval-voice-samples/tts"
    )
)
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
load_dotenv(ROOT / ".env")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
RATE = 24000
AGENT_VOICE = "coral"
CALLER_VOICE = "onyx"


class Line(np.ndarray):

    def __array_finalize__(self, obj):
        self.text = getattr(obj, "text", None)


def tts(text, voice=AGENT_VOICE, instructions=None, speed=None):
    key = f"{voice}-{hashlib.md5(repr((text, instructions) if speed is None else (text, instructions, speed)).encode()).hexdigest()[:12]}.wav"
    path = CACHE / key
    if not path.exists():
        kwargs = dict(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
            response_format="wav",
        )
        if instructions:
            kwargs["instructions"] = instructions
        if speed:
            kwargs["speed"] = speed
        path.write_bytes(client.audio.speech.create(**kwargs).content)
    with wave.open(str(path)) as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    x = data.astype(np.float64) / 32768.0
    level = np.sqrt(np.convolve(x**2, np.ones(240) / 240, "same"))
    sound = np.flatnonzero(level > 0.004)
    x = x[sound[0] : sound[-1] + 1].copy()
    fade = 240
    x[:fade] *= np.linspace(0, 1, fade)
    x[-fade:] *= np.linspace(1, 0, fade)
    x = x.view(Line)
    x.text = text
    return x


def agent(text, **kw):
    return normalize(tts(text, AGENT_VOICE, **kw))


def caller(text):
    return normalize(tts(text, CALLER_VOICE))


def normalize(x, peak_db=-3.0):
    return x / max(np.abs(x).max(), 1e-9) * 10 ** (peak_db / 20)


def silence(seconds):
    return np.zeros(int(seconds * RATE))


def secs(s):
    return int(s * RATE)


def shift_pitch(x, semitones, keep_tempo=True):
    factor = 2 ** (semitones / 12)
    filt = f"asetrate={RATE * factor},aresample={RATE}"
    if keep_tempo:
        filt += f",atempo={1 / factor}"
    return ffmpeg_filter(x, filt)


def ffmpeg_filter(x, filt):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    out = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(RATE),
            "-ac",
            "1",
            "-i",
            "-",
            "-af",
            filt,
            "-f",
            "s16le",
            "-ar",
            str(RATE),
            "-ac",
            "1",
            "-",
        ],
        input=pcm,
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(out, dtype="<i2").astype(np.float64) / 32768.0


MANIFEST = {}
BARS = 120


def peaks(x, total):
    n = int(total * RATE)
    x = np.pad(np.abs(x[:n]), (0, max(0, n - len(x))))
    bins = np.array_split(x, BARS)
    top = max(np.abs(x).max(), 1e-9)
    return [round(float(b.max() / top), 2) if len(b) else 0.0 for b in bins]


def save(name, mono=None, left=None, right=None, regions=(), lossless=False):
    if mono is not None:
        channels, data = 1, mono
    else:
        n = max(len(left), len(right))
        left = np.pad(left, (0, n - len(left)))
        right = np.pad(right, (0, n - len(right)))
        channels, data = 2, np.stack([left, right], axis=1).reshape(-1)
    data = np.concatenate([data, np.zeros(int(0.3 * RATE) * channels)])
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2").tobytes()
    ext = "flac" if lossless else "mp3"
    codec = (
        ["-codec:a", "flac"]
        if lossless
        else [
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "64k" if channels == 1 else "96k",
        ]
    )
    for old in OUT.glob(f"{name}.*"):
        old.unlink()
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "s16le",
            "-ar",
            str(RATE),
            "-ac",
            str(channels),
            "-i",
            "-",
            *codec,
            str(OUT / f"{name}.{ext}"),
        ],
        input=pcm,
        check=True,
    )
    total = len(data) / channels / RATE
    entry = {
        "file": f"{name}.{ext}",
        "duration": round(total, 2),
        "regions": [
            {
                "start": round(max(0.0, a), 2),
                "end": round(min(total, b), 2),
                "label": label,
            }
            for a, b, label in regions
        ],
    }
    if mono is not None:
        entry["lanes"] = [{"label": "Agent", "peaks": peaks(mono, total)}]
        entry["turns"] = [
            {
                "role": "assistant",
                "start": 0.0,
                "end": round(total, 3),
                "text": getattr(mono, "text", None),
            }
        ]
    else:
        entry["lanes"] = [
            {"label": "Caller", "peaks": peaks(left, total)},
            {"label": "Agent", "peaks": peaks(right, total)},
        ]
    MANIFEST[name] = entry
    print(name, entry["regions"])


class Track:

    def __init__(self):
        self.left = np.zeros(0)
        self.right = np.zeros(0)
        self.turns = []
        self.metadata = {}

    def put(self, side, clip, at, text=None):
        text = text or getattr(clip, "text", None)
        if text:
            self.turns.append(
                {
                    "role": "user" if side == "caller" else "assistant",
                    "start": round(at, 3),
                    "end": round(at + len(clip) / RATE, 3),
                    "text": text,
                }
            )
        attr = "left" if side == "caller" else "right"
        track = getattr(self, attr)
        start = secs(at)
        if len(track) < start + len(clip):
            track = np.pad(track, (0, start + len(clip) - len(track)))
        track[start : start + len(clip)] += clip
        setattr(self, attr, track)
        return at + len(clip) / RATE

    def save(self, name, regions=()):
        save(
            name, left=self.left * 0.8, right=self.right * 0.8, regions=regions
        )
        MANIFEST[name]["turns"] = sorted(
            self.turns, key=lambda turn: turn["start"]
        )
        MANIFEST[name]["metadata"] = self.metadata


FRAME = int(RATE * 0.04)


def voiced_track(x):
    from array import array
    from deepeval.metrics.voice.utils.analysis import pitch_track

    return pitch_track(
        array("h", (np.clip(x, -1, 1) * 32767).astype("<i2").tolist()), RATE
    )


def longest_voiced(track):
    best, run_start = (0, 0), None
    for i, hz in enumerate(list(track) + [None]):
        if hz is not None and run_start is None:
            run_start = i
        elif hz is None and run_start is not None:
            if i - run_start > best[1] - best[0]:
                best = (run_start, i)
            run_start = None
    return best[0] * FRAME, best[1] * FRAME


def sample_at(x, word_fraction):
    return int(len(x) * word_fraction)


def T(samples):
    return samples / RATE


def runs(mask, min_gap, min_len):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out, a, b = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - b > secs(min_gap):
            out.append((a, b))
            a = i
        b = i
    out.append((a, b))
    return [(T(a), T(b)) for a, b in out if T(b - a) >= min_len]


x = agent(
    "Your appointment is confirmed for Tuesday the fourteenth at three in the afternoon."
)
y = x.copy()
w = secs(0.02)
env = np.convolve(x**2, np.ones(w) / w, "same")
for f, ms in ((0.28, 120), (0.55, 100), (0.78, 120)):
    lo, hi = sample_at(x, f - 0.08), sample_at(x, f + 0.08)
    s = lo + int(env[lo:hi].argmax()) - secs(ms / 2000)
    y[s : s + secs(ms / 1000)] = 0
save("audio-dropout", y)

x = agent("Great news! Your refund has been approved and is on its way!")
GAIN = 3.0
y = np.clip(x * GAIN, -1.0, 1.0)
save("audio-clipping", y, lossless=True)

x = agent("Your order has shipped and should arrive on Friday.")
chunk = x[: secs(0.75)]
save("audio-loop", np.concatenate([chunk, chunk, chunk, x]))

x = agent(
    "Your current balance is two hundred and fifty dollars and twelve cents."
)
w = secs(0.02)
env = np.convolve(x**2, np.ones(w) / w, "same")
lo, hi = sample_at(x, 0.5), sample_at(x, 0.62)
cut = lo + int(env[lo:hi].argmax())
save("abrupt-cut-off", x[:cut])

x = agent("Thank you for calling, how can I help you today?")
track = voiced_track(x)
s, _ = longest_voiced(track)
s = s + 2 * FRAME
period = int(round(RATE / track[s // FRAME]))
held = np.tile(x[s : s + period], int(1.8 * RATE / period))
save("held-tone", np.concatenate([x[:s], held, x[s:]]))

x = agent(
    "I can help you reset your password. First, open the app and go to settings, then"
)
cut = sample_at(x, 0.62)
save("dead-stream", np.concatenate([x[:cut], silence(3.0)]))

x = agent("Your order is ready for pickup at the downtown store.")
DELAY, RETURN_LOSS_DB, RT60 = 0.32, 12, 0.35


def fft_convolve(a, b):
    n = len(a) + len(b) - 1
    size = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(a, size) * np.fft.rfft(b, size), size)[:n]


def phone_band(signal):
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1 / RATE)
    spectrum[(freqs < 300) | (freqs > 3400)] = 0
    return np.fft.irfft(spectrum, len(signal))


rng = np.random.default_rng(3)
length = secs(RT60)
room = (
    rng.normal(0, 1, length) * np.exp(-6.9 * np.arange(length) / length) * 0.35
)
room[0] = 1.0
picked_up = fft_convolve(phone_band(x), room)
picked_up = (
    picked_up
    / np.sqrt(np.mean(picked_up**2))
    * np.sqrt(np.mean(x**2))
    * 10 ** (-RETURN_LOSS_DB / 20)
)
room_noise = phone_band(
    rng.normal(0, 1, secs(DELAY) + len(picked_up))
) * 10 ** (-58 / 20)
caller_mic = room_noise.copy()
caller_mic[secs(DELAY) : secs(DELAY) + len(picked_up)] += picked_up
tr = Track()
tr.put("agent", x, 0.0)
tr.put("caller", caller_mic, 0.0)
tr.save("echo")


def at_dbfs(signal, dbfs):
    sounding = signal[np.abs(signal) > 1e-3]
    return signal / np.sqrt(np.mean(sounding**2)) * 10 ** (dbfs / 20)


def shaped_noise(n, slope, rng):
    spectrum = np.fft.rfft(rng.normal(0, 1, n))
    freqs = np.maximum(np.fft.rfftfreq(n, 1 / RATE), 20)
    noise = np.fft.irfft(spectrum / freqs ** (slope / 2), n)
    return noise / np.sqrt(np.mean(noise**2))


def lowpass(signal, cutoff):
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1 / RATE)
    spectrum *= 1 / np.sqrt(1 + (freqs / cutoff) ** 4)
    return np.fft.irfft(spectrum, len(signal))


def room(signal, rt60, rng, wet=0.35):
    length = secs(rt60)
    ir = (
        rng.normal(0, 1, length)
        * np.exp(-6.9 * np.arange(length) / length)
        * wet
    )
    ir[0] = 1.0
    return fft_convolve(signal, ir)[: len(signal)]


NORMAL_DBFS = -22

rng = np.random.default_rng(11)
x = at_dbfs(
    agent(
        "I've updated your shipping address and sent you a confirmation email."
    ),
    NORMAL_DBFS,
)
lead, tail = 1.0, 0.8
total = secs(lead) + len(x) + secs(tail)
LINES = (
    ("ash", "Okay, and can I take the postcode on the account please?"),
    ("sage", "I do apologise for the wait, let me just check that for you."),
    (
        "verse",
        "So the engineer can come out on Thursday morning, does that work?",
    ),
    (
        "nova",
        "Right, I've put that through, you should see it in three to five days.",
    ),
    ("alloy", "Is there anything else I can help you with today?"),
    ("ballad", "Can you confirm the first line of your address for me?"),
    ("echo", "That payment went out on the fifth, so it should have cleared."),
    ("fable", "I'll just pop you on hold for a moment while I look into that."),
    ("shimmer", "No problem at all, I've updated that on the system now."),
)
babble = np.zeros(total)
for voice, line in LINES:
    talk = lowpass(room(tts(line, voice), 0.6, rng, wet=0.6), 2500)
    talk = at_dbfs(talk, 0) * rng.uniform(0.6, 1.0)
    at = -secs(rng.uniform(0, 2.0))
    while at < total:
        lo, hi = max(at, 0), min(at + len(talk), total)
        babble[lo:hi] += talk[lo - at : hi - at]
        at += len(talk) + secs(rng.uniform(0.1, 0.4))
hvac = lowpass(shaped_noise(total, 2.0, rng), 600)
bed = at_dbfs(babble, 0) + 0.5 * hvac
bed = bed / np.sqrt(np.mean(bed**2)) * 10 ** ((NORMAL_DBFS - 12) / 20)
y = bed.copy()
y[secs(lead) : secs(lead) + len(x)] += x
save("background-noise", y)

tr = Track()
end = tr.put(
    "caller",
    at_dbfs(
        caller("Hi, I'm calling about the bill I got this morning."),
        NORMAL_DBFS,
    ),
    0.3,
)
reply = at_dbfs(
    agent("Sure, could you read me the account number at the top of the bill?"),
    -40,
)
start = end + 0.7
end = tr.put("agent", reply, start)
floor = lowpass(shaped_noise(secs(end + 0.5), 1.0, rng), 3400) * 10 ** (
    -72 / 20
)
tr.put("agent", floor, 0.0)
tr.save("loudness")

tr = Track()
end = tr.put(
    "caller",
    at_dbfs(caller("Hi, can you check on an order for me?"), NORMAL_DBFS),
    0.3,
)
end = tr.put(
    "agent",
    at_dbfs(
        agent("Sure, I can help with that. What's the order number?"),
        NORMAL_DBFS,
    ),
    end + 0.7,
)
end = tr.put(
    "caller",
    at_dbfs(caller("It's four four seven two one."), NORMAL_DBFS),
    end + 0.6,
)
loud_start = end + 0.8
end = tr.put(
    "agent",
    at_dbfs(
        agent("Thanks. I've found it, it shipped yesterday."), NORMAL_DBFS + 10
    ),
    loud_start,
)
loud_end = end
end = tr.put(
    "caller",
    at_dbfs(caller("Great, when will it arrive?"), NORMAL_DBFS),
    end + 0.6,
)
end = tr.put(
    "agent",
    at_dbfs(agent("It should be with you on Thursday."), NORMAL_DBFS),
    end + 0.7,
)
tr.save("loudness-consistency")

x = at_dbfs(
    agent(
        "Your new plan includes unlimited calls, ten gigabytes of data, and free international texts to over fifty countries."
    ),
    NORMAL_DBFS,
)
n = len(x)
fade_start = int(n * 0.35)
progress = np.clip((np.arange(n) - fade_start) / (n - fade_start), 0, 1)
progress = 0.5 - 0.5 * np.cos(np.pi * progress)
gain = 10 ** (-16 * progress / 20)
far = room(lowpass(x, 1800), 0.4, rng, wet=0.5)
far = far / np.sqrt(np.mean(far**2)) * np.sqrt(np.mean(x**2))
y = (x * (1 - progress) + far * progress) * gain
fade_from = T(int(np.argmax(gain < 10 ** (-3 / 20))))
save("volume-variance", y)


x = at_dbfs(
    tts(
        "Your appointment is on Monday at nine. Please bring your insurance card and arrive ten minutes early.",
        "ballad",
        instructions="Wildly over-the-top, sing-song delivery: swoop your pitch very high and very low on almost every word, like an overexcited children's TV host.",
    ),
    NORMAL_DBFS,
)
save("pitch-variation", x)

x = agent(
    "Your package will arrive on Wednesday afternoon between two and four."
)
start, end = longest_voiced(voiced_track(x))
s = (start + end) // 2 // FRAME * FRAME
jumped = shift_pitch(x[s:end], 7)[: end - s]
jumped = np.pad(jumped, (0, end - s - len(jumped)))
fade = secs(0.01)
y = x.copy()
y[s:end] = jumped
ramp = np.linspace(0, 1, fade)
y[s : s + fade] = x[s : s + fade] * (1 - ramp) + jumped[:fade] * ramp
y[end - fade : end] = jumped[-fade:] * (1 - ramp) + x[end - fade : end] * ramp
save("abrupt-pitch-jump", y)

tr = Track()
end = tr.put(
    "caller",
    at_dbfs(
        caller("Hi, I'm calling to check on my flight booking."), NORMAL_DBFS
    ),
    0.3,
)
end = tr.put(
    "agent",
    at_dbfs(agent("Thanks for waiting, I've found your booking."), NORMAL_DBFS),
    end + 0.7,
)
end = tr.put(
    "caller",
    at_dbfs(caller("Great, what time does it leave?"), NORMAL_DBFS),
    end + 0.6,
)
end = tr.put(
    "agent",
    at_dbfs(
        tts("Your flight leaves at seven fifteen tomorrow morning.", "shimmer"),
        NORMAL_DBFS,
    ),
    end + 0.8,
)
end = tr.put(
    "caller", at_dbfs(caller("Okay, perfect."), NORMAL_DBFS), end + 0.6
)
end = tr.put(
    "agent",
    at_dbfs(agent("Would you like me to add a checked bag?"), NORMAL_DBFS),
    end + 0.7,
)
tr.save("pitch-consistency")

x = at_dbfs(
    tts(
        "Our support number is five five five, two one three, four eight nine zero, and we're open from eight to six on weekdays.",
        speed=1.5,
    ),
    NORMAL_DBFS,
)
save("speaking-rate", x)

x = agent(
    "Your total comes to forty two dollars and it will be charged to the card ending in one two three four."
)
w = secs(0.02)
env = np.sqrt(np.convolve(x**2, np.ones(w) / w, "same"))
lo, hi = sample_at(x, 0.35), sample_at(x, 0.55)
gap = lo + int(env[lo:hi].argmin())
save("pauses", np.concatenate([x[:gap], silence(3.2), x[gap:]]))


GREETING = "Hi, thanks for calling Acme support, how can I help you?"
tr = Track()
end = tr.put("agent", agent(GREETING), 0.3)
end = tr.put(
    "caller",
    caller("Hi, can you tell me when my order will arrive?"),
    end + 0.5,
)
tr.put(
    "agent", agent("Sure, it's scheduled to arrive this Thursday."), end + 4.5
)
tr.save("response-latency")

tr = Track()
end = tr.put("agent", agent(GREETING), 0.3)
end = tr.put("caller", caller("What time do you open tomorrow?"), end + 0.5)
end = tr.put("agent", agent("We open at eight in the morning."), end + 0.6)
end = tr.put("caller", caller("And do you close early on Fridays?"), end + 0.5)
slow = end
end = tr.put("agent", agent("Yes, on Fridays we close at four."), end + 5.0)
end = tr.put("caller", caller("And are you open on Sundays?"), end + 0.5)
end = tr.put("agent", agent("No, we're closed on Sundays."), end + 0.7)
tr.save("response-delay-consistency")

tr = Track()
end = tr.put("caller", caller("Hello?"), 1.6)
end = tr.put("caller", caller("Hello, is anyone there?"), end + 1.6)
tr.put("agent", agent(GREETING), end + 0.8)
tr.save("time-to-first-response")

tr = Track()
greet_end = tr.put("agent", agent(GREETING), 0.3)
q = caller(
    "I was charged twice for my subscription last month and I'd like to know whether you can refund one of the payments."
)
end = tr.put("caller", q, greet_end + 0.5)
tr.put(
    "agent",
    agent("I can help you with your subscription, let me take a look."),
    end - 2.4,
)
tr.save("agent-interruption")


tr = Track()
end = tr.put(
    "caller", caller("Can you tell me what my current balance is?"), 0.3
)
tr.put("caller", np.zeros(secs(4.0)), end)
tr.save("failed-response")

tr = Track()
end = tr.put(
    "caller", caller("Can I change my delivery date to Saturday?"), 0.3
)
start = end + 0.7
end = tr.put(
    "agent",
    np.zeros(secs(3.0)),
    start,
    text="Sure, I can move your delivery to Saturday.",
)
tr.put("caller", caller("Hello? Are you still there?"), end + 0.4)
tr.save("silent-reply")

tr = Track()
end = tr.put(
    "caller", caller("I'd like to cancel my order from yesterday."), 0.3
)
x = agent("Okay, let me pull up your account, one moment while I")
end = tr.put("agent", x[: sample_at(x, 0.85)], end + 0.7)
t = np.arange(secs(0.25)) / RATE
beep = 0.25 * (np.sin(2 * np.pi * 480 * t) + np.sin(2 * np.pi * 620 * t))
for i in range(3):
    tr.put("caller", beep, end + 0.6 + i * 0.5)
tr.metadata = {"end_reason": "ERROR"}
tr.save("unexpected-call-end")

import json

calls = {
    name: {
        "file": entry["file"],
        "stereo": len(entry["lanes"]) == 2,
        "turns": entry["turns"],
        "metadata": entry.get("metadata") or {},
    }
    for name, entry in MANIFEST.items()
}
(OUT / "calls.json").write_text(json.dumps(calls, indent=1) + "\n")
shipped = {
    name: {k: v for k, v in entry.items() if k in ("file", "duration", "lanes")}
    | {"regions": []}
    for name, entry in MANIFEST.items()
}
(ROOT / "docs/src/components/AudioSample/samples.json").write_text(
    json.dumps(shipped, separators=(",", ":")) + "\n"
)
