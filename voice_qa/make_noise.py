import array
import math
import os
import random
import wave

RATE = 24000
SECONDS = 30
PEAK = 32767


def _write(name, samples):
    path = os.path.join(os.path.dirname(__file__), "noise", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clipped = array.array(
        "h", (max(-PEAK, min(PEAK, int(s))) for s in samples)
    )
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(clipped.tobytes())
    print(f"wrote {path}")


def white(level=0.25):
    return [random.uniform(-1, 1) * PEAK * level for _ in range(RATE * SECONDS)]


def street(level=0.35):
    samples = []
    last = 0.0
    for _ in range(RATE * SECONDS):
        last = 0.995 * last + 0.005 * random.uniform(-1, 1)
        samples.append(last * PEAK * level * 20)
    return samples


def cafe(level=0.3):
    samples = [0.0] * (RATE * SECONDS)
    rng = random.Random(7)
    for _ in range(220):
        start = rng.randrange(0, RATE * (SECONDS - 2))
        length = rng.randrange(RATE // 4, RATE * 2)
        pitch = rng.uniform(90, 350)
        loud = rng.uniform(0.2, 1.0)
        for i in range(length):
            env = math.sin(math.pi * i / length)
            wobble = math.sin(2 * math.pi * pitch * i / RATE)
            wobble += 0.5 * math.sin(2 * math.pi * pitch * 2.1 * i / RATE)
            samples[start + i] += wobble * env * loud
    peak = max(abs(s) for s in samples) or 1.0
    return [s / peak * PEAK * level for s in samples]


if __name__ == "__main__":
    _write("white.wav", white())
    _write("street.wav", street())
    _write("cafe.wav", cafe())
