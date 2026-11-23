# Voice metric example clips

Generates the example clip shown on each voice metric's docs page. Needs an
`OPENAI_API_KEY` in the repo's `.env` (agent and caller speech come from
OpenAI TTS; responses are cached in `~/.cache/deepeval-voice-samples`).

```bash
python docs/scripts/voice-metric-samples/make_samples.py
python docs/scripts/voice-metric-samples/attach_ranges.py
```

`make_samples.py` writes the clips and `calls.json` (each clip's turns) to
`docs/public/audio/voice-metrics/`, and the waveforms to
`docs/src/components/AudioSample/samples.json`. `attach_ranges.py` runs each
metric on its published clip and stores where its events say the problems
are; the player highlights exactly those ranges.
