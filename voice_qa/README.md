# Voice QA matrix 

Run deepeval's voice simulation machinery with interruptions and background noise, against scripted agents whose defects are known in advance. If deepeval reports something we did not plant or misses something we did, the machinery needs improvement.

## What each cell tests


| Cell                    | Agent defect                    | Persona                  | Ground truth we assert                                                        |
| ----------------------- | ------------------------------- | ------------------------ | ----------------------------------------------------------------------------- |
| control_off             | none (3 short polite lines)     | no interruptions         | zero interrupted turns, >=2 spoken replies                                    |
| rambler_frequent_insist | 40s monologue                   | frequent + insist        | at least one interrupted turn                                                 |
| rambler_frequent_yield  | 40s monologue                   | frequent + yield         | recorded only: caller should back off (listen)                                |
| rambler_normal_adaptive | 40s monologue                   | normal + adaptive        | recorded only: awkward-silence restart (listen)                               |
| corpse_hangup           | answers once, then pure silence | hold_timeout=20s         | <=2 spoken replies, call ends well under 4 min                                |
| thinker_eager           | 3.5s pause mid-reply            | rare/yield (arms duplex) | first reply audio gets clipped at the pause                                   |
| thinker_patient         | 3.5s pause mid-reply            | rare/yield               | first reply audio survives the pause (cross-cell check)                       |
| noise_cafe_light        | none                            | cafe noise at 0.15       | call completes, >=2 spoken replies                                            |
| noise_cafe_heavy        | none                            | cafe noise at 0.5        | call completes; listen for noise under caller speech and during caller pauses |
| noise_white_heavy       | none                            | white noise at 0.5       | call completes; stresses the agent-side endpointing                           |


Every cell also runs TurnTakingNaturalness, AgentResponsiveness and VoiceNaturalness and records score/skipped/reason in the summary.

## Running

```
export OPENAI_API_KEY=...          # simulator LLM + TTS/STT

poetry run python voice_qa/run_matrix.py
poetry run python voice_qa/run_matrix.py --cells control_off,corpse_hangup # optional
poetry run python voice_qa/run_matrix.py --repeats 3 # optional($$$)
```

Outputs land in voice_qa/results//: per-cell WAV recordings (combined per conversation) and summary.json with facts, metric scores and PASS/FAIL checks.

