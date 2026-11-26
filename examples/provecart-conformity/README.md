# ProveCart conformity tests for DeepEval / pytest

The ProveCart public practice set as a ready-to-run test suite:
12 customer instructions with planted traps (wrong brand, wrong
fit, over-limit quantity, blocklisted merchant, fees over the
ceiling, wrong fare terms, wrong date, wrong reservation name,
ambiguous instructions). Your agent decides; a deterministic
oracle scores the decision — no LLM judge, no API key, no network.

```bash
pip install -r requirements.txt
pytest test_provecart_conformity.py
# or, with DeepEval reporting:
deepeval test run test_provecart_conformity.py
```

Replace `decide()` in `agent_under_test.py` with a call to your
agent. It receives the instance materials and returns one JSON
decision: `{"decision": "buy", "submission": {...}}` or
`{"decision": "ask"}`. Scoring mirrors the live ProveCart Verified
Agent exam's safe-decision rule (the live pass bar is 80+).

Files: `test_provecart_conformity.py` (suite + a DeepEval
`BaseMetric` wrapper used when DeepEval is installed),
`agent_under_test.py` (replaceable naive stub), `oracle.py`
(framework-free scorer), `materials.json` (generated per-instance
prompts; shelf/draft order shuffled), `practice-set.json`
(byte-identical to https://provecart.com/practice-set.json).

The live exam — fresh instances per sitting, public scoreboard,
90-day certification — is at https://provecart.com/agents.
Published by ProveCart, a verification service for agent commerce
based in the Bay Area. Patent pending.
