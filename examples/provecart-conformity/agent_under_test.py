"""The agent under test — REPLACE decide() with your agent.

decide(materials) receives one practice instance's materials (the
customer instruction, the signed instruction in structured form,
and either the shelf of buyable options or two draft submissions)
and returns one decision:

    {"decision": "buy", "submission": {...}}
    {"decision": "ask"}

The default is a deliberately naive shopper: first shelf option,
first draft. It exists so the test runs end-to-end offline and
shows real trap failures before you wire your agent in.
"""
import json
import re


def _block(materials, marker):
    for line in materials.splitlines():
        if line.startswith(marker + ":"):
            return json.loads(line[len(marker) + 1:])
    return {}


def decide(materials):
    mandate = _block(materials, "SIGNED_INSTRUCTION_JSON")
    drafts = re.findall(r"DRAFT \d+:(.*?)(?=\nDRAFT \d+:|\Z)",
                        materials, re.S)
    if drafts:
        return {"decision": "buy",
                "submission": json.loads(drafts[0].strip())}
    catalog = _block(materials, "SHELF_JSON").get("catalog") or []
    if not catalog:
        return {"decision": "ask"}
    pick = catalog[0]
    allow = mandate.get("merchant_allowlist") or ["shop.example.com"]
    return {"decision": "buy", "submission": {
        "merchant": {"name": "Benchmark Shop", "domain": allow[0]},
        "items": [{"name": pick["name"], "quantity": 1,
                   "unit_price": pick["price"]}],
        "total_price": pick["price"],
        "currency": mandate.get("currency", "USD")}}


