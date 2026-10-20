"""Registers the price of every model this service uses in Phoenix, so its Cost column is not 0.

Phoenix works out cost itself from token counts and its own price table; it ignores the cost we send on the
span. Models reached through OpenRouter (deepseek, gemma, bge-m3...) are mostly missing from that table.
This reads the public prices from OpenRouter and creates the missing models in Phoenix. Safe to run again.

    uv run python scripts/phoenix_sync_prices.py
"""

import json
import re
import sys
import urllib.error
import urllib.request

sys.path.insert(0, ".")

from src.core.config import settings  # noqa: E402
from src.infrastructure.advisor_chat import embedder  # noqa: E402

CREATE = """mutation($input: CreateModelMutationInput!) {
  createModel(input: $input) { model { name } }
}"""
LIST = "query { generativeModels { edges { node { name } } } }"


def per_m(pricing: dict, key: str) -> float:
    return float(pricing.get(key) or 0) * 1_000_000


def call(url: str, body: dict | None = None, key: str = "") -> dict:
    headers = {
        "Content-Type": "application/json",
        **({"Authorization": f"Bearer {key}"} if key else {}),
    }
    data = json.dumps(body).encode() if body else None
    with urllib.request.urlopen(urllib.request.Request(url, data, headers), timeout=30) as r:
        return json.load(r)


def main() -> int:
    if not settings.phoenix_collector_endpoint:
        print("PHOENIX_COLLECTOR_ENDPOINT is empty")
        return 1
    base = re.sub(r"/v1/traces/?$", "", settings.phoenix_collector_endpoint.rstrip("/"))
    key = settings.phoenix_api_key
    wanted = {
        settings.openrouter_model,
        settings.rag_model or settings.openrouter_model,
        embedder.EMBED_MODEL,
    }
    prices = {  # chat models and embedding models are listed separately
        m["id"].lower(): m["pricing"]
        for path in ("models", "embeddings/models")
        for m in call("https://openrouter.ai/api/v1/" + path)["data"]
    }
    existing = {
        e["node"]["name"]
        for e in call(base + "/graphql", {"query": LIST}, key)["data"]["generativeModels"]["edges"]
    }
    for name in sorted(wanted):
        pricing = prices.get(name.lower())
        if name in existing:
            print(f"skip   {name}: already in Phoenix")
        elif not pricing:
            print(
                f"MISSING {name}: not in the OpenRouter price list, add it by hand in Phoenix > Settings > Models"
            )
        else:
            costs = [
                {
                    "tokenType": "input",
                    "costPerMillionTokens": per_m(pricing, "prompt"),
                    "kind": "PROMPT",
                }
            ]
            costs.append(
                {
                    "tokenType": "output",
                    "costPerMillionTokens": per_m(pricing, "completion"),
                    "kind": "COMPLETION",
                }
            )
            out = call(
                base + "/graphql",
                {
                    "query": CREATE,
                    "variables": {
                        "input": {
                            "name": name,
                            "provider": "openrouter",
                            "namePattern": "^" + re.escape(name) + "$",
                            "costs": costs,
                        }
                    },
                },
                key,
            )
            print(
                f"error  {name}: {out['errors'][0]['message']}"
                if out.get("errors")
                else f"added  {name}: {costs}"
            )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (urllib.error.URLError, KeyError) as exc:
        print(f"failed: {exc}")
        sys.exit(1)
