# Tracing with Arize Phoenix

Every request to the AI service can be followed in [Arize Phoenix](https://arize.com/docs/phoenix): which
stages ran, what each took, what the model and the retriever saw, how many tokens were used and what it cost.
The spans follow the OpenInference conventions, so Phoenix shows them with its agent, retriever, reranker,
embedding, LLM and guardrail views.

Tracing is **off by default** and never changes behavior. If it is off, or the collector is down, requests are
answered exactly as before.

## Turn it on

| Variable | Meaning | Default |
|---|---|---|
| `PHOENIX_COLLECTOR_ENDPOINT` | Phoenix URL. Setting it turns tracing on. `/v1/traces` is added for you. | empty (off) |
| `PHOENIX_API_KEY` | Phoenix Cloud or an authenticated Phoenix. Leave empty for a local one. | empty |
| `PHOENIX_PROJECT_NAME` | Project the traces appear under. | `finpass-ai` |
| `TRACING_ENABLED` | `false` forces tracing off even with an endpoint. | unset |
| `TRACING_CAPTURE_CONTENT` | Export prompts, questions and answers. | on in development, **off in production** |
| `OTEL_TRACES_SAMPLER`, `OTEL_TRACES_SAMPLER_ARG` | Standard OpenTelemetry sampling, e.g. `parentbased_traceidratio` and `0.2`. | all traces |

Start a local Phoenix, then the service:

```bash
docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest      # UI at http://localhost:6006
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006 uv run uvicorn src.app:app --port 8001
```

Without Docker: `uvx --from arize-phoenix phoenix serve`.

Phoenix Cloud: set `PHOENIX_COLLECTOR_ENDPOINT` to your space URL (`https://app.phoenix.arize.com/s/<space>`) and
`PHOENIX_API_KEY`. A gRPC collector (port 4317) is also accepted.

## What one chat turn looks like

```
advisor_chat                         AGENT root (no HTTP server spans)
└─ advisor_chat                [AGENT]       question, answer; totals for tokens and cost
   ├─ retrieve_context         [CHAIN]
   │  ├─ plan_retrieval        [CHAIN]       intent, tickers, source types, years, expanded query
   │  ├─ gather_candidates     [CHAIN]       candidates before and after language dedupe
   │  │  └─ vector_search      [RETRIEVER]   Chroma filters; returned documents with id, title, page, text
   │  │     └─ embed_query     [EMBEDDING]   bge-m3, tokens, cost
   │  ├─ rerank_passages       [RERANKER]    model; input and output documents with scores
   │  └─ select_passages       [CHAIN]       score cutoff, year balance, top n
   ├─ add_news_context         [CHAIN]       news used when fewer than 3 documents were found
   ├─ generate_answer          [CHAIN]       attempt 1 (attempt 2 appears only after a refusal)
   │  ├─ chat_completion       [LLM]         messages, answer, tokens, cost, finish reason
   │  ├─ guardrail_check       [GUARDRAIL]   passed, violations, citation markers
   │  └─ answer_ready          [CHAIN]       source, grounded, citations
   ├─ source_excerpts_fallback / answer_not_found   [CHAIN]   only when the answer was refused twice or nothing was found
   └─ turn_title:<step> ...    [LLM]         the small calls that write the progress titles
```

The profile explanation is `explain_profile` [AGENT] with `build_facts`, `explain_profile_wording` [LLM],
`parse_output`, `guardrail_check` [GUARDRAIL] and, when the model is off or refused, `template_fallback`.

### Cost and tokens

Each request asks OpenRouter to report usage (`usage.include`), so every LLM and embedding span carries
`llm.token_count.prompt/completion/total` and `llm.cost.total` in USD, the amount OpenRouter charged. Phoenix adds
them up per trace. The root span also carries the sums as `metadata.cost_usd`, `metadata.tokens_total`,
`metadata.llm_calls`, `metadata.embedding_calls` and `metadata.priced_calls` (calls for which a cost was reported).

### Finding things in Phoenix

* **Sessions**: `session.id` is a hash of the run id, so one user's turns group together without the user id.
* **Why was an answer refused?** Open `guardrail_check`: `metadata.violations` and `metadata.passed`.
* **Why these documents?** `rerank_passages` shows candidates in and ranked out with scores, `select_passages` the cutoff.
* **Why is it slow?** Compare the durations of `embed_query`, `vector_search`, `rerank_passages` and `chat_completion`.
* **Which questions cost most?** Sort traces by cost; filter with `metadata.intent`, `metadata.source`, `metadata.grounded`.

## Privacy

Prompts contain a person's income, savings and holdings. With `TRACING_CAPTURE_CONTENT` off (the production
default) Phoenix receives only metadata: models, token counts, costs, scores, timings, intents, document ids and
titles, and the outcome of each check. Source documents are public, so retrieved text is shown even then. Turn
content on only in development, or in a production Phoenix that you treat like the database. A Phoenix Cloud space
is a third party that would receive that data. Names and amounts are never put in metadata.

## Checks

```bash
uv run pytest tests/test_tracing.py     # span tree, nesting, tokens, cost, hidden content, no-op when off
```

If spans do not arrive: look for `Failed to export spans` in the service log. A `405` means the URL is wrong
(the service adds `/v1/traces`; do not point at a different path). A `401` means the API key is missing or wrong.


## Cost shows 0 in Phoenix

Phoenix computes cost itself, from token counts and its own price table. It ignores the cost we put on the span
(`llm.cost.total` and the `cost_usd` in the root metadata stay, as the exact figure OpenRouter charged). Models
reached through OpenRouter are mostly missing from that table, so their cost is 0 until they are added. Run once
per Phoenix instance, and again after changing a model:

    PHOENIX_COLLECTOR_ENDPOINT=... PHOENIX_API_KEY=... uv run python scripts/phoenix_sync_prices.py

It reads the prices from OpenRouter and creates the chat, answer and embedding models in Phoenix. Models it cannot
find are listed; add those by hand under Settings > Models.

## Empty input and output

With `TRACING_CAPTURE_CONTENT` off (the production default) the input and output columns show
`[content hidden: N chars ...]` instead of the text. Set it to `true` to see the real text, knowing that prompts
carry the user's financial data and go to the Phoenix host. There are no HTTP server spans any more; each trace
starts at `advisor_chat` or `explain_profile`.
