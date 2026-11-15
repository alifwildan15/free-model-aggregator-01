# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/alifwildan15/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/alifwildan15/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A zero-config server that serves the OpenAI Chat Completions API and the Anthropic Messages API over ten simulated free-tier providers, routing each request to the best free model that meets its requirements, with rate-limit-aware failover and a `--list` of the catalog it found.

## Run it

```sh
node main.ts
```

The trial sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE` and `MAKEFUN_TRIAL_SEED`. The server reads the brief, probes each provider's catalog, registers its port and catalog with the env, answers the twenty requests the env's client sends, and exits 0 when the workload is done. `node main.ts --list` prints the catalog document it registers, byte for byte; outside a trial it prints `{"providers":[]}`.

## How it routes

Each request's candidates are the free models that meet its requirements (tools, vision, JSON mode, minimum context), ranked by the quality table's rating. The twenty primaries are planned up front — weight-2 requests first, each taking the best model whose provider still has planned budget (the kit's published per-trial minimum per provider). A refused call never ends a request: the next best candidate is tried, a 402 marks its provider dead for the trial, and two 429s or 5xx in a row do the same. Every answer's text, usage and stop reason come back verbatim from the provider that answered, re-emitted in the wire the client used, in JSON or in server-sent events. All three upstream wires are spoken: OpenAI, Anthropic and Gemini. A JSON-mode request from an OpenAI client is never routed to an anthropic-wire provider, which cannot carry `response_format`.

## Layout

- `main.ts` — the server: the brief, the catalog, the plan, the routes, the register and the poll.
- `lib/providers.ts` — the brief's documents, the catalog probe, capabilities and requirements.
- `lib/wire.ts` — the translations between the OpenAI, Anthropic and Gemini wires, and the SSE writers.

No dependencies beyond the Node 22 standard library. No model calls of its own: the routing is a deterministic program, so a replay leaves the same state export, byte for byte.

---

<sub>Entered on [make.fun](https://make.fun/?ref=cf6zvk8s&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
