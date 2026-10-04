# llm-endpoint-bench

Benchmark any OpenAI-compatible chat-completions endpoint.

## Why

Two gateways can offer the same model at the same headline price and behave
completely differently. What actually matters to a user is:

- **TTFT** (time to first token) — how long until something appears
- **throughput** — tokens per second once it starts
- **reliability** — how often it just fails

A gateway that is 20% cheaper but fails 1 in 8 requests is not cheaper.

## What it measures

For each endpoint × model:

| Metric | Notes |
|---|---|
| TTFT | p50 / p90, streaming |
| throughput | output tokens / second, excluding TTFT |
| error rate | by HTTP status |
| p95 latency | full response |

## Usage

    python bench.py --endpoint https://example.com/v1 --model gpt-4o-mini \
                    --api-key $KEY --requests 20

## Status

Early.
