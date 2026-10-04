#!/usr/bin/env python3
"""
llm-endpoint-bench
==================

Benchmark any OpenAI-compatible chat-completions endpoint.

Why
---
Two gateways can offer the same model at the same headline price and behave
completely differently. What matters to a user is not the price on the page:

    time to first token   how long until *something* appears
    throughput            tokens per second once it starts
    reliability           how often it just fails

A gateway that is 20% cheaper but drops one request in eight is not cheaper.

What it measures
----------------
Per endpoint x model:

    TTFT p50 / p90      streaming, time to first content chunk
    throughput          output tokens / second, excluding the TTFT wait
    error rate          grouped by HTTP status
    full latency        p50 / p95 for the whole response

Usage
-----
    python bench.py --endpoint https://example.com/v1 --model gpt-4o-mini \\
                    --api-key $OPENAI_API_KEY --requests 10

    # compare two gateways serving the same model
    python bench.py --endpoint https://a.example/v1 --model gpt-4o-mini --api-key $A
    python bench.py --endpoint https://b.example/v1 --model gpt-4o-mini --api-key $B

The tool is intentionally dependency-free: urllib and the standard library only.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

PROMPT = "Write one short paragraph about why latency matters more than headline price."


def stream_once(endpoint, model, api_key, timeout=60, max_tokens=200):
    """Run one streaming request. Returns a result dict."""
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + api_key)
    req.add_header("Accept", "text/event-stream")
    req.add_header("User-Agent", "llm-endpoint-bench")

    started = time.monotonic()
    ttft = None
    chunks = 0
    text = ""

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.monotonic() - started
                    chunks += 1
                    text += piece
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "error": e.read().decode()[:200]}
    except Exception as e:
        return {"ok": False, "status": 0, "error": str(e)}

    total = time.monotonic() - started

    # Rough token count: whitespace words are close enough for a throughput
    # estimate across providers, and avoid pulling in a tokenizer.
    approx_tokens = max(1, len(text.split()))
    stream_time = max(0.001, total - (ttft or 0))

    return {
        "ok": True,
        "status": 200,
        "ttft": ttft,
        "total": total,
        "tokens": approx_tokens,
        "throughput": approx_tokens / stream_time,
        "text_len": len(text),
    }


def pct(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1)))))
    return values[k]


def main():
    ap = argparse.ArgumentParser(description="Benchmark OpenAI-compatible endpoints.")
    ap.add_argument("--endpoint", required=True, help="base url, e.g. https://x.example/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--requests", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    results = []
    for i in range(args.requests):
        r = stream_once(args.endpoint, args.model, args.api_key, args.timeout)
        results.append(r)
        if r["ok"]:
            print("  #%02d  ttft=%.2fs  total=%.2fs  ~%d tok  %.1f tok/s"
                  % (i + 1, r["ttft"], r["total"], r["tokens"], r["throughput"]))
        else:
            print("  #%02d  FAILED  status=%s  %s"
                  % (i + 1, r["status"], r.get("error", "")[:90]))

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]

    print()
    print("=" * 62)
    print("endpoint : %s" % args.endpoint)
    print("model    : %s" % args.model)
    print("requests : %d ok, %d failed (error rate %.1f%%)"
          % (len(ok), len(fail), 100.0 * len(fail) / max(1, len(results))))

    if fail:
        by_status = {}
        for r in fail:
            by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        print("failures : %s" % ", ".join("HTTP %s x%d" % (k, v)
                                          for k, v in sorted(by_status.items())))

    if ok:
        ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
        totals = [r["total"] for r in ok]
        tps = [r["throughput"] for r in ok]
        if ttfts:
            print("ttft     : p50 %.2fs  p90 %.2fs" % (pct(ttfts, 50), pct(ttfts, 90)))
        print("total    : p50 %.2fs  p95 %.2fs" % (pct(totals, 50), pct(totals, 95)))
        print("through  : median %.1f tok/s  (approx)" % statistics.median(tps))

    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
