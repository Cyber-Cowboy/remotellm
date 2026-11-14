#!/usr/bin/env python3
"""
Benchmark cold vs warm latency, time-to-first-token and tokens/sec through
the OpenAI-compatible proxy, using the real `openai` SDK.

Usage:
    pip install openai
    export PROXY_BASE_URL=https://your-proxy.example.com/v1
    export PROXY_API_KEY=...
    python3 scripts/benchmark.py --model gpt-oss-120b-uncensored --rounds 3
"""
import argparse
import os
import time

from openai import OpenAI

PROMPT = "In one short paragraph, explain what a GPU is."


def run_once(client: OpenAI, model: str, label: str):
    t0 = time.time()
    first_token_at = None
    n_tokens = 0
    text = []

    with client.chat.completions.stream(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=256,
    ) as stream:
        for event in stream:
            if event.type == "content.delta":
                if first_token_at is None:
                    first_token_at = time.time()
                n_tokens += 1
                text.append(event.delta)
        # Not calling stream.get_final_completion(): reasoning models (like
        # this one) can spend the whole max_tokens budget on hidden
        # "reasoning" content before any visible content, which the SDK's
        # strict parser treats as a truncation error even though we already
        # have everything we need (ttft/total/tokens) from the live stream.

    t1 = time.time()
    ttft = (first_token_at - t0) if first_token_at else float("nan")
    total = t1 - t0
    tok_s = n_tokens / (t1 - first_token_at) if first_token_at and t1 > first_token_at else float("nan")

    print(f"[{label}] total={total:.2f}s  ttft={ttft:.2f}s  "
          f"tokens~={n_tokens}  tok/s~={tok_s:.1f}")
    print(f"[{label}] sample output: {''.join(text)[:120]!r}")
    return {"total": total, "ttft": ttft, "tokens": n_tokens, "tok_s": tok_s}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "gpt-oss-120b-uncensored"))
    p.add_argument("--rounds", type=int, default=3, help="warm requests to send after the first (cold) one")
    p.add_argument("--base-url", default=os.environ.get("PROXY_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--api-key", default=os.environ.get("PROXY_API_KEY", "not-needed"))
    args = p.parse_args()

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    print("== cold request (worker may need to start + load model) ==")
    run_once(client, args.model, "cold")

    for i in range(args.rounds):
        print(f"== warm request {i + 1}/{args.rounds} ==")
        run_once(client, args.model, f"warm-{i + 1}")


if __name__ == "__main__":
    main()
