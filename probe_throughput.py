"""Measure OpenRouter throughput for the configured model before a large run.

Sends bursts of real classification requests (canonical prompt, reasoning off)
at several concurrency levels and reports throughput, latency, errors, and
which upstream providers answered. Uses the key/model in config.py. SDK retries
are off so rate limiting shows up as errors instead of being hidden.

    uv run python probe_throughput.py                       # concurrency 8,32,64; 96 requests each
    uv run python probe_throughput.py --concurrency 16,128 --requests 200
    uv run python probe_throughput.py --provider deepseek   # only the given provider(s)
"""

import argparse
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openai import OpenAI

import config
from openrouter_client import REASONING, normalize_opinion
from prompts import SYSTEM_PROMPT, build_user_message

SAMPLE_POSTS = [
    "AI大模型太棒了，写代码效率提高了好几倍，未来可期！",
    "有点担心AI生成的假新闻越来越多",
    "AI有利有弊吧，看怎么用",
    "AI coding assistants are an absolute game changer, my productivity doubled!",
    "AI is going to wipe out millions of jobs, this is dangerous",
    "Not sure yet what LLMs will mean for society",
]


def one_request(client, provider_only):
    extra_body = {"reasoning": REASONING}
    if provider_only:
        extra_body["provider"] = {"only": provider_only, "allow_fallbacks": False}
    text = SAMPLE_POSTS[int(time.time() * 1000) % len(SAMPLE_POSTS)]
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=config.OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(text)},
            ],
            extra_body=extra_body,
        )
        content = response.choices[0].message.content or ""
        valid = normalize_opinion(_opinion(content)) is not None
        provider = getattr(response, "provider", None) or "unknown"
        tokens = response.usage.completion_tokens if response.usage else 0
        return time.time() - start, "ok" if valid else "bad_output", provider, tokens
    except Exception as e:
        status = getattr(e, "status_code", None) or type(e).__name__
        return time.time() - start, f"error {status}", None, 0


def _opinion(content):
    import json

    start, end = content.find("{"), content.rfind("}") + 1
    try:
        return json.loads(content[start:end]).get("opinion") if start != -1 else None
    except (json.JSONDecodeError, AttributeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--concurrency", default="8,32,64")
    parser.add_argument("--requests", type=int, default=96)
    parser.add_argument("--provider", default=None, help="comma-separated provider slugs to restrict to")
    args = parser.parse_args()
    provider_only = args.provider.split(",") if args.provider else None

    client = OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        timeout=config.REQUEST_TIMEOUT,
        max_retries=0,
    )
    print(f"model {config.OPENROUTER_MODEL}, providers {provider_only or 'OpenRouter default routing'}\n")
    for concurrency in [int(c) for c in args.concurrency.split(",")]:
        start = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda _: one_request(client, provider_only), range(args.requests)))
        wall = time.time() - start
        latencies = np.array([r[0] for r in results if r[1] == "ok"])
        outcomes = Counter(r[1] for r in results)
        providers = Counter(r[2] for r in results if r[2])
        out_tokens = np.mean([r[3] for r in results if r[1] == "ok"]) if len(latencies) else float("nan")
        print(
            f"concurrency {concurrency:>4}: {args.requests / wall:6.1f} req/s "
            f"({args.requests} requests in {wall:.1f}s) | "
            + (f"latency p50 {np.percentile(latencies, 50):.2f}s p90 {np.percentile(latencies, 90):.2f}s | "
               if len(latencies) else "")
            + f"avg output {out_tokens:.0f} tokens | {dict(outcomes)}"
        )
        print(f"    providers: {dict(providers.most_common())}")


if __name__ == "__main__":
    main()
