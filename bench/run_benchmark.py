from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Dict, List

import httpx

from bench.workload import generate_prompts


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int((len(values) - 1) * pct)
    return values[idx]


async def _run_single(
    client: httpx.AsyncClient,
    *,
    url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> Dict[str, float]:
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    started = time.perf_counter()
    ttft = None
    completion_tokens = 0

    if stream:
        async with client.stream("POST", url, json=payload, timeout=None) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                item = line[6:]
                if item == "[DONE]":
                    break
                if ttft is None:
                    ttft = time.perf_counter() - started
                obj = json.loads(item)
                choices = obj.get("choices") or []
                if choices:
                    text = choices[0].get("text") or ""
                    if text:
                        completion_tokens += 1
    else:
        resp = await client.post(url, json=payload, timeout=None)
        resp.raise_for_status()
        obj = resp.json()
        ttft = time.perf_counter() - started
        completion_tokens = int(obj.get("usage", {}).get("completion_tokens", 0))

    latency = time.perf_counter() - started
    return {
        "latency": latency,
        "ttft": ttft or latency,
        "completion_tokens": float(completion_tokens),
    }


async def run_benchmark(
    *,
    base_url: str,
    requests: int,
    concurrency: int,
    mode: str,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> Dict[str, float]:
    prompts = generate_prompts(mode=mode, count=requests)
    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, float]] = []

    async with httpx.AsyncClient() as client:
        async def worker(prompt: str) -> None:
            async with sem:
                result = await _run_single(
                    client,
                    url=f"{base_url}/v1/completions",
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                )
                results.append(result)

        start = time.perf_counter()
        await asyncio.gather(*(worker(prompt) for prompt in prompts))
        elapsed = time.perf_counter() - start

    latencies = [r["latency"] for r in results]
    ttfts = [r["ttft"] for r in results]
    total_tokens = sum(r["completion_tokens"] for r in results)

    return {
        "requests": float(requests),
        "concurrency": float(concurrency),
        "elapsed_s": elapsed,
        "throughput_rps": requests / elapsed if elapsed > 0 else 0.0,
        "throughput_tokens_s": total_tokens / elapsed if elapsed > 0 else 0.0,
        "latency_p50": percentile(latencies, 0.50),
        "latency_p95": percentile(latencies, 0.95),
        "latency_p99": percentile(latencies, 0.99),
        "ttft_p50": percentile(ttfts, 0.50),
        "ttft_p95": percentile(ttfts, 0.95),
        "ttft_p99": percentile(ttfts, 0.99),
        "avg_completion_tokens": statistics.mean([r["completion_tokens"] for r in results])
        if results
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HelixServe benchmark runner")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--mode", choices=["short", "long", "mixed", "repeated_prefix"], default="mixed")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    report = asyncio.run(
        run_benchmark(
            base_url=args.url,
            requests=args.requests,
            concurrency=args.concurrency,
            mode=args.mode,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=args.stream,
        )
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
