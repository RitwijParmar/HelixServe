from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

SHARED_PREFIX = """You are an execution-risk assistant at an electronic market maker. Analyze the order-flow snapshot, distinguish signal from noise, and return a concise risk action with rationale. Never invent missing fields.\n\n"""


@dataclass
class Result:
    request_id: int
    ok: bool
    status: int
    ttft_ms: float | None
    e2e_ms: float
    itl_ms: list[float]
    output_tokens: int
    error: str | None = None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def prompt(index: int, repeated_prefix: bool) -> str:
    book = (
        f"symbol=XYZ venue=NASDAQ seq={80_000 + index}; "
        f"bid={100 + index % 7 / 100:.2f}x{900 + index % 13 * 40}; "
        f"ask={100.02 + index % 7 / 100:.2f}x{700 + index % 11 * 50}; "
        f"last_100ms_trades={10 + index % 19}; cancel_ratio={0.20 + index % 17 / 100:.2f}."
    )
    return (SHARED_PREFIX if repeated_prefix else "") + book


async def one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    request_id: int,
    max_tokens: int,
    repeated_prefix: bool,
) -> Result:
    body = {
        "model": model,
        "prompt": prompt(request_id, repeated_prefix),
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    first: float | None = None
    previous: float | None = None
    itl: list[float] = []
    output_tokens = 0
    try:
        async with client.stream("POST", f"{base_url}/v1/completions", json=body) as response:
            status = response.status_code
            if status != 200:
                message = (await response.aread()).decode(errors="replace")[:300]
                return Result(request_id, False, status, None, (time.perf_counter() - started) * 1000, [], 0, message)
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    event: dict[str, Any] = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                usage = event.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    output_tokens = int(usage["completion_tokens"])
                choices = event.get("choices") or []
                text = choices[0].get("text", "") if choices else ""
                if not text:
                    continue
                now = time.perf_counter()
                if first is None:
                    first = now
                elif previous is not None:
                    itl.append((now - previous) * 1000)
                previous = now
                if not usage:
                    output_tokens += 1
        ended = time.perf_counter()
        return Result(request_id, True, 200, (first - started) * 1000 if first else None, (ended - started) * 1000, itl, output_tokens)
    except Exception as exc:  # benchmark records transport errors instead of aborting a run
        return Result(request_id, False, 0, None, (time.perf_counter() - started) * 1000, [], 0, str(exc)[:300])


async def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=args.concurrency)) as client:
        async def guarded(index: int) -> Result:
            if args.request_rate > 0:
                await asyncio.sleep(random.expovariate(args.request_rate))
            async with semaphore:
                return await one_request(client, args.url.rstrip("/"), args.model, index, args.max_tokens, args.repeated_prefix)

        results = await asyncio.gather(*(guarded(i) for i in range(args.requests)))
    duration = time.perf_counter() - started
    successful = [item for item in results if item.ok]
    ttft = [item.ttft_ms for item in successful if item.ttft_ms is not None]
    e2e = [item.e2e_ms for item in successful]
    itl = [value for item in successful for value in item.itl_ms]
    output_tokens = sum(item.output_tokens for item in successful)
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "configuration": {key: value for key, value in vars(args).items() if key != "output"},
        "summary": {
            "requests": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "duration_s": duration,
            "request_throughput_rps": len(successful) / duration,
            "output_throughput_tok_s": output_tokens / duration,
            "ttft_ms": {"p50": percentile(ttft, 0.50), "p95": percentile(ttft, 0.95), "p99": percentile(ttft, 0.99)},
            "itl_ms": {"p50": percentile(itl, 0.50), "p95": percentile(itl, 0.95), "p99": percentile(itl, 0.99)},
            "e2e_ms": {"p50": percentile(e2e, 0.50), "p95": percentile(e2e, 0.95), "p99": percentile(e2e, 0.99)},
            "mean_e2e_ms": statistics.fmean(e2e) if e2e else None,
        },
        "requests": [asdict(item) for item in results],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Measure true streaming TTFT, ITL, and E2E latency")
    result.add_argument("--url", default="http://127.0.0.1:8000")
    result.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    result.add_argument("--requests", type=int, default=100)
    result.add_argument("--concurrency", type=int, default=16)
    result.add_argument("--request-rate", type=float, default=0, help="Poisson delay rate; 0 sends immediately")
    result.add_argument("--max-tokens", type=int, default=64)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--timeout", type=float, default=180)
    result.add_argument("--repeated-prefix", action="store_true")
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
