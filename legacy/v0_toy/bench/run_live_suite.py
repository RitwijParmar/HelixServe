from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import httpx

from bench.run_benchmark import run_benchmark


@dataclass(frozen=True)
class Workload:
    name: str
    mode: str
    requests: int
    concurrency: int


DEFAULT_WORKLOADS: List[Workload] = [
    Workload(name="short", mode="short", requests=200, concurrency=16),
    Workload(name="long", mode="long", requests=200, concurrency=16),
    Workload(name="mixed", mode="mixed", requests=200, concurrency=16),
    Workload(name="repeated_prefix", mode="repeated_prefix", requests=200, concurrency=16),
    Workload(name="burst", mode="mixed", requests=400, concurrency=64),
]


async def fetch_stats(base_url: str) -> Dict[str, object]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{base_url}/stats", timeout=30.0)
        resp.raise_for_status()
        return resp.json()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


async def run_suite(args: argparse.Namespace) -> Dict[str, object]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(args.output_dir) if args.output_dir else Path("docs/results") / ts
    outdir.mkdir(parents=True, exist_ok=True)

    stats_before = await fetch_stats(args.url)
    write_json(outdir / "stats_before.json", stats_before)

    reports: Dict[str, Dict[str, float]] = {}
    for workload in DEFAULT_WORKLOADS:
        report = await run_benchmark(
            base_url=args.url,
            requests=workload.requests,
            concurrency=workload.concurrency,
            mode=workload.mode,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=args.stream,
        )
        reports[workload.name] = report
        write_json(outdir / f"{workload.name}.json", report)
        print(
            f"{workload.name}: p95={report['latency_p95']:.3f}s "
            f"ttft_p95={report['ttft_p95']:.3f}s tok/s={report['throughput_tokens_s']:.1f}"
        )

    stats_after = await fetch_stats(args.url)
    write_json(outdir / "stats_after.json", stats_after)

    suite = {
        "timestamp_utc": ts,
        "url": args.url,
        "stream": args.stream,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "workloads": [asdict(w) for w in DEFAULT_WORKLOADS],
        "reports": reports,
        "stats_before": stats_before,
        "stats_after": stats_after,
    }
    write_json(outdir / "suite.json", suite)
    return {"output_dir": str(outdir), "suite": suite}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HelixServe live benchmark suite")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    result = asyncio.run(run_suite(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
