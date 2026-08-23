from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import httpx

from bench.run_benchmark import run_benchmark


@dataclass(frozen=True)
class PhaseVariant:
    name: str
    env: Dict[str, str]


BASE_ENV = {
    "HELIX_USE_TOY_BACKEND": "1",
    "HELIX_DEVICE": "cuda",
}


PHASES: List[PhaseVariant] = [
    PhaseVariant(
        name="Baseline",
        env={
            "HELIX_KV_TOTAL_BLOCKS": "64",
            "HELIX_KV_BLOCK_SIZE": "1024",
            "HELIX_MAX_DECODE_BATCH": "1",
            "HELIX_MAX_BATCHED_TOKENS": "65536",
            "HELIX_PREFILL_CHUNK": "65536",
            "HELIX_PREFIX_MIN_TOKENS": "1000000000",
            "HELIX_ENABLE_CUDA_GRAPH": "0",
            "HELIX_ENABLE_TRITON_RMSNORM": "0",
        },
    ),
    PhaseVariant(
        name="+ Paged KV",
        env={
            "HELIX_KV_TOTAL_BLOCKS": "4096",
            "HELIX_KV_BLOCK_SIZE": "16",
            "HELIX_MAX_DECODE_BATCH": "1",
            "HELIX_MAX_BATCHED_TOKENS": "65536",
            "HELIX_PREFILL_CHUNK": "65536",
            "HELIX_PREFIX_MIN_TOKENS": "1000000000",
            "HELIX_ENABLE_CUDA_GRAPH": "0",
            "HELIX_ENABLE_TRITON_RMSNORM": "0",
        },
    ),
    PhaseVariant(
        name="+ Continuous Batching",
        env={
            "HELIX_KV_TOTAL_BLOCKS": "4096",
            "HELIX_KV_BLOCK_SIZE": "16",
            "HELIX_MAX_DECODE_BATCH": "16",
            "HELIX_MAX_BATCHED_TOKENS": "65536",
            "HELIX_PREFILL_CHUNK": "65536",
            "HELIX_PREFIX_MIN_TOKENS": "1000000000",
            "HELIX_ENABLE_CUDA_GRAPH": "0",
            "HELIX_ENABLE_TRITON_RMSNORM": "0",
        },
    ),
    PhaseVariant(
        name="+ Chunked Prefill",
        env={
            "HELIX_KV_TOTAL_BLOCKS": "4096",
            "HELIX_KV_BLOCK_SIZE": "16",
            "HELIX_MAX_DECODE_BATCH": "16",
            "HELIX_MAX_BATCHED_TOKENS": "1024",
            "HELIX_PREFILL_CHUNK": "128",
            "HELIX_PREFIX_MIN_TOKENS": "1000000000",
            "HELIX_ENABLE_CUDA_GRAPH": "0",
            "HELIX_ENABLE_TRITON_RMSNORM": "0",
        },
    ),
    PhaseVariant(
        name="+ Prefix Cache",
        env={
            "HELIX_KV_TOTAL_BLOCKS": "4096",
            "HELIX_KV_BLOCK_SIZE": "16",
            "HELIX_MAX_DECODE_BATCH": "16",
            "HELIX_MAX_BATCHED_TOKENS": "1024",
            "HELIX_PREFILL_CHUNK": "128",
            "HELIX_PREFIX_MIN_TOKENS": "16",
            "HELIX_ENABLE_CUDA_GRAPH": "0",
            "HELIX_ENABLE_TRITON_RMSNORM": "0",
        },
    ),
    PhaseVariant(
        name="+ CUDA Graph",
        env={
            "HELIX_KV_TOTAL_BLOCKS": "4096",
            "HELIX_KV_BLOCK_SIZE": "16",
            "HELIX_MAX_DECODE_BATCH": "16",
            "HELIX_MAX_BATCHED_TOKENS": "1024",
            "HELIX_PREFILL_CHUNK": "128",
            "HELIX_PREFIX_MIN_TOKENS": "16",
            "HELIX_ENABLE_CUDA_GRAPH": "1",
            "HELIX_ENABLE_TRITON_RMSNORM": "0",
        },
    ),
    PhaseVariant(
        name="+ Triton Kernel",
        env={
            "HELIX_KV_TOTAL_BLOCKS": "4096",
            "HELIX_KV_BLOCK_SIZE": "16",
            "HELIX_MAX_DECODE_BATCH": "16",
            "HELIX_MAX_BATCHED_TOKENS": "1024",
            "HELIX_PREFILL_CHUNK": "128",
            "HELIX_PREFIX_MIN_TOKENS": "16",
            "HELIX_ENABLE_CUDA_GRAPH": "1",
            "HELIX_ENABLE_TRITON_RMSNORM": "1",
        },
    ),
]


def _gcloud_ssh(
    *,
    project: str,
    zone: str,
    instance: str,
    remote_cmd: str,
) -> None:
    cmd = [
        "gcloud",
        "compute",
        "ssh",
        instance,
        "--project",
        project,
        "--zone",
        zone,
        "--command",
        remote_cmd,
    ]
    subprocess.run(cmd, check=True)


def restart_container(
    *,
    project: str,
    zone: str,
    instance: str,
    image: str,
    env: Dict[str, str],
) -> None:
    merged = dict(BASE_ENV)
    merged.update(env)
    env_parts = " ".join(f"-e {k}={v}" for k, v in merged.items())
    remote_cmd = f"""
set -euo pipefail
sudo docker rm -f helixserve >/dev/null 2>&1 || true
sudo docker run --gpus all -d --name helixserve -p 8000:8000 {env_parts} {image}
sudo docker ps --filter name=helixserve --format 'table {{{{.Names}}}}\\t{{{{.Status}}}}'
"""
    _gcloud_ssh(project=project, zone=zone, instance=instance, remote_cmd=remote_cmd)


async def wait_for_health(url: str, timeout_s: int = 180) -> None:
    start = time.time()
    async with httpx.AsyncClient() as client:
        while True:
            try:
                resp = await client.get(f"{url}/healthz", timeout=5.0)
                if resp.status_code == 200:
                    return
            except Exception:
                pass
            if time.time() - start > timeout_s:
                raise TimeoutError(f"timeout waiting for {url}/healthz")
            await asyncio.sleep(1.5)


async def fetch_stats(url: str) -> Dict[str, object]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{url}/stats", timeout=20.0)
        resp.raise_for_status()
        return resp.json()


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


async def run_phase_suite(
    *,
    project: str,
    zone: str,
    instance: str,
    url: str,
    image: str,
    requests: int,
    concurrency: int,
    max_tokens: int,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for phase in PHASES:
        print(f"=== {phase.name} ===")
        restart_container(
            project=project,
            zone=zone,
            instance=instance,
            image=image,
            env=phase.env,
        )
        await wait_for_health(url)

        stats_before = await fetch_stats(url)
        mixed = await run_benchmark(
            base_url=url,
            requests=requests,
            concurrency=concurrency,
            mode="mixed",
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
        )
        rep = await run_benchmark(
            base_url=url,
            requests=200,
            concurrency=16,
            mode="repeated_prefix",
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
        )
        stats_after = await fetch_stats(url)

        b_pref = stats_before.get("prefix_cache", {})
        a_pref = stats_after.get("prefix_cache", {})
        delta_hits = float(a_pref.get("hits", 0)) - float(b_pref.get("hits", 0))
        delta_lookups = float(a_pref.get("lookups", 0)) - float(b_pref.get("lookups", 0))
        prefix_hit_rate = _safe_div(delta_hits, delta_lookups)

        kv = stats_after.get("kv", {})
        backend = stats_after.get("backend", {})
        row = {
            "variant": phase.name,
            "throughput_tokens_s": mixed["throughput_tokens_s"],
            "ttft_p50": mixed["ttft_p50"],
            "ttft_p95": mixed["ttft_p95"],
            "itl_p50": mixed["itl_p50"],
            "itl_p95": mixed["itl_p95"],
            "latency_p95": mixed["latency_p95"],
            "kv_utilization": kv.get("block_utilization", 0.0),
            "fragmentation_tokens": kv.get("internal_waste_tokens", 0),
            "prefix_hit_rate": prefix_hit_rate,
            "repeated_prefix_ttft_p95": rep["ttft_p95"],
            "cuda_graph_decode": backend.get("cuda_graph_decode", False),
            "triton_kernel_enabled": backend.get("triton_kernel_enabled", False),
            "cuda_cpp_extension_loaded": backend.get("cuda_cpp_extension_loaded", False),
            "stats_before": stats_before,
            "stats_after": stats_after,
            "env": phase.env,
        }
        rows.append(row)
        print(
            f"{phase.name}: tok/s={row['throughput_tokens_s']:.1f} "
            f"ttft_p95={row['ttft_p95']:.3f}s itl_p95={row['itl_p95']:.4f}s "
            f"prefix_hit={row['prefix_hit_rate']:.3f}"
        )

    return {"rows": rows}


def render_markdown(rows: List[Dict[str, object]]) -> str:
    lines = []
    lines.append("| Variant | Throughput (tok/s) | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | E2E p95 | KV Util | Fragmentation | Prefix Hit |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {variant} | {throughput:.2f} | {ttft50:.3f} | {ttft95:.3f} | {itl50:.4f} | {itl95:.4f} | {lat95:.3f} | {kvu:.3f} | {frag} | {phr:.3f} |".format(
                variant=row["variant"],
                throughput=row["throughput_tokens_s"],
                ttft50=row["ttft_p50"],
                ttft95=row["ttft_p95"],
                itl50=row["itl_p50"],
                itl95=row["itl_p95"],
                lat95=row["latency_p95"],
                kvu=row["kv_utilization"],
                frag=row["fragmentation_tokens"],
                phr=row["prefix_hit_rate"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phase-by-phase benchmark suite on GCP VM")
    parser.add_argument("--project", default=os.getenv("PROJECT_ID", ""))
    parser.add_argument("--zone", default=os.getenv("ZONE", "us-central1-c"))
    parser.add_argument("--instance", default=os.getenv("INSTANCE_NAME", "helixserve-g2"))
    parser.add_argument("--url", default=os.getenv("HELIX_URL", "http://34.31.35.113:8000"))
    parser.add_argument("--image", default=os.getenv("HELIX_IMAGE", "helixserve:latest"))
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    if not args.project:
        raise SystemExit("--project (or PROJECT_ID env) is required")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(args.output_dir) if args.output_dir else Path("docs/results") / f"phase_table_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(
        run_phase_suite(
            project=args.project,
            zone=args.zone,
            instance=args.instance,
            url=args.url,
            image=args.image,
            requests=args.requests,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
        )
    )

    rows = result["rows"]
    payload = {
        "timestamp_utc": ts,
        "project": args.project,
        "zone": args.zone,
        "instance": args.instance,
        "url": args.url,
        "image": args.image,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "rows": rows,
    }
    (outdir / "phase_table.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (outdir / "phase_table.md").write_text(render_markdown(rows), encoding="utf-8")
    print(json.dumps({"output_dir": str(outdir)}, indent=2))


if __name__ == "__main__":
    main()
