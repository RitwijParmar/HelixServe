#!/usr/bin/env python3
"""Generate demo screenshots and video artifacts for HelixServe."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import textwrap
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "docs" / "assets" / "demo"
SCREENSHOT_DIR = DEMO_DIR / "screenshots"
RESULTS_DIR = ROOT / "docs" / "results"


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/Library/Fonts/Courier New.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _json_request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    req = urllib.request.Request(url=url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _safe_get_json(url: str, payload: dict[str, Any] | None = None) -> Tuple[bool, dict[str, Any] | str]:
    try:
        return True, _json_request(url, payload)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _latest_suite_path() -> Path | None:
    candidates = sorted(RESULTS_DIR.glob("20*/suite.json"))
    return candidates[-1] if candidates else None


def _latest_phase_table_path() -> Path | None:
    candidates = sorted(RESULTS_DIR.glob("phase_table_*/phase_table.md"))
    return candidates[-1] if candidates else None


def _summarize_suite(suite: dict[str, Any]) -> list[str]:
    reports = suite.get("reports", {})
    lines = [
        "$ python -m bench.run_live_suite --url <endpoint> --stream",
        "",
        "Workload summary:",
        "workload            tok/s      ttft_p95(s)  itl_p95(s)  e2e_p95(s)",
        "-" * 66,
    ]
    ordered_names = ["short", "long", "mixed", "repeated_prefix", "burst"]
    for name in ordered_names:
        row = reports.get(name)
        if not row:
            continue
        lines.append(
            f"{name:<18} {row['throughput_tokens_s']:>8.2f}   "
            f"{row['ttft_p95']:>10.3f}   {row['itl_p95']:>9.4f}   {row['latency_p95']:>9.3f}"
        )
    return lines


def _render_terminal(
    title: str,
    subtitle: str,
    lines: Iterable[str],
    out_path: Path,
    width: int = 1800,
    height: int = 1080,
    line_height: int = 34,
) -> None:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    lines_list = list(lines)

    title_font = _load_font(36)
    subtitle_font = _load_font(22)
    mono_font = _load_font(24)

    wrapped_lines: List[str] = []
    max_chars = 120
    for line in lines_list:
        if len(line) <= max_chars:
            wrapped_lines.append(line)
        else:
            wrapped_lines.extend(textwrap.wrap(line, width=max_chars))

    body_height = height
    img = Image.new("RGB", (width, body_height), color=(14, 18, 28))
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle([(0, 0), (width, 100)], fill=(26, 32, 48))
    draw.text((28, 18), title, font=title_font, fill=(232, 241, 255))
    draw.text((30, 66), subtitle, font=subtitle_font, fill=(165, 186, 220))

    # Terminal frame
    pad = 28
    frame_top = 120
    frame_bottom = body_height - 28
    draw.rounded_rectangle(
        [(pad, frame_top), (width - pad, frame_bottom)],
        radius=18,
        fill=(10, 13, 20),
        outline=(64, 86, 120),
        width=2,
    )

    # window controls
    draw.ellipse((pad + 20, frame_top + 16, pad + 36, frame_top + 32), fill=(255, 95, 86))
    draw.ellipse((pad + 44, frame_top + 16, pad + 60, frame_top + 32), fill=(255, 189, 46))
    draw.ellipse((pad + 68, frame_top + 16, pad + 84, frame_top + 32), fill=(39, 201, 63))

    # content
    y = frame_top + 56
    max_lines = max(1, int((frame_bottom - (frame_top + 70)) / line_height))
    for line in wrapped_lines[:max_lines]:
        color = (210, 226, 255)
        if line.startswith("$"):
            color = (143, 255, 179)
        elif line.startswith("ERROR") or "Traceback" in line:
            color = (255, 146, 146)
        draw.text((pad + 24, y), line, font=mono_font, fill=color)
        y += line_height

    img.save(out_path)


def _phase_table_lines(phase_path: Path) -> list[str]:
    raw = phase_path.read_text(encoding="utf-8").strip().splitlines()
    lines = ["$ cat docs/results/<phase_table>/phase_table.md", ""]
    lines.extend(raw[:10])
    return lines


def _completion_lines(result: dict[str, Any] | str) -> list[str]:
    lines = [
        "$ curl -s <endpoint>/v1/completions -H 'content-type: application/json' "
        "-d '{\"prompt\":\"Explain paged KV cache\", \"max_tokens\":48}'",
        "",
    ]
    if isinstance(result, str):
        lines.append(f"ERROR: {result}")
        return lines
    choices = result.get("choices", [])
    if not choices:
        lines.append(json.dumps(result, indent=2))
        return lines
    text = choices[0].get("text", "").strip()
    safe_text = text.encode("unicode_escape", "backslashreplace").decode("ascii", "ignore")
    lines.append("Model output:")
    lines.extend(textwrap.wrap(safe_text or "<empty>", width=112))
    usage = result.get("usage")
    if usage:
        lines.append("")
        lines.append(f"usage: {json.dumps(usage)}")
    return lines


def _stats_lines(stats: dict[str, Any] | str) -> list[str]:
    lines = ["$ curl -s <endpoint>/stats", ""]
    if isinstance(stats, str):
        lines.append(f"ERROR: {stats}")
        return lines
    summary = {
        "config": {
            "kv_block_size": stats.get("config", {}).get("kv_block_size"),
            "kv_total_blocks": stats.get("config", {}).get("kv_total_blocks"),
            "max_decode_batch_size": stats.get("config", {}).get("max_decode_batch_size"),
            "prefill_chunk_size": stats.get("config", {}).get("prefill_chunk_size"),
            "max_num_batched_tokens": stats.get("config", {}).get("max_num_batched_tokens"),
            "enable_cuda_graph_decode": stats.get("config", {}).get("enable_cuda_graph_decode"),
            "enable_triton_rmsnorm": stats.get("config", {}).get("enable_triton_rmsnorm"),
        },
        "backend": stats.get("backend", {}),
        "prefix_cache": stats.get("prefix_cache", {}),
        "kv": {
            "live_blocks": stats.get("kv", {}).get("live_blocks"),
            "used_tokens": stats.get("kv", {}).get("used_tokens"),
            "block_utilization": stats.get("kv", {}).get("block_utilization"),
            "memory_pressure": stats.get("kv", {}).get("memory_pressure"),
        },
    }
    lines.extend(json.dumps(summary, indent=2).splitlines())
    return lines


def _healthz_lines(url: str, health: dict[str, Any] | str) -> list[str]:
    lines = [
        f"$ curl -s {url}/healthz",
        "",
    ]
    if isinstance(health, str):
        lines.append(f"ERROR: {health}")
    else:
        lines.append(json.dumps(health))
    return lines


def _pytest_lines() -> list[str]:
    cmd = [str(ROOT / ".venv" / "bin" / "pytest"), "-q"]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
        output = (proc.stdout + "\n" + proc.stderr).strip().splitlines()
        output = [line for line in output if line.strip()]
        lines = ["$ .venv/bin/pytest -q", ""]
        lines.extend(output[:35])
        return lines
    except OSError as exc:
        return ["$ .venv/bin/pytest -q", "", f"ERROR: {exc}"]


def _build_video(images: list[Path], gif_path: Path, mp4_path: Path) -> None:
    frames = [imageio.imread(img) for img in images]
    durations = [2.5, 3.0, 4.0, 5.0, 6.0, 2.5]
    if len(durations) < len(frames):
        durations.extend([3.0] * (len(frames) - len(durations)))
    durations = durations[: len(frames)]

    imageio.mimsave(gif_path, frames, duration=durations, loop=0)

    fps = 30
    with imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame, duration in zip(frames, durations):
            repeat = max(1, int(duration * fps))
            for _ in range(repeat):
                writer.append_data(np.asarray(frame))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo screenshots and video artifacts.")
    parser.add_argument("--url", default="http://34.136.218.176:8000", help="HelixServe endpoint URL")
    args = parser.parse_args()

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    endpoint = args.url.rstrip("/")
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    ok_health, health = _safe_get_json(f"{endpoint}/healthz")
    ok_stats, stats = _safe_get_json(f"{endpoint}/stats")
    ok_completion, completion = _safe_get_json(
        f"{endpoint}/v1/completions",
        {
            "prompt": "Explain why paged KV-cache and continuous batching improve LLM serving.",
            "max_tokens": 64,
            "temperature": 0.0,
        },
    )

    suite_path = _latest_suite_path()
    phase_path = _latest_phase_table_path()

    suite_lines = ["No suite.json found in docs/results."]
    if suite_path:
        loaded = json.loads(suite_path.read_text(encoding="utf-8"))
        suite = loaded.get("suite", loaded)
        suite_lines = _summarize_suite(suite)
        suite_lines.append("")
        suite_lines.append(f"artifact: {suite_path.relative_to(ROOT)}")

    phase_lines = ["No phase table markdown found in docs/results."]
    if phase_path:
        phase_lines = _phase_table_lines(phase_path)
        phase_lines.append("")
        phase_lines.append(f"artifact: {phase_path.relative_to(ROOT)}")

    files = [
        ("01_healthz.png", "Live Endpoint Health Check", f"{endpoint} • {now}", _healthz_lines(endpoint, health if ok_health else str(health))),
        ("02_runtime_stats.png", "Runtime Feature Verification", "paged KV + batching + CUDA Graph + Triton + CUDA C++", _stats_lines(stats if ok_stats else str(stats))),
        ("03_completion.png", "Live Completion Request", "decoder output from running endpoint", _completion_lines(completion if ok_completion else str(completion))),
        ("04_live_suite.png", "Benchmark Suite Summary", "fresh workload run results", suite_lines),
        ("05_phase_table.png", "Baseline -> Optimization Table", "phase-by-phase measured deltas", phase_lines),
        ("06_tests.png", "Local Test Suite", "unit + server smoke tests", _pytest_lines()),
    ]

    image_paths: list[Path] = []
    for filename, title, subtitle, lines in files:
        out = SCREENSHOT_DIR / filename
        _render_terminal(title=title, subtitle=subtitle, lines=lines, out_path=out)
        image_paths.append(out)

    gif_path = DEMO_DIR / "helixserve_demo.gif"
    mp4_path = DEMO_DIR / "helixserve_demo.mp4"
    _build_video(image_paths, gif_path=gif_path, mp4_path=mp4_path)

    manifest = {
        "generated_at_utc": now,
        "endpoint": endpoint,
        "screenshots": [str(path.relative_to(ROOT)) for path in image_paths],
        "gif": str(gif_path.relative_to(ROOT)),
        "mp4": str(mp4_path.relative_to(ROOT)),
        "suite_source": str(suite_path.relative_to(ROOT)) if suite_path else None,
        "phase_table_source": str(phase_path.relative_to(ROOT)) if phase_path else None,
    }
    (DEMO_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
