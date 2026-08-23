#!/usr/bin/env python3
"""Build the narrated TickYantra product and engineering demo."""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import edge_tts
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets" / "demo" / "tickyantra"
SCREEN_DIR = ASSET_DIR / "screens"
FINAL_DIR = ASSET_DIR / "final"

WIDTH = 1920
HEIGHT = 1080
FPS = 24
VOICE = "en-IN-PrabhatNeural"

INK = (235, 242, 237)
MUTED = (145, 158, 150)
ACID = (183, 243, 74)
DANGER = (255, 101, 79)
VOID = (7, 10, 9)
PANEL = (13, 18, 16)
LINE = (38, 48, 43)

VOICEOVER = (
    "Meet TickYantra. Tick is the atomic event in an electronic market. Yantra means a precise "
    "machine. Together, the name describes an inference control system built for workloads where "
    "milliseconds, queue pressure, and reproducibility matter. "
    "This is not another chatbot wrapper, and it does not contain a toy decoder. SGLang owns the "
    "real CUDA data plane: tokenization, continuous batching, paged key value memory, RadixAttention, "
    "and model execution. TickYantra sits in front as an inspectable S L O control plane. "
    "The request path is simple. Open A I traffic enters the gateway. Admission is bounded by the "
    "current concurrency limit. Prefix affinity can prioritize repeated system context, while a queue "
    "deadline prevents affinity from turning into starvation. Prometheus metrics expose queue wait, "
    "time to first token, end to end latency, active work, and controller state. "
    "This is the interactive pressure chamber. Arrival rate, shared-prefix ratio, and the T T F T "
    "target can be changed while the chart visualizes the control law. Yahan rule simple hai. When "
    "rolling p ninety five T T F T exceeds one point one times the target, the limit falls. When "
    "latency is safely below the target and work is waiting, the limit may rise. This screen is a "
    "behavioral visualization, not a fabricated GPU benchmark. "
    "The real experiment ran Qwen two point five, seven B, in B F sixteen on one NVIDIA L four in "
    "Google Cloud. One hundred ordered shared-prefix requests used concurrency sixteen, fixed prompt "
    "and output lengths, seed seventeen, and eight warm-ups. The official SGLang native runner "
    "completed every request at three point seven requests per second, with p ninety five T T F T of "
    "one hundred thirty eight milliseconds and a ninety nine point five percent prompt cache hit rate. "
    "The gateway result was intentionally not polished into a false victory. Static and adaptive "
    "windows reduced throughput to about two point nine requests per second and pushed p ninety five "
    "T T F T near four seconds under the saturated burst. That negative result exposed two defects: "
    "queue time was missing from adaptive feedback, and an overly long fingerprint fragmented shared "
    "prefixes. Both were fixed and covered by regression tests. Native SGLang remains the deployment "
    "choice for this workload until a controller clears the native performance gate. "
    "The GCP deployment pins the measured SGLang image digest, binds inference ports to loopback, "
    "checks GPU readiness, and arms a two-hour automatic shutdown guard. Raw request rows, the invalid "
    "runner attempt, the valid native run, methodology, tests, and the decision report are committed. "
    "That is the engineering position behind TickYantra: control the queue, trust the engine, measure "
    "the tail, and never hide a result that disproves the first design."
)

SCENES: list[dict[str, Any]] = [
    {
        "id": "identity",
        "kind": "card",
        "duration": 11.0,
        "title": "TICKYANTRA",
        "eyebrow": "LOW-LATENCY INFERENCE SYSTEMS",
        "subtitle": "Tick for the market event. Yantra for the precise machine.",
        "lines": [
            "SLO-aware control plane for real SGLang inference",
            "NVIDIA L4  •  Qwen2.5-7B  •  raw latency evidence",
            "Built by Ritwij Parmar",
        ],
    },
    {
        "id": "hero",
        "kind": "image",
        "duration": 12.0,
        "image": "01_hero.png",
        "title": "NOT A CHATBOT WRAPPER",
        "subtitle": "A queue-control and measurement system around a real CUDA data plane.",
    },
    {
        "id": "request_path",
        "kind": "card",
        "duration": 15.0,
        "title": "REQUEST PATH",
        "eyebrow": "OPENAI-COMPATIBLE STREAMING",
        "subtitle": "Bounded admission, fair affinity, observable release.",
        "lines": [
            "CLIENT  →  TICKYANTRA GATE  →  SGLANG  →  NVIDIA L4",
            "Prefix affinity yields to age-based fairness near deadline",
            "Metrics: queue wait • TTFT • ITL • E2E • controller state",
        ],
    },
    {
        "id": "control_lab",
        "kind": "image",
        "duration": 17.0,
        "image": "02_control_lab_nominal.png",
        "title": "PRESSURE CHAMBER",
        "subtitle": "Change arrival pressure, shared context, and the client TTFT target.",
    },
    {
        "id": "control_law",
        "kind": "card",
        "duration": 14.0,
        "title": "CONTROL LAW",
        "eyebrow": "CLIENT-OBSERVED TAIL LATENCY",
        "subtitle": "Queue time is part of the SLO signal.",
        "lines": [
            "p95 TTFT > 1.10 × target   →   concurrency limit − 1",
            "p95 TTFT < 0.75 × target + queued work   →   limit + 1",
            "Fairness deadline overrides prefix affinity",
        ],
    },
    {
        "id": "benchmark",
        "kind": "benchmark",
        "duration": 19.0,
        "title": "MEASURED ON GCP L4",
        "subtitle": "100 requests • concurrency 16 • shared-prefix workload • zero valid-run failures",
    },
    {
        "id": "negative_result",
        "kind": "card",
        "duration": 17.0,
        "title": "THE NEGATIVE RESULT MATTERS",
        "eyebrow": "MEASURE → DISPROVE → FIX",
        "subtitle": "Native SGLang remains the deployment choice for this burst.",
        "lines": [
            "Gateway windowing created a second generation-wave queue",
            "Adaptive feedback excluded queue wait and raised 12 → 13",
            "1,024-char keys fragmented one shared context into 100 keys",
            "Fix: arrival-clock TTFT + 128-char prefix fingerprint",
        ],
    },
    {
        "id": "principles",
        "kind": "image",
        "duration": 12.0,
        "image": "04_system.png",
        "title": "ENGINEERING POSITION",
        "subtitle": "No toy path. Fair affinity. Raw evidence.",
    },
    {
        "id": "deployment",
        "kind": "card",
        "duration": 14.0,
        "title": "DEPLOYMENT POSTURE",
        "eyebrow": "GCP G2 + NVIDIA L4",
        "subtitle": "Reproducible, private by default, and bounded by cost controls.",
        "lines": [
            "Pinned SGLang v0.5.16 image digest",
            "Qwen2.5-7B BF16 • RadixCache • CUDA graphs",
            "Inference ports bound to loopback",
            "Health checks + Prometheus + two-hour shutdown guard",
            "Live verified: /readyz → SGLang + 32-token Qwen completion",
        ],
    },
    {
        "id": "outro",
        "kind": "card",
        "duration": 11.0,
        "title": "CONTROL THE QUEUE.",
        "eyebrow": "TRUST THE ENGINE.",
        "subtitle": "Measure the tail. Preserve the evidence. Fix what the data disproves.",
        "lines": [
            "github.com/RitwijParmar/TickYantra",
            "Real SGLang • Real L4 • Honest systems work",
        ],
    },
]


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/SFNSMono.ttf" if not bold else "/System/Library/Fonts/SFNSMonoBold.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, face: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if not current or draw.textlength(candidate, font=face) <= width:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def grid_background() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), VOID)
    draw = ImageDraw.Draw(image)
    for x in range(0, WIDTH, 48):
        draw.line((x, 0, x, HEIGHT), fill=(15, 22, 18), width=1)
    for y in range(0, HEIGHT, 48):
        draw.line((0, y, WIDTH, y), fill=(15, 22, 18), width=1)
    draw.rectangle((72, 68, WIDTH - 72, HEIGHT - 68), outline=LINE, width=2)
    return image


def draw_card(scene: dict[str, Any]) -> Image.Image:
    image = grid_background()
    draw = ImageDraw.Draw(image)
    eyebrow_face = font(25, bold=True)
    title_face = font(80, bold=True)
    subtitle_face = font(34)
    line_face = font(31)

    draw.rectangle((98, 112, 118, 132), fill=ACID)
    draw.text((136, 106), scene.get("eyebrow", "TICKYANTRA / SYSTEMS DEMO"), font=eyebrow_face, fill=ACID)
    draw.text((100, 188), scene["title"], font=title_face, fill=INK)

    y = 330
    for subtitle_line in wrap(draw, scene["subtitle"], subtitle_face, WIDTH - 220):
        draw.text((102, y), subtitle_line, font=subtitle_face, fill=MUTED)
        y += 46

    y += 60
    for index, line in enumerate(scene.get("lines", []), start=1):
        draw.text((112, y + 6), f"{index:02d}", font=font(20, bold=True), fill=ACID)
        draw.line((168, y + 23, 218, y + 23), fill=LINE, width=2)
        for wrapped in wrap(draw, line, line_face, WIDTH - 360):
            draw.text((250, y), wrapped, font=line_face, fill=INK)
            y += 42
        y += 30

    draw.text((100, HEIGHT - 108), "TICKYANTRA  /  LOW-LATENCY INFERENCE SYSTEMS", font=font(20), fill=(91, 104, 96))
    draw.text((WIDTH - 390, HEIGHT - 108), "RITWIJ PARMAR", font=font(20, bold=True), fill=ACID)
    return image


def draw_benchmark(scene: dict[str, Any]) -> Image.Image:
    image = grid_background()
    draw = ImageDraw.Draw(image)
    draw.text((100, 92), scene["title"], font=font(66, bold=True), fill=INK)
    draw.text((102, 180), scene["subtitle"], font=font(25), fill=MUTED)

    columns = [100, 720, 1010, 1310, 1590]
    headers = ["PATH", "REQ/S", "P95 TTFT", "P95 E2E", "DECISION"]
    for x, header in zip(columns, headers, strict=True):
        draw.text((x, 290), header, font=font(20, bold=True), fill=ACID)
    draw.line((100, 335, WIDTH - 100, 335), fill=LINE, width=2)

    rows = [
        ("Official SGLang native", "3.70", "137.93 ms", "3,874.09 ms", "SHIP", ACID),
        ("TickYantra native proxy", "3.55", "472.78 ms", "4,177.12 ms", "CROSS-CHECK", INK),
        ("Static window, limit 12", "2.90", "3,939.72 ms", "7,631.61 ms", "DO NOT SHIP", DANGER),
        ("Adaptive, initial 12", "2.90", "3,960.67 ms", "7,672.68 ms", "FIX + RERUN", DANGER),
    ]
    y = 390
    for name, rps, ttft, e2e, decision, color in rows:
        draw.rectangle((96, y - 18, WIDTH - 96, y + 86), fill=PANEL, outline=LINE, width=1)
        values = [name, rps, ttft, e2e, decision]
        for x, value in zip(columns, values, strict=True):
            draw.text((x, y + 14), value, font=font(24, bold=x != columns[0]), fill=color if x == columns[-1] else INK)
        y += 126

    draw.rectangle((100, 865, WIDTH - 100, 930), fill=(18, 27, 21), outline=ACID, width=2)
    draw.text((130, 884), "99.5% prompt-token cache hit  •  native wins this saturated burst  •  raw artifacts committed", font=font(25, bold=True), fill=ACID)
    return image


def scene_image(scene: dict[str, Any]) -> Image.Image:
    if scene["kind"] == "card":
        return draw_card(scene)
    if scene["kind"] == "benchmark":
        return draw_benchmark(scene)
    source = Image.open(SCREEN_DIR / scene["image"]).convert("RGB")
    return ImageOps.fit(source, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS)


def add_overlay(image: Image.Image, scene: dict[str, Any], scene_number: int) -> Image.Image:
    frame = image.convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, WIDTH, 105), fill=(5, 8, 7, 218))
    draw.rectangle((0, HEIGHT - 142, WIDTH, HEIGHT), fill=(5, 8, 7, 220))
    draw.rectangle((0, 0, 14, HEIGHT), fill=(*ACID, 255))
    draw.text((52, 30), f"{scene_number:02d}  {scene['title']}", font=font(32, bold=True), fill=INK)
    draw.text((WIDTH - 310, 36), "TICKYANTRA", font=font(23, bold=True), fill=ACID)
    subtitle_face = font(29)
    y = HEIGHT - 112
    for line in wrap(draw, scene["subtitle"], subtitle_face, WIDTH - 180)[:2]:
        draw.text((58, y), line, font=subtitle_face, fill=INK)
        y += 38
    return Image.alpha_composite(frame, overlay).convert("RGB")


def zoom_frame(base: Image.Image, progress: float, zoom_amount: float) -> Image.Image:
    zoom = 1 + zoom_amount * progress
    crop_width = int(WIDTH / zoom)
    crop_height = int(HEIGHT / zoom)
    left = (WIDTH - crop_width) // 2
    top = (HEIGHT - crop_height) // 2
    return base.crop((left, top, left + crop_width, top + crop_height)).resize(
        (WIDTH, HEIGHT), Image.Resampling.BICUBIC
    )


async def synthesize(path: Path) -> None:
    speech = edge_tts.Communicate(VOICEOVER, VOICE, rate="+18%", pitch="-6Hz")
    await speech.save(str(path))


def media_duration(path: Path) -> float:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True, check=False)
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", result.stderr)
    if not match:
        raise RuntimeError(f"Unable to read duration for {path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def build() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    voice_path = FINAL_DIR / "tickyantra_voiceover_en_in_prabhat.mp3"
    script_path = FINAL_DIR / "voiceover_script.txt"
    timeline_path = FINAL_DIR / "timeline.json"
    silent_path = FINAL_DIR / "tickyantra_demo_silent.mp4"
    final_path = FINAL_DIR / "tickyantra_end_to_end_demo.mp4"
    poster_path = FINAL_DIR / "tickyantra_demo_poster.png"

    script_path.write_text(VOICEOVER + "\n", encoding="utf-8")
    asyncio.run(synthesize(voice_path))
    audio_duration = media_duration(voice_path)
    base_duration = sum(float(scene["duration"]) for scene in SCENES)
    scale = max(1.0, (audio_duration + 4.0) / base_duration)

    timeline: dict[str, Any] = {"fps": FPS, "voice": VOICE, "scenes": []}
    cursor = 0.0
    bases: list[Image.Image] = []
    for number, scene in enumerate(SCENES, start=1):
        base = add_overlay(scene_image(scene), scene, number)
        bases.append(base)
        duration = float(scene["duration"]) * scale
        timeline["scenes"].append(
            {
                "id": scene["id"],
                "start_s": round(cursor, 3),
                "end_s": round(cursor + duration, 3),
                "title": scene["title"],
                "subtitle": scene["subtitle"],
            }
        )
        cursor += duration
    timeline["duration_s"] = round(cursor, 3)
    timeline["voiceover_duration_s"] = round(audio_duration, 3)
    timeline_path.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")
    bases[0].save(poster_path)

    with imageio.get_writer(
        silent_path,
        fps=FPS,
        codec="libx264",
        quality=7,
        macro_block_size=1,
        ffmpeg_log_level="warning",
    ) as writer:
        for scene, base in zip(SCENES, bases, strict=True):
            duration = float(scene["duration"]) * scale
            frame_count = max(1, round(duration * FPS))
            zoom_amount = 0.035 if scene["kind"] == "image" else 0.015
            for frame_index in range(frame_count):
                progress = frame_index / max(1, frame_count - 1)
                frame = zoom_frame(base, progress, zoom_amount)
                array = np.asarray(frame, dtype=np.uint8)
                fade_frames = min(10, frame_count // 5)
                if frame_index < fade_frames:
                    factor = 0.25 + 0.75 * frame_index / max(1, fade_frames)
                    array = np.clip(array.astype(np.float32) * factor, 0, 255).astype(np.uint8)
                writer.append_data(array)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(silent_path),
            "-i",
            str(voice_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(final_path),
        ],
        check=True,
    )
    silent_path.unlink()

    manifest = {
        "video": str(final_path.relative_to(ROOT)),
        "poster": str(poster_path.relative_to(ROOT)),
        "voiceover": str(voice_path.relative_to(ROOT)),
        "script": str(script_path.relative_to(ROOT)),
        "timeline": str(timeline_path.relative_to(ROOT)),
        "source_screens": sorted(str(path.relative_to(ROOT)) for path in SCREEN_DIR.glob("*.png")),
    }
    (FINAL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**manifest, "duration_s": round(media_duration(final_path), 2)}, indent=2))


if __name__ == "__main__":
    build()
