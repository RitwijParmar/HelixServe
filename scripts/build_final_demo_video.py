#!/usr/bin/env python3
"""Build a LinkedIn-ready final HelixServe demo video with voiceover."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import edge_tts
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "docs" / "assets" / "demo"
SCREENSHOT_DIR = DEMO_DIR / "screenshots"
FINAL_DIR = DEMO_DIR / "final"

WIDTH = 1920
HEIGHT = 1080
FPS = 30

VOICE = "en-IN-NeerjaNeural"


VOICEOVER_SCRIPT = (
    "Quick one. This is HelixServe, a mini LLM serving engine running on a single NVIDIA L4. "
    "I built this as a runtime internals project, not another chatbot wrapper. "
    "The baseline problem was simple: contiguous KV allocation wastes memory, long prefills block short requests, "
    "and decode launches repeat too much CPU work. "
    "Now this is the live system. Health is green, and stats confirm CUDA Graph decode, Triton kernel path, "
    "and a CUDA C plus plus extension are active. "
    "The architecture is split cleanly into server, scheduler, paged KV cache, kernels, and extension, "
    "so each optimization is measurable. "
    "In the phase table, baseline throughput is around one seventy six tokens per second. "
    "With continuous batching, chunked prefill, and CUDA Graph, we reach around one thousand eight tokens per second "
    "in this setup, with a major inter token latency improvement. "
    "Concrete example: if one request has a long prompt and another is a short chat turn, "
    "chunked prefill helps prevent the long one from monopolizing decode. "
    "Prefix cache hit rate is near one on repeated prefix workloads. "
    "This is still early and improving, but it is useful in practice today. "
    "Product and GitHub links are in the post."
)


SCENES: List[Dict[str, Any]] = [
    {
        "id": "intro",
        "type": "card",
        "duration": 7.0,
        "overlay": "HelixServe L4 Demo",
        "subtitle": "Mini LLM runtime internals project",
        "title": "HelixServe",
        "card_lines": [
            "Mini LLM serving engine on GCP L4",
            "Paged KV | Continuous batching | Chunked prefill",
            "Prefix cache | CUDA Graph decode | Triton + CUDA C++",
        ],
    },
    {
        "id": "health",
        "type": "image",
        "duration": 8.0,
        "overlay": "Live Health Check",
        "subtitle": "Endpoint is live on a single L4 GPU",
        "image": "01_healthz.png",
    },
    {
        "id": "stats",
        "type": "image",
        "duration": 10.0,
        "overlay": "Runtime Flags Active",
        "subtitle": "CUDA Graph, Triton kernel, and CUDA C++ path are enabled",
        "image": "02_runtime_stats.png",
    },
    {
        "id": "architecture",
        "type": "card",
        "duration": 10.0,
        "overlay": "Architecture Split",
        "subtitle": "Server, scheduler, cache, kernels, extension",
        "title": "Why This Layout",
        "card_lines": [
            "server/: API + streaming",
            "engine/: scheduler + decode loop",
            "cache/: paged KV allocator + prefix cache",
            "kernels/: Triton RMSNorm",
            "cuda_ext/: CUDA C++ hot op",
        ],
    },
    {
        "id": "phase_table",
        "type": "image",
        "duration": 14.0,
        "overlay": "Measured Phase Gains",
        "subtitle": "Throughput moves from 176 to 1008 tok/s in this setup",
        "image": "05_phase_table.png",
    },
    {
        "id": "suite",
        "type": "image",
        "duration": 12.0,
        "overlay": "Fresh Live Bench",
        "subtitle": "Short, long, mixed, repeated-prefix, and burst traffic",
        "image": "04_live_suite.png",
    },
    {
        "id": "example",
        "type": "image",
        "duration": 9.0,
        "overlay": "Concrete Example",
        "subtitle": "Chunked prefill keeps short turns responsive",
        "image": "03_completion.png",
    },
    {
        "id": "tests",
        "type": "image",
        "duration": 8.0,
        "overlay": "Reproducible Build",
        "subtitle": "Tests pass and reports are committed",
        "image": "06_tests.png",
    },
    {
        "id": "outro",
        "type": "card",
        "duration": 8.0,
        "overlay": "Early But Useful",
        "subtitle": "Product + GitHub links in post",
        "title": "Build. Measure. Improve.",
        "card_lines": [
            "Thanks for watching.",
            "Feedback welcome on scheduler and kernel roadmap.",
        ],
    },
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for font_path in candidates:
        path = Path(font_path)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = []
    for word in words:
        probe = " ".join(current + [word])
        if draw.textlength(probe, font=font) <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def _card_background(title: str, card_lines: List[str]) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(12, 18, 34))
    draw = ImageDraw.Draw(img)

    for i in range(HEIGHT):
        blend = i / max(1, HEIGHT - 1)
        color = (
            int(13 + 18 * blend),
            int(26 + 12 * blend),
            int(53 + 38 * blend),
        )
        draw.line([(0, i), (WIDTH, i)], fill=color)

    draw.rectangle([(70, 110), (WIDTH - 70, HEIGHT - 120)], outline=(83, 124, 178), width=3)

    title_font = _load_font(86)
    body_font = _load_font(44)
    accent_font = _load_font(34)

    draw.text((110, 170), title, font=title_font, fill=(238, 245, 255))
    draw.text((110, 260), "Runtime internals focused demo", font=accent_font, fill=(173, 203, 246))

    y = 360
    for line in card_lines:
        draw.text((130, y), f"- {line}", font=body_font, fill=(224, 235, 255))
        y += 76

    return img


def _scene_base(scene: Dict[str, Any]) -> Image.Image:
    if scene["type"] == "card":
        return _card_background(scene["title"], scene["card_lines"])

    image_path = SCREENSHOT_DIR / scene["image"]
    if not image_path.exists():
        raise FileNotFoundError(f"Missing screenshot: {image_path}")
    image = Image.open(image_path).convert("RGB")
    return ImageOps.fit(image, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS)


def _zoom_frame(base: Image.Image, progress: float, zoom_span: float) -> Image.Image:
    zoom = 1.0 + zoom_span * progress
    if zoom <= 1.0001:
        return base.copy()
    crop_w = int(WIDTH / zoom)
    crop_h = int(HEIGHT / zoom)
    x0 = (WIDTH - crop_w) // 2
    y0 = (HEIGHT - crop_h) // 2
    cropped = base.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    return cropped.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def _draw_overlay(frame: Image.Image, scene: Dict[str, Any]) -> Image.Image:
    rgba = frame.convert("RGBA")
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Top overlay
    draw.rectangle([(0, 0), (WIDTH, 110)], fill=(8, 14, 26, 190))
    label_font = _load_font(50)
    draw.text((56, 28), scene["overlay"], font=label_font, fill=(235, 244, 255, 255))

    # Bottom subtitle box
    box_h = 130
    draw.rectangle([(70, HEIGHT - box_h - 50), (WIDTH - 70, HEIGHT - 50)], fill=(6, 10, 18, 190))
    subtitle_font = _load_font(42)
    subtitle_draw = ImageDraw.Draw(overlay)
    lines = _wrap_text(subtitle_draw, scene["subtitle"], subtitle_font, max_width=WIDTH - 220)
    y = HEIGHT - box_h - 16
    for line in lines[:2]:
        subtitle_draw.text((110, y), line, font=subtitle_font, fill=(245, 248, 255, 255))
        y += 48

    composed = Image.alpha_composite(rgba, overlay).convert("RGB")
    return composed


def _fade(frame: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


async def _synthesize_voiceover(text: str, out_path: Path) -> None:
    communicate = edge_tts.Communicate(text=text, voice=VOICE, rate="-3%")
    await communicate.save(str(out_path))


def _timeline_payload() -> Dict[str, Any]:
    payload: Dict[str, Any] = {"fps": FPS, "scenes": []}
    cursor = 0.0
    for scene in SCENES:
        start = cursor
        end = start + float(scene["duration"])
        payload["scenes"].append(
            {
                "id": scene["id"],
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "overlay": scene["overlay"],
                "subtitle": scene["subtitle"],
            }
        )
        cursor = end
    payload["duration_s"] = round(cursor, 3)
    payload["voice"] = VOICE
    return payload


def build_video(keep_silent: bool = False) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    voiceover_txt = FINAL_DIR / "voiceover_script.txt"
    voiceover_mp3 = FINAL_DIR / "voiceover_en_in_neerja.mp3"
    silent_mp4 = FINAL_DIR / "helixserve_linkedin_final_silent.mp4"
    final_mp4 = FINAL_DIR / "helixserve_linkedin_final.mp4"
    timeline_json = FINAL_DIR / "timeline.json"

    voiceover_txt.write_text(VOICEOVER_SCRIPT + "\n", encoding="utf-8")
    timeline_json.write_text(json.dumps(_timeline_payload(), indent=2), encoding="utf-8")

    print("Generating voiceover...")
    asyncio.run(_synthesize_voiceover(VOICEOVER_SCRIPT, voiceover_mp3))

    print("Rendering silent video...")
    with imageio.get_writer(
        silent_mp4,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        for scene in SCENES:
            base = _scene_base(scene)
            frames = max(1, int(scene["duration"] * FPS))
            zoom_span = 0.05 if scene["type"] == "image" else 0.02

            for i in range(frames):
                progress = 0.0 if frames == 1 else i / (frames - 1)
                frame = _zoom_frame(base, progress, zoom_span=zoom_span)
                frame = _draw_overlay(frame, scene)
                arr = np.asarray(frame, dtype=np.uint8)

                fade_in_frames = min(10, frames // 4)
                fade_out_frames = min(10, frames // 4)
                if i < fade_in_frames:
                    factor = 0.25 + 0.75 * (i / max(1, fade_in_frames))
                    arr = _fade(arr, factor)
                elif i >= frames - fade_out_frames:
                    out_pos = i - (frames - fade_out_frames)
                    factor = 1.0 - 0.25 * (out_pos / max(1, fade_out_frames))
                    arr = _fade(arr, factor)

                writer.append_data(arr)

    print("Muxing voiceover with video...")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(silent_mp4),
            "-i",
            str(voiceover_mp3),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(final_mp4),
        ],
        check=True,
    )

    if not keep_silent and silent_mp4.exists():
        silent_mp4.unlink()

    summary = {
        "final_video": str(final_mp4.relative_to(ROOT)),
        "voiceover_audio": str(voiceover_mp3.relative_to(ROOT)),
        "voiceover_script": str(voiceover_txt.relative_to(ROOT)),
        "timeline": str(timeline_json.relative_to(ROOT)),
    }
    if keep_silent:
        summary["silent_video"] = str(silent_mp4.relative_to(ROOT))
    (FINAL_DIR / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build final narrated HelixServe demo video.")
    parser.add_argument(
        "--keep-silent",
        action="store_true",
        help="Keep the intermediate silent mp4 after muxing.",
    )
    args = parser.parse_args()
    build_video(keep_silent=args.keep_silent)
