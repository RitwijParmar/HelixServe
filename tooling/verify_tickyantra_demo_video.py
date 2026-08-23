#!/usr/bin/env python3
"""Verify the final TickYantra demo video and emit machine-readable metadata."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "docs" / "assets" / "demo" / "tickyantra" / "final"
VIDEO = FINAL_DIR / "tickyantra_end_to_end_demo.mp4"


def main() -> int:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run([ffmpeg, "-i", str(VIDEO)], capture_output=True, text=True, check=False)
    metadata = result.stderr
    duration_match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", metadata)
    video_match = re.search(r"Video: ([^,]+).*?, (\d{3,5})x(\d{3,5})", metadata)
    audio_match = re.search(r"Audio: ([^,]+)", metadata)
    if not duration_match or not video_match:
        raise RuntimeError("Final video metadata is incomplete")

    hours, minutes, seconds = duration_match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    codec, width, height = video_match.groups()
    summary = {
        "video": str(VIDEO.relative_to(ROOT)),
        "duration_s": duration,
        "video_codec": codec.strip(),
        "audio_codec": audio_match.group(1).strip() if audio_match else None,
        "width": int(width),
        "height": int(height),
        "valid_1080p": (int(width), int(height)) == (1920, 1080),
        "valid_duration": 150 <= duration <= 210,
        "has_audio": audio_match is not None,
        "faststart": VIDEO.read_bytes()[:2_000_000].find(b"moov") >= 0,
        "size_bytes": VIDEO.stat().st_size,
    }
    summary["valid"] = all(
        [summary["valid_1080p"], summary["valid_duration"], summary["has_audio"], summary["faststart"]]
    )
    output = FINAL_DIR / "video_verification.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
