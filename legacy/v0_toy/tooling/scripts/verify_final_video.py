#!/usr/bin/env python3
"""Verify final demo video metadata and write a JSON summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        default="docs/assets/demo/final/helixserve_linkedin_final.mp4",
        help="Path to final mp4.",
    )
    parser.add_argument(
        "--output",
        default="docs/assets/demo/final/video_verification.json",
        help="Path to write JSON verification.",
    )
    args = parser.parse_args()

    reader = imageio.get_reader(args.video)
    meta = reader.get_meta_data()
    reader.close()

    summary = {
        "video": args.video,
        "codec": meta.get("codec"),
        "audio_codec": meta.get("audio_codec"),
        "fps": meta.get("fps"),
        "size": meta.get("size"),
        "duration_s": meta.get("duration"),
        "valid_1080p": meta.get("size") == (1920, 1080),
        "valid_duration_window": 75 <= float(meta.get("duration", 0)) <= 120,
        "has_audio": bool(meta.get("audio_codec")),
    }
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

