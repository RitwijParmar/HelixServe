#!/usr/bin/env python3
"""Validate that a voiceover script is product-centric (no first-person singular)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FORBIDDEN = ("i", "me", "my", "mine", "myself")


def find_forbidden_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    return sorted({token for token in tokens if token in FORBIDDEN})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default="docs/assets/demo/final/voiceover_script.txt",
        help="Path to voiceover script text file.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    text = path.read_text(encoding="utf-8")
    forbidden = find_forbidden_tokens(text)
    summary = {
        "path": str(path),
        "forbidden_found": forbidden,
        "valid_product_voice": len(forbidden) == 0,
    }
    print(json.dumps(summary, indent=2))
    return 0 if not forbidden else 1


if __name__ == "__main__":
    raise SystemExit(main())

