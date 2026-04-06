from __future__ import annotations

import random
from typing import List

BASE_PREFIX = (
    "HelixServe runtime experiment. "
    "Measure TTFT, inter-token latency, and throughput under mixed request patterns. "
)

SHORT_SUFFIXES = [
    "Summarize scheduler behavior.",
    "Explain paged KV allocation.",
    "Describe decode prioritization.",
    "Give one optimization tip.",
]

LONG_SUFFIX = (
    "Write a technical note about how chunked prefill can reduce decode interference "
    "in a continuous batching system and mention queueing tradeoffs. "
)


def repeated_prefix_prompt(extra: str) -> str:
    return BASE_PREFIX + extra


def make_short_prompt() -> str:
    return repeated_prefix_prompt(random.choice(SHORT_SUFFIXES))


def make_long_prompt() -> str:
    return repeated_prefix_prompt(LONG_SUFFIX * 6)


def generate_prompts(mode: str, count: int) -> List[str]:
    prompts: List[str] = []
    for _ in range(count):
        if mode == "short":
            prompts.append(make_short_prompt())
        elif mode == "long":
            prompts.append(make_long_prompt())
        elif mode == "mixed":
            prompts.append(make_short_prompt() if random.random() < 0.7 else make_long_prompt())
        elif mode == "repeated_prefix":
            tail = random.choice(SHORT_SUFFIXES + [LONG_SUFFIX])
            prompts.append(repeated_prefix_prompt(tail))
        else:
            raise ValueError(f"unknown mode: {mode}")
    return prompts
