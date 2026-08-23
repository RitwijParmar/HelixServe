from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional


class RequestState(str, Enum):
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class GenerationParams:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 0
    stop_token_ids: List[int] = field(default_factory=list)


@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    prompt_tokens: List[int]
    params: GenerationParams
    created_at: float = field(default_factory=time.time)
    state: RequestState = RequestState.QUEUED

    prefill_cursor: int = 0
    generated_tokens: List[int] = field(default_factory=list)
    cached_prefix_tokens: int = 0
    queue_entered_at: float = field(default_factory=time.time)
    decode_started_at: Optional[float] = None
    first_token_at: Optional[float] = None
    finished_at: Optional[float] = None

    # Prefix correctness: first decode token should not modify prompt page.
    force_new_decode_block: bool = False

    # For model decode input.
    last_token: Optional[int] = None


@dataclass
class StreamEvent:
    event: str
    request_id: str
    payload: Dict[str, Any]


class RequestHandle:
    def __init__(self, request: InferenceRequest) -> None:
        self.request = request
        self._events: asyncio.Queue[StreamEvent] = asyncio.Queue()
        self._done = asyncio.Event()

    async def push(self, event: StreamEvent) -> None:
        await self._events.put(event)
        if event.event in {"done", "error", "rejected"}:
            self._done.set()

    async def stream(self) -> AsyncIterator[StreamEvent]:
        while True:
            event = await self._events.get()
            yield event
            if event.event in {"done", "error", "rejected"}:
                return

    async def collect_text(self) -> str:
        pieces: List[str] = []
        async for event in self.stream():
            if event.event == "token":
                pieces.append(str(event.payload.get("text", "")))
        return "".join(pieces)

    async def wait(self) -> None:
        await self._done.wait()
