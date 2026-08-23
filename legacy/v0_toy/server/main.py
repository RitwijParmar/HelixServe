from __future__ import annotations

import json
import os
from typing import AsyncIterator, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from engine.config import EngineConfig
from engine.runtime import HelixEngine


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int = Field(default=0, ge=0)
    stream: bool = False


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    model: str
    choices: list[CompletionChoice]
    usage: dict


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int_list(name: str, default: list[int]) -> list[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or default


def load_config_from_env() -> EngineConfig:
    return EngineConfig(
        model_name=os.getenv("HELIX_MODEL", "sshleifer/tiny-gpt2"),
        use_toy_backend=_env_flag("HELIX_USE_TOY_BACKEND", True),
        device=os.getenv("HELIX_DEVICE", "cuda"),
        kv_block_size=int(os.getenv("HELIX_KV_BLOCK_SIZE", "16")),
        kv_total_blocks=int(os.getenv("HELIX_KV_TOTAL_BLOCKS", "4096")),
        max_queue_size=int(os.getenv("HELIX_MAX_QUEUE_SIZE", "2048")),
        max_decode_batch_size=int(os.getenv("HELIX_MAX_DECODE_BATCH", "16")),
        max_num_batched_tokens=int(os.getenv("HELIX_MAX_BATCHED_TOKENS", "1024")),
        prefill_chunk_size=int(os.getenv("HELIX_PREFILL_CHUNK", "128")),
        prefix_cache_max_entries=int(os.getenv("HELIX_PREFIX_MAX_ENTRIES", "4096")),
        prefix_cache_max_tokens=int(os.getenv("HELIX_PREFIX_MAX_TOKENS", "500000")),
        prefix_cache_min_tokens=int(os.getenv("HELIX_PREFIX_MIN_TOKENS", "16")),
        prefix_cache_lengths=_env_int_list("HELIX_PREFIX_LENGTHS", [64, 128, 256, 512]),
        enable_cuda_graph_decode=_env_flag("HELIX_ENABLE_CUDA_GRAPH", True),
        enable_triton_rmsnorm=_env_flag("HELIX_ENABLE_TRITON_RMSNORM", True),
        default_max_new_tokens=int(os.getenv("HELIX_DEFAULT_MAX_NEW_TOKENS", "128")),
        default_temperature=float(os.getenv("HELIX_DEFAULT_TEMPERATURE", "0.0")),
    )


def create_app(config: Optional[EngineConfig] = None) -> FastAPI:
    app = FastAPI(title="HelixServe", version="0.1.0")
    engine = HelixEngine(config or load_config_from_env())

    @app.on_event("startup")
    async def _startup() -> None:
        await engine.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await engine.stop()

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/stats")
    async def stats() -> dict:
        return engine.stats()

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        payload = engine.metrics_payload()
        return PlainTextResponse(payload.decode("utf-8"), media_type="text/plain")

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        handle = await engine.submit(
            prompt=req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        )

        if req.stream:
            async def _event_stream() -> AsyncIterator[str]:
                generated = 0
                async for event in handle.stream():
                    if event.event == "token":
                        generated += 1
                        payload = {
                            "id": event.request_id,
                            "object": "text_completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "text": event.payload.get("text", ""),
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                    elif event.event in {"error", "rejected"}:
                        payload = {
                            "id": event.request_id,
                            "error": event.payload,
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                    elif event.event == "done":
                        payload = {
                            "id": event.request_id,
                            "object": "text_completion.chunk",
                            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                            "usage": {
                                "prompt_tokens": len(handle.request.prompt_tokens),
                                "completion_tokens": generated,
                                "total_tokens": len(handle.request.prompt_tokens) + generated,
                            },
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(_event_stream(), media_type="text/event-stream")

        text = await handle.collect_text()
        usage = {
            "prompt_tokens": len(handle.request.prompt_tokens),
            "completion_tokens": len(handle.request.generated_tokens),
            "total_tokens": len(handle.request.prompt_tokens) + len(handle.request.generated_tokens),
        }

        return CompletionResponse(
            id=handle.request.request_id,
            model=engine.stats()["backend"]["type"],
            choices=[CompletionChoice(text=text, finish_reason="stop")],
            usage=usage,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict):
        # Lightweight OpenAI-style compatibility.
        messages = payload.get("messages", [])
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        max_tokens = int(payload.get("max_tokens", 128))
        temperature = float(payload.get("temperature", 0.0))
        top_k = int(payload.get("top_k", 0))
        stream = bool(payload.get("stream", False))

        req = CompletionRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            stream=stream,
        )
        return await completions(req)

    return app


app = create_app()


def run() -> None:
    host = os.getenv("HELIX_HOST", "0.0.0.0")
    port = int(os.getenv("HELIX_PORT", "8000"))
    uvicorn.run("server.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
