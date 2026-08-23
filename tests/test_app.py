from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from tickyantra.app import create_app
from tickyantra.config import Settings


def fake_sglang() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]}

    @app.post("/v1/completions")
    async def completions(request: Request):
        payload = await request.json()
        if payload.get("stream"):

            async def chunks():
                yield 'data: {"choices":[{"text":"hello"}]}\n\n'
                yield "data: [DONE]\n\n"

            return StreamingResponse(chunks(), media_type="text/event-stream")
        return {"choices": [{"text": "hello"}], "usage": {"completion_tokens": 1}}

    return app


@pytest.mark.asyncio
async def test_readiness_and_models_proxy() -> None:
    transport = httpx.ASGITransport(app=fake_sglang())
    app = create_app(Settings(mode="adaptive"), transport=transport)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/readyz")).status_code == 200
            response = await client.get("/v1/models")
            assert response.json()["data"][0]["id"].startswith("Qwen")


@pytest.mark.asyncio
async def test_non_streaming_proxy_and_stats() -> None:
    transport = httpx.ASGITransport(app=fake_sglang())
    app = create_app(Settings(mode="static", static_limit=2), transport=transport)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/completions", json={"prompt": "hello", "max_tokens": 1}
            )
            assert response.status_code == 200
            assert response.json()["choices"][0]["text"] == "hello"
            stats = (await client.get("/stats")).json()
            assert stats["controller"]["admitted_total"] == 1
            assert stats["controller"]["active"] == 0


@pytest.mark.asyncio
async def test_streaming_proxy_releases_admission() -> None:
    transport = httpx.ASGITransport(app=fake_sglang())
    app = create_app(Settings(mode="static", static_limit=1), transport=transport)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with client.stream(
                "POST", "/v1/completions", json={"prompt": "hello", "stream": True}
            ) as response:
                body = b"".join([part async for part in response.aiter_bytes()])
            assert b"hello" in body
            assert (await client.get("/stats")).json()["controller"]["active"] == 0


@pytest.mark.asyncio
async def test_api_key_is_enforced() -> None:
    transport = httpx.ASGITransport(app=fake_sglang())
    app = create_app(Settings(mode="native", api_key="secret"), transport=transport)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            assert (await client.get("/v1/models")).status_code == 401
            response = await client.get(
                "/v1/models", headers={"authorization": "Bearer secret"}
            )
            assert response.status_code == 200
