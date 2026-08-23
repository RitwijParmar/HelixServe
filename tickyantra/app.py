from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from tickyantra import __version__
from tickyantra.config import Settings
from tickyantra.controller import Admission, QueueTimeoutError, SLOAdmissionController
from tickyantra.metrics import Metrics


def _request_text(payload: dict[str, Any]) -> str:
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        return prompt
    messages = payload.get("messages")
    if isinstance(messages, list):
        system = [
            str(item.get("content", ""))
            for item in messages
            if isinstance(item, dict) and item.get("role") == "system"
        ]
        if system:
            return "\n".join(system)
        return "\n".join(
            str(item.get("content", "")) for item in messages if isinstance(item, dict)
        )
    return ""


def _prefix_key(payload: dict[str, Any], chars: int) -> str:
    text = _request_text(payload)[:chars].encode("utf-8", errors="ignore")
    return hashlib.blake2s(text, digest_size=8).hexdigest()


def create_app(
    settings: Settings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    cfg = settings or Settings.from_env()
    cfg.validate()
    controller = SLOAdmissionController(cfg)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(
            base_url=cfg.upstream_url, timeout=None, transport=transport
        )
        yield
        await app.state.client.aclose()

    app = FastAPI(title="TickYantra", version=__version__, lifespan=lifespan)
    app.state.controller = controller
    app.state.metrics = metrics
    app.state.settings = cfg

    def authorized(authorization: str | None) -> bool:
        if not cfg.api_key:
            return True
        return authorization == f"Bearer {cfg.api_key}"

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    async def readyz(request: Request) -> Response:
        try:
            upstream = await request.app.state.client.get("/health", timeout=5.0)
            if upstream.status_code >= 400:
                return JSONResponse({"status": "not_ready"}, status_code=503)
            return JSONResponse({"status": "ready", "upstream": "sglang"})
        except httpx.HTTPError as exc:
            return JSONResponse({"status": "not_ready", "reason": type(exc).__name__}, 503)

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        return {
            "version": __version__,
            "upstream": cfg.upstream_url,
            "controller": await controller.snapshot(),
        }

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.api_route("/v1/models", methods=["GET"])
    async def models(request: Request, authorization: str | None = Header(default=None)) -> Response:
        if not authorized(authorization):
            return JSONResponse({"error": "unauthorized"}, 401)
        upstream = await request.app.state.client.get("/v1/models")
        return Response(upstream.content, upstream.status_code, media_type=upstream.headers.get("content-type"))

    async def proxy_generation(
        request: Request, authorization: str | None, path: str
    ) -> Response:
        if not authorized(authorization):
            return JSONResponse({"error": "unauthorized"}, 401)
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return JSONResponse({"error": "invalid JSON"}, 400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "request body must be an object"}, 400)

        prefix = _prefix_key(payload, cfg.prefix_chars)
        started = time.perf_counter()
        try:
            admission = await controller.acquire(prefix)
        except QueueTimeoutError:
            metrics.requests.labels("rejected").inc()
            return JSONResponse(
                {"error": {"type": "overloaded", "message": "admission deadline exceeded"}},
                429,
                headers={"Retry-After": "1"},
            )
        metrics.queue_wait.observe(admission.queue_wait_s)
        snapshot = await controller.snapshot()
        metrics.active.set(snapshot["active"])
        metrics.limit.set(snapshot["limit"])
        metrics.queued.set(snapshot["queued"])

        headers = {"x-request-id": request.headers.get("x-request-id", str(uuid.uuid4()))}
        if authorization and not cfg.api_key:
            headers["authorization"] = authorization
        stream = bool(payload.get("stream", False))
        if not stream:
            return await _proxy_non_streaming(
                request, path, payload, headers, admission, started, controller, metrics
            )
        return await _proxy_streaming(
            request, path, payload, headers, admission, started, controller, metrics
        )

    @app.post("/v1/completions")
    async def completions(request: Request, authorization: str | None = Header(default=None)):
        return await proxy_generation(request, authorization, "/v1/completions")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
        return await proxy_generation(request, authorization, "/v1/chat/completions")

    return app


async def _proxy_non_streaming(
    request: Request,
    path: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    admission: Admission,
    started: float,
    controller: SLOAdmissionController,
    metrics: Metrics,
) -> Response:
    ttft_ms: float | None = None
    try:
        upstream = await request.app.state.client.post(path, json=payload, headers=headers)
        ttft_ms = (time.perf_counter() - admission.admitted_at) * 1000.0
        metrics.ttft.observe(ttft_ms / 1000.0)
        metrics.requests.labels("completed" if upstream.status_code < 400 else "upstream_error").inc()
        return Response(
            upstream.content,
            upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )
    except httpx.HTTPError as exc:
        metrics.requests.labels("upstream_error").inc()
        return JSONResponse({"error": {"type": "upstream", "message": type(exc).__name__}}, 502)
    finally:
        elapsed = time.perf_counter() - started
        metrics.e2e.observe(elapsed)
        await controller.release(admission, ttft_ms=ttft_ms)


async def _proxy_streaming(
    request: Request,
    path: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    admission: Admission,
    started: float,
    controller: SLOAdmissionController,
    metrics: Metrics,
) -> Response:
    upstream_request = request.app.state.client.build_request("POST", path, json=payload, headers=headers)
    try:
        upstream = await request.app.state.client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        metrics.requests.labels("upstream_error").inc()
        await controller.release(admission, ttft_ms=None)
        return JSONResponse({"error": {"type": "upstream", "message": type(exc).__name__}}, 502)

    if upstream.status_code >= 400:
        body = await upstream.aread()
        await upstream.aclose()
        metrics.requests.labels("upstream_error").inc()
        await controller.release(admission, ttft_ms=None)
        return Response(body, upstream.status_code, media_type=upstream.headers.get("content-type"))

    async def body() -> AsyncIterator[bytes]:
        ttft_ms: float | None = None
        try:
            async for chunk in upstream.aiter_bytes():
                if ttft_ms is None and b"data:" in chunk and b"[DONE]" not in chunk:
                    ttft_ms = (time.perf_counter() - admission.admitted_at) * 1000.0
                    metrics.ttft.observe(ttft_ms / 1000.0)
                yield chunk
            metrics.requests.labels("completed").inc()
        finally:
            await upstream.aclose()
            metrics.e2e.observe(time.perf_counter() - started)
            await controller.release(admission, ttft_ms=ttft_ms)

    return StreamingResponse(body(), media_type="text/event-stream")


app = create_app()
