from __future__ import annotations

import asyncio

import pytest

from tickyantra.config import Settings
from tickyantra.controller import QueueTimeoutError, SLOAdmissionController


@pytest.mark.asyncio
async def test_static_limit_queues_and_releases() -> None:
    controller = SLOAdmissionController(Settings(mode="static", static_limit=1))
    first = await controller.acquire("a")
    second_task = asyncio.create_task(controller.acquire("b"))
    await asyncio.sleep(0)
    assert (await controller.snapshot())["queued"] == 1
    await controller.release(first, ttft_ms=100)
    second = await second_task
    assert second.queue_wait_s >= 0
    await controller.release(second, ttft_ms=100)
    assert (await controller.snapshot())["active"] == 0


@pytest.mark.asyncio
async def test_queue_timeout_rejects() -> None:
    controller = SLOAdmissionController(
        Settings(mode="static", static_limit=1, queue_timeout_ms=5)
    )
    first = await controller.acquire("a")
    with pytest.raises(QueueTimeoutError):
        await controller.acquire("b")
    await controller.release(first, ttft_ms=100)
    assert (await controller.snapshot())["rejected_total"] == 1


@pytest.mark.asyncio
async def test_adaptive_controller_reduces_limit_on_slo_violation() -> None:
    settings = Settings(
        mode="adaptive",
        adaptive_min_limit=1,
        adaptive_initial_limit=3,
        adaptive_max_limit=5,
        sample_window=2,
        target_ttft_ms=100,
    )
    controller = SLOAdmissionController(settings)
    for _ in range(2):
        admission = await controller.acquire("same")
        await controller.release(admission, ttft_ms=250)
    assert (await controller.snapshot())["limit"] == 2


@pytest.mark.asyncio
async def test_native_mode_never_queues() -> None:
    controller = SLOAdmissionController(Settings(mode="native"))
    admissions = await asyncio.gather(*(controller.acquire(str(i)) for i in range(100)))
    assert (await controller.snapshot())["active"] == 100
    await asyncio.gather(*(controller.release(item, ttft_ms=1) for item in admissions))
