from fastapi.testclient import TestClient

from engine.config import EngineConfig
from server.main import create_app


def test_completion_endpoint_smoke() -> None:
    app = create_app(
        EngineConfig(
            use_toy_backend=True,
            device="cpu",
            enable_cuda_graph_decode=False,
            kv_total_blocks=256,
            max_decode_batch_size=4,
            prefill_chunk_size=16,
        )
    )

    with TestClient(app) as client:
        resp = client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_tokens": 8, "temperature": 0.0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data


def test_metrics_endpoint_smoke() -> None:
    app = create_app(EngineConfig(use_toy_backend=True, device="cpu", enable_cuda_graph_decode=False))
    with TestClient(app) as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "helixserve_requests_total" in resp.text
