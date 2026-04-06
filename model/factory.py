from __future__ import annotations

from engine.config import EngineConfig
from model.backend import DecoderBackend
from model.hf_backend import TransformersDecoderBackend
from model.toy_backend import ToyBackendConfig, ToyDecoderBackend


def build_backend(config: EngineConfig) -> DecoderBackend:
    if config.use_toy_backend:
        return ToyDecoderBackend(
            device=config.device,
            config=ToyBackendConfig(
                enable_cuda_graph_decode=config.enable_cuda_graph_decode,
                enable_triton_rmsnorm=config.enable_triton_rmsnorm,
            ),
        )

    try:
        return TransformersDecoderBackend(config.model_name, device=config.device)
    except Exception:
        # Fallback keeps the engine bootable even if model download/auth fails.
        return ToyDecoderBackend(
            device=config.device,
            config=ToyBackendConfig(
                enable_cuda_graph_decode=config.enable_cuda_graph_decode,
                enable_triton_rmsnorm=config.enable_triton_rmsnorm,
            ),
        )
