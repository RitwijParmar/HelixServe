from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

from engine.cuda_graph import CUDAGraphDecodeCache
from kernels.rmsnorm_triton import rms_norm_reference, rms_norm_triton
from model.backend import DecoderBackend
from model.tokenizer import ByteTokenizer


@dataclass
class ToyBackendConfig:
    hidden_size: int = 512
    seed: int = 7
    enable_cuda_graph_decode: bool = True
    enable_triton_rmsnorm: bool = True


class TinyGRULM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self, tokens: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(tokens)
        out, hidden_out = self.gru(x, hidden)
        logits = self.lm_head(out)
        return logits, hidden_out


class ToyDecoderBackend(DecoderBackend):
    def __init__(self, *, device: str = "cuda", config: ToyBackendConfig | None = None) -> None:
        self._tokenizer = ByteTokenizer()
        self.eos_token_id = self._tokenizer.eos_token_id
        self.pad_token_id = self._tokenizer.pad_token_id

        cfg = config or ToyBackendConfig()
        torch.manual_seed(cfg.seed)

        if device == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self.device = str(self._device)

        self._model = TinyGRULM(self._tokenizer.vocab_size, cfg.hidden_size).to(self._device)
        self._model.eval()
        self._hidden_size = cfg.hidden_size
        self._states: Dict[str, torch.Tensor] = {}
        self._use_triton_rmsnorm = cfg.enable_triton_rmsnorm and self._device.type == "cuda"
        self._rms_weight = torch.ones(
            (self._hidden_size,),
            device=self._device,
            dtype=torch.float32,
        )

        self._graph_cache: CUDAGraphDecodeCache | None = None
        if cfg.enable_cuda_graph_decode and self._device.type == "cuda":
            self._graph_cache = CUDAGraphDecodeCache(
                step_fn=self._decode_step,
                hidden_size=self._hidden_size,
                vocab_size=self._tokenizer.vocab_size,
                device=self._device,
            )

    @property
    def supports_cuda_graph_decode(self) -> bool:
        return self._graph_cache is not None

    @property
    def uses_triton_kernel(self) -> bool:
        return self._use_triton_rmsnorm

    def tokenize(self, text: str) -> List[int]:
        return self._tokenizer.encode(text)

    def detokenize(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(token_ids))

    def _get_hidden(self, request_id: str) -> torch.Tensor:
        hidden = self._states.get(request_id)
        if hidden is None:
            hidden = torch.zeros((1, 1, self._hidden_size), device=self._device, dtype=torch.float32)
            self._states[request_id] = hidden
        return hidden

    @torch.no_grad()
    def prefill(self, request_id: str, tokens: Sequence[int]) -> None:
        if not tokens:
            return
        hidden = self._get_hidden(request_id)
        tokens_t = torch.tensor([list(tokens)], device=self._device, dtype=torch.long)
        _, hidden_out = self._model(tokens_t, hidden)
        self._states[request_id] = hidden_out

    def _decode_step(
        self,
        tokens_t: torch.Tensor,
        hidden_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._model.embedding(tokens_t)
        out, hidden_out = self._model.gru(x, hidden_t)
        last_hidden = out[:, -1, :].to(dtype=torch.float32)

        if self._use_triton_rmsnorm:
            try:
                last_hidden = rms_norm_triton(last_hidden, self._rms_weight)
            except Exception:
                # Keep serving if Triton codegen fails at runtime.
                self._use_triton_rmsnorm = False
                last_hidden = rms_norm_reference(last_hidden, self._rms_weight)

        logits_last = self._model.lm_head(last_hidden)
        return logits_last.unsqueeze(1), hidden_out

    @staticmethod
    def _sample_next(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)

        scaled = logits / max(temperature, 1e-5)
        if top_k > 0 and top_k < scaled.shape[-1]:
            values, indices = torch.topk(scaled, k=top_k, dim=-1)
            probs = torch.softmax(values, dim=-1)
            sample = torch.multinomial(probs, num_samples=1).squeeze(-1)
            return indices.gather(-1, sample.unsqueeze(-1)).squeeze(-1)

        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def decode_batch(
        self,
        request_ids: Sequence[str],
        prev_tokens: Sequence[int],
        *,
        temperature: float,
        top_k: int,
    ) -> List[int]:
        if not request_ids:
            return []

        hidden = torch.cat([self._get_hidden(rid) for rid in request_ids], dim=1)
        tokens_t = torch.tensor(prev_tokens, device=self._device, dtype=torch.long).unsqueeze(-1)

        logits_last: torch.Tensor
        hidden_out: torch.Tensor

        graph_result = None
        if (
            self._graph_cache is not None
            and temperature <= 0
            and top_k <= 0
            and tokens_t.shape[0] > 0
        ):
            graph_result = self._graph_cache.run(tokens=tokens_t, hidden=hidden)

        if graph_result is not None:
            logits_last, hidden_out = graph_result
        else:
            logits, hidden_out = self._decode_step(tokens_t, hidden)
            logits_last = logits[:, -1, :]

        next_tokens = self._sample_next(logits_last, temperature=temperature, top_k=top_k)

        for idx, rid in enumerate(request_ids):
            self._states[rid] = hidden_out[:, idx : idx + 1, :]

        return next_tokens.detach().cpu().tolist()

    def remove_request(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    def export_state(self, request_id: str) -> Any:
        hidden = self._states.get(request_id)
        if hidden is None:
            return None
        return hidden.detach().clone()

    def import_state(self, request_id: str, state: Any) -> bool:
        if state is None:
            return False
        if not isinstance(state, torch.Tensor):
            return False
        self._states[request_id] = state.to(self._device)
        return True
