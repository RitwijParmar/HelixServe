from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

StepFn = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class CapturedDecodeGraph:
    batch_size: int
    static_tokens: torch.Tensor
    static_hidden_in: torch.Tensor
    static_logits_out: torch.Tensor
    static_hidden_out: torch.Tensor
    graph: torch.cuda.CUDAGraph


class CUDAGraphDecodeCache:
    """
    Captures and replays a steady decode step for fixed batch sizes.
    """

    def __init__(
        self,
        *,
        step_fn: StepFn,
        hidden_size: int,
        vocab_size: int,
        device: torch.device,
        warmup_iters: int = 3,
    ) -> None:
        self._step_fn = step_fn
        self._hidden_size = hidden_size
        self._vocab_size = vocab_size
        self._device = device
        self._warmup_iters = warmup_iters
        self._graphs: Dict[int, CapturedDecodeGraph] = {}
        self._capture_failed = False

    def _capture(self, batch_size: int) -> Optional[CapturedDecodeGraph]:
        if self._capture_failed:
            return None
        if not torch.cuda.is_available() or self._device.type != "cuda":
            return None

        try:
            static_tokens = torch.zeros((batch_size, 1), device=self._device, dtype=torch.long)
            static_hidden_in = torch.zeros(
                (1, batch_size, self._hidden_size), device=self._device, dtype=torch.float32
            )
            static_logits_out = torch.zeros(
                (batch_size, self._vocab_size), device=self._device, dtype=torch.float32
            )
            static_hidden_out = torch.zeros_like(static_hidden_in)

            stream = torch.cuda.Stream(device=self._device)
            stream.wait_stream(torch.cuda.current_stream(self._device))
            with torch.cuda.stream(stream):
                for _ in range(self._warmup_iters):
                    logits, hidden = self._step_fn(static_tokens, static_hidden_in)
                    static_logits_out.copy_(logits[:, -1, :])
                    static_hidden_out.copy_(hidden)
            torch.cuda.current_stream(self._device).wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                logits, hidden = self._step_fn(static_tokens, static_hidden_in)
                static_logits_out.copy_(logits[:, -1, :])
                static_hidden_out.copy_(hidden)

            captured = CapturedDecodeGraph(
                batch_size=batch_size,
                static_tokens=static_tokens,
                static_hidden_in=static_hidden_in,
                static_logits_out=static_logits_out,
                static_hidden_out=static_hidden_out,
                graph=graph,
            )
            self._graphs[batch_size] = captured
            return captured
        except Exception:
            self._capture_failed = True
            return None

    def run(
        self,
        *,
        tokens: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = int(tokens.shape[0])
        captured = self._graphs.get(batch_size)
        if captured is None:
            captured = self._capture(batch_size)
        if captured is None:
            return None

        captured.static_tokens.copy_(tokens)
        captured.static_hidden_in.copy_(hidden)
        captured.graph.replay()

        return captured.static_logits_out.clone(), captured.static_hidden_out.clone()
