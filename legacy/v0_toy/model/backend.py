from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Sequence


class DecoderBackend(ABC):
    eos_token_id: int
    pad_token_id: int
    device: str

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def detokenize(self, token_ids: Sequence[int]) -> str:
        raise NotImplementedError

    @abstractmethod
    def prefill(self, request_id: str, tokens: Sequence[int]) -> None:
        raise NotImplementedError

    @abstractmethod
    def decode_batch(
        self,
        request_ids: Sequence[str],
        prev_tokens: Sequence[int],
        *,
        temperature: float,
        top_k: int,
    ) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def remove_request(self, request_id: str) -> None:
        raise NotImplementedError

    def export_state(self, request_id: str) -> Any:
        return None

    def import_state(self, request_id: str, state: Any) -> bool:
        return False

    @property
    def supports_cuda_graph_decode(self) -> bool:
        return False

    @property
    def uses_triton_kernel(self) -> bool:
        return False
