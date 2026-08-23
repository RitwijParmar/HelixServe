from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

from model.backend import DecoderBackend

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None


class TransformersDecoderBackend(DecoderBackend):
    """
    Decoder backend using Hugging Face causal LM.

    For reliability across mixed sequence lengths, decode uses padded full-context
    forward passes instead of batched past-key-values packing.
    """

    def __init__(self, model_name: str, *, device: str = "cuda") -> None:
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise RuntimeError("transformers is not available")

        if device == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
            dtype = torch.float16
        else:
            self._device = torch.device("cpu")
            dtype = torch.float32

        self.device = str(self._device)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token_id is None:
            if self._tokenizer.eos_token_id is None:
                self._tokenizer.add_special_tokens({"eos_token": "<|eos|>"})
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self.eos_token_id = int(self._tokenizer.eos_token_id)
        self.pad_token_id = int(self._tokenizer.pad_token_id)

        self._model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self._model.to(self._device)
        self._model.eval()

        self._state_tokens: Dict[str, List[int]] = {}

    def tokenize(self, text: str) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, token_ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def prefill(self, request_id: str, tokens: Sequence[int]) -> None:
        if request_id not in self._state_tokens:
            self._state_tokens[request_id] = []
        self._state_tokens[request_id].extend(int(t) for t in tokens)

    @staticmethod
    def _sample_next(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)

        scaled = logits / max(temperature, 1e-5)
        if top_k > 0 and top_k < scaled.shape[-1]:
            vals, idx = torch.topk(scaled, k=top_k, dim=-1)
            probs = torch.softmax(vals, dim=-1)
            sample = torch.multinomial(probs, num_samples=1).squeeze(-1)
            return idx.gather(-1, sample.unsqueeze(-1)).squeeze(-1)

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

        contexts = [self._state_tokens.setdefault(rid, []) for rid in request_ids]
        for context, prev in zip(contexts, prev_tokens):
            if not context:
                context.append(int(prev))

        lengths = [len(ctx) for ctx in contexts]
        max_len = max(lengths)

        input_ids = torch.full(
            (len(contexts), max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=self._device,
        )
        attn_mask = torch.zeros_like(input_ids)

        for i, ctx in enumerate(contexts):
            seq = torch.tensor(ctx, dtype=torch.long, device=self._device)
            input_ids[i, : len(ctx)] = seq
            attn_mask[i, : len(ctx)] = 1

        outputs = self._model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits
        last_positions = torch.tensor([length - 1 for length in lengths], device=self._device)
        row_idx = torch.arange(len(contexts), device=self._device)
        last_logits = logits[row_idx, last_positions, :]

        next_tokens = self._sample_next(last_logits, temperature=temperature, top_k=top_k)
        next_list = next_tokens.detach().cpu().tolist()

        for rid, token in zip(request_ids, next_list):
            self._state_tokens[rid].append(int(token))

        return next_list

    def remove_request(self, request_id: str) -> None:
        self._state_tokens.pop(request_id, None)

    def export_state(self, request_id: str) -> Any:
        tokens = self._state_tokens.get(request_id)
        if tokens is None:
            return None
        return list(tokens)

    def import_state(self, request_id: str, state: Any) -> bool:
        if not isinstance(state, list):
            return False
        self._state_tokens[request_id] = [int(x) for x in state]
        return True
