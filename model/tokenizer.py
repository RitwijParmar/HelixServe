from __future__ import annotations

from typing import List


class ByteTokenizer:
    """
    Minimal UTF-8 byte tokenizer.

    Token ids 0..255 map to raw bytes. 256 is EOS.
    """

    eos_token_id = 256
    pad_token_id = 0
    vocab_size = 257

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, token_ids: List[int]) -> str:
        raw = bytes(token for token in token_ids if 0 <= token < 256)
        return raw.decode("utf-8", errors="ignore")
