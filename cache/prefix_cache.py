from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class PrefixCacheEntry:
    key: str
    token_count: int
    block_ids: List[int]
    created_at: float
    last_hit_at: float
    hits: int = 0
    backend_state: Any = None


class PrefixCache:
    """Prefix cache for KV page reuse with LRU eviction."""

    def __init__(
        self,
        *,
        block_size: int,
        max_entries: int,
        max_cached_tokens: int,
        pin_blocks: Callable[[str, List[int]], None],
        unpin_blocks: Callable[[str, List[int]], None],
        cacheable_prefix_lengths: Optional[Sequence[int]] = None,
        min_prefix_tokens: int = 1,
    ) -> None:
        self.block_size = block_size
        self.max_entries = max_entries
        self.max_cached_tokens = max_cached_tokens
        self.min_prefix_tokens = min_prefix_tokens

        self._pin_blocks = pin_blocks
        self._unpin_blocks = unpin_blocks
        self._cacheable_prefix_lengths = sorted(set(cacheable_prefix_lengths or []))

        self._entries: Dict[str, PrefixCacheEntry] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._lookups = 0
        self._hits = 0

    @staticmethod
    def _hash_tokens(tokens: Sequence[int]) -> str:
        digest = hashlib.blake2b(digest_size=16)
        for token in tokens:
            digest.update(int(token).to_bytes(4, byteorder="little", signed=False))
        return digest.hexdigest()

    def _make_key(self, prefix_tokens: Sequence[int]) -> str:
        return f"{len(prefix_tokens)}:{self._hash_tokens(prefix_tokens)}"

    def _candidate_lengths(self, total_tokens: int) -> List[int]:
        lengths = {total_tokens}
        for length in self._cacheable_prefix_lengths:
            if self.min_prefix_tokens <= length <= total_tokens:
                lengths.add(length)
        # Block boundaries are natural checkpoints for page sharing.
        block_boundary = (total_tokens // self.block_size) * self.block_size
        while block_boundary >= self.min_prefix_tokens and block_boundary > 0:
            lengths.add(block_boundary)
            block_boundary -= self.block_size
        return sorted(lengths, reverse=True)

    def _entry_owner(self, key: str) -> str:
        return f"prefix:{key}"

    def _set_entry(
        self,
        key: str,
        token_count: int,
        block_ids: List[int],
        backend_state: Any,
    ) -> None:
        owner = self._entry_owner(key)
        new_entry = PrefixCacheEntry(
            key=key,
            token_count=token_count,
            block_ids=list(block_ids),
            created_at=time.time(),
            last_hit_at=time.time(),
            hits=0,
            backend_state=backend_state,
        )

        old = self._entries.get(key)
        if old is not None:
            self._unpin_blocks(self._entry_owner(old.key), old.block_ids)
            self._lru.pop(key, None)

        self._pin_blocks(owner, new_entry.block_ids)
        self._entries[key] = new_entry
        self._lru[key] = None
        self._lru.move_to_end(key, last=True)

    def _evict_if_needed(self) -> None:
        while self._entries and (
            len(self._entries) > self.max_entries
            or self.cached_tokens() > self.max_cached_tokens
        ):
            key, _ = self._lru.popitem(last=False)
            entry = self._entries.pop(key)
            self._unpin_blocks(self._entry_owner(key), entry.block_ids)

    def insert(
        self,
        prompt_tokens: Sequence[int],
        block_ids: Sequence[int],
        *,
        backend_state: Any = None,
    ) -> None:
        token_count = len(prompt_tokens)
        if token_count < self.min_prefix_tokens or not block_ids:
            return

        for length in self._candidate_lengths(token_count):
            if length < self.min_prefix_tokens:
                continue
            prefix_tokens = prompt_tokens[:length]
            key = self._make_key(prefix_tokens)
            prefix_block_count = math.ceil(length / self.block_size)
            self._set_entry(
                key,
                length,
                list(block_ids[:prefix_block_count]),
                backend_state,
            )

        self._evict_if_needed()

    def lookup(self, prompt_tokens: Sequence[int]) -> Optional[PrefixCacheEntry]:
        self._lookups += 1
        if len(prompt_tokens) < self.min_prefix_tokens:
            return None

        for length in self._candidate_lengths(len(prompt_tokens)):
            if length > len(prompt_tokens):
                continue
            key = self._make_key(prompt_tokens[:length])
            entry = self._entries.get(key)
            if entry is None:
                continue
            self._hits += 1
            entry.hits += 1
            entry.last_hit_at = time.time()
            self._lru.pop(key, None)
            self._lru[key] = None
            return entry
        return None

    def cached_tokens(self) -> int:
        return sum(entry.token_count for entry in self._entries.values())

    def stats(self) -> dict:
        hit_rate = self._hits / self._lookups if self._lookups > 0 else 0.0
        return {
            "entries": len(self._entries),
            "cached_tokens": self.cached_tokens(),
            "lookups": self._lookups,
            "hits": self._hits,
            "hit_rate": hit_rate,
        }

    def clear(self) -> None:
        for key, entry in list(self._entries.items()):
            self._unpin_blocks(self._entry_owner(key), entry.block_ids)
        self._entries.clear()
        self._lru.clear()
