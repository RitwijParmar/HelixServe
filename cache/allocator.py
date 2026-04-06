from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Set


class OutOfKVBlocksError(RuntimeError):
    """Raised when KV cache runs out of free blocks."""


@dataclass
class KVBlock:
    block_id: int
    used_tokens: int = 0
    refcount: int = 0
    request_owners: Set[str] = field(default_factory=set)
    pin_owners: Set[str] = field(default_factory=set)

    @property
    def is_free(self) -> bool:
        return self.refcount == 0


class KVBlockAllocator:
    """
    Fixed-size paged KV allocator.

    The allocator tracks ownership with reference counts so prefix cache entries
    can pin blocks while live requests attach/detach from the same pages.
    """

    def __init__(self, *, total_blocks: int, block_size: int) -> None:
        if total_blocks <= 0:
            raise ValueError("total_blocks must be > 0")
        if block_size <= 0:
            raise ValueError("block_size must be > 0")

        self.total_blocks = total_blocks
        self.block_size = block_size

        self._blocks: Dict[int, KVBlock] = {
            idx: KVBlock(block_id=idx) for idx in range(total_blocks)
        }
        self._free: deque[int] = deque(range(total_blocks))
        self._request_blocks: Dict[str, List[int]] = {}
        self._lock = threading.RLock()

    def _acquire_block(self) -> KVBlock:
        if not self._free:
            raise OutOfKVBlocksError("KV allocator exhausted")
        block_id = self._free.popleft()
        return self._blocks[block_id]

    def _release_block_if_unused(self, block: KVBlock) -> None:
        if block.refcount == 0:
            block.used_tokens = 0
            block.request_owners.clear()
            block.pin_owners.clear()
            self._free.append(block.block_id)

    def attach_blocks(self, request_id: str, block_ids: List[int]) -> None:
        """Attach existing blocks to a request (used for prefix cache hits)."""
        with self._lock:
            owned = self._request_blocks.setdefault(request_id, [])
            for block_id in block_ids:
                block = self._blocks[block_id]
                block.refcount += 1
                block.request_owners.add(request_id)
                owned.append(block_id)

    def pin_blocks(self, owner: str, block_ids: List[int]) -> None:
        """Pin blocks for a non-request owner (e.g., prefix cache entry)."""
        with self._lock:
            for block_id in block_ids:
                block = self._blocks[block_id]
                block.refcount += 1
                block.pin_owners.add(owner)

    def unpin_blocks(self, owner: str, block_ids: List[int]) -> None:
        with self._lock:
            for block_id in block_ids:
                block = self._blocks[block_id]
                if owner not in block.pin_owners:
                    continue
                block.pin_owners.remove(owner)
                block.refcount -= 1
                if block.refcount < 0:
                    raise RuntimeError(f"negative refcount on block {block_id}")
                self._release_block_if_unused(block)

    def append_tokens(
        self,
        request_id: str,
        num_tokens: int,
        *,
        force_new_block: bool = False,
    ) -> List[int]:
        """
        Append tokens to request pages.

        Returns newly allocated block ids.
        """
        if num_tokens < 0:
            raise ValueError("num_tokens must be >= 0")
        if num_tokens == 0:
            return []

        with self._lock:
            blocks = self._request_blocks.setdefault(request_id, [])
            allocated: List[int] = []
            remaining = num_tokens

            if force_new_block and blocks:
                # Mark current block as sealed for prefix correctness by forcing
                # a fresh page before appending decode tokens.
                current = self._blocks[blocks[-1]]
                if current.used_tokens < self.block_size:
                    current.used_tokens = self.block_size

            while remaining > 0:
                needs_new = not blocks
                if not needs_new:
                    current_block = self._blocks[blocks[-1]]
                    needs_new = current_block.used_tokens >= self.block_size

                if needs_new:
                    block = self._acquire_block()
                    block.used_tokens = 0
                    block.refcount = 1
                    block.request_owners.add(request_id)
                    blocks.append(block.block_id)
                    allocated.append(block.block_id)

                current_block = self._blocks[blocks[-1]]
                available = self.block_size - current_block.used_tokens
                write = min(available, remaining)
                current_block.used_tokens += write
                remaining -= write

            return allocated

    def release_request(self, request_id: str) -> None:
        with self._lock:
            block_ids = self._request_blocks.pop(request_id, [])
            for block_id in block_ids:
                block = self._blocks[block_id]
                if request_id not in block.request_owners:
                    continue
                block.request_owners.remove(request_id)
                block.refcount -= 1
                if block.refcount < 0:
                    raise RuntimeError(f"negative refcount on block {block_id}")
                self._release_block_if_unused(block)

    def get_request_blocks(self, request_id: str) -> List[int]:
        with self._lock:
            return list(self._request_blocks.get(request_id, []))

    def token_capacity(self) -> int:
        return self.total_blocks * self.block_size

    def allocated_tokens(self) -> int:
        with self._lock:
            return sum(block.used_tokens for block in self._blocks.values() if block.refcount > 0)

    def stats(self) -> dict:
        with self._lock:
            free_blocks = len(self._free)
            live_blocks = self.total_blocks - free_blocks
            total_live_capacity = live_blocks * self.block_size
            used_tokens = sum(
                block.used_tokens for block in self._blocks.values() if block.refcount > 0
            )
            internal_waste = max(total_live_capacity - used_tokens, 0)
            utilization = (
                used_tokens / total_live_capacity if total_live_capacity > 0 else 1.0
            )

            return {
                "total_blocks": self.total_blocks,
                "free_blocks": free_blocks,
                "live_blocks": live_blocks,
                "block_size": self.block_size,
                "used_tokens": used_tokens,
                "total_live_capacity_tokens": total_live_capacity,
                "internal_waste_tokens": internal_waste,
                "block_utilization": utilization,
                "memory_pressure": 1.0 - (free_blocks / self.total_blocks),
            }
