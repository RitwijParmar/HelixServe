from cache.prefix_cache import PrefixCache


def test_prefix_cache_insert_lookup_and_hit_rate() -> None:
    pinned = {}

    def pin(owner: str, block_ids: list[int]) -> None:
        pinned[owner] = list(block_ids)

    def unpin(owner: str, block_ids: list[int]) -> None:
        pinned.pop(owner, None)

    cache = PrefixCache(
        block_size=4,
        max_entries=16,
        max_cached_tokens=100,
        pin_blocks=pin,
        unpin_blocks=unpin,
        cacheable_prefix_lengths=[4, 8],
        min_prefix_tokens=2,
    )

    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    blocks = [10, 11]
    cache.insert(prompt, blocks, backend_state={"foo": 1})

    entry = cache.lookup([1, 2, 3, 4, 9])
    assert entry is not None
    assert entry.token_count == 4
    assert entry.backend_state == {"foo": 1}

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["lookups"] == 1
    assert stats["hit_rate"] == 1.0


def test_prefix_cache_lru_evict() -> None:
    active = {}

    def pin(owner: str, block_ids: list[int]) -> None:
        active[owner] = list(block_ids)

    def unpin(owner: str, block_ids: list[int]) -> None:
        active.pop(owner, None)

    cache = PrefixCache(
        block_size=2,
        max_entries=2,
        max_cached_tokens=100,
        pin_blocks=pin,
        unpin_blocks=unpin,
    )

    cache.insert([1, 2], [1])
    cache.insert([3, 4], [2])
    cache.insert([5, 6], [3])

    assert cache.stats()["entries"] <= 2
