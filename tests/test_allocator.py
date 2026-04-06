from cache.allocator import KVBlockAllocator


def test_allocator_append_and_release() -> None:
    alloc = KVBlockAllocator(total_blocks=8, block_size=4)

    alloc.append_tokens("r1", 3)
    stats = alloc.stats()
    assert stats["live_blocks"] == 1
    assert stats["used_tokens"] == 3

    alloc.append_tokens("r1", 5)
    blocks = alloc.get_request_blocks("r1")
    assert len(blocks) == 2

    stats = alloc.stats()
    assert stats["used_tokens"] == 8

    alloc.release_request("r1")
    stats = alloc.stats()
    assert stats["live_blocks"] == 0
    assert stats["free_blocks"] == 8


def test_allocator_pin_and_attach_refcounts() -> None:
    alloc = KVBlockAllocator(total_blocks=4, block_size=4)

    alloc.append_tokens("r1", 4)
    blocks = alloc.get_request_blocks("r1")

    alloc.pin_blocks("prefix:a", blocks)
    alloc.attach_blocks("r2", blocks)

    alloc.release_request("r1")
    stats_mid = alloc.stats()
    assert stats_mid["live_blocks"] == 1

    alloc.release_request("r2")
    stats_mid2 = alloc.stats()
    assert stats_mid2["live_blocks"] == 1

    alloc.unpin_blocks("prefix:a", blocks)
    stats_end = alloc.stats()
    assert stats_end["live_blocks"] == 0
