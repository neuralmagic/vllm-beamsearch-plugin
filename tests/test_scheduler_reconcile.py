from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from vllm_beam_search.scheduler import BeamSearchScheduler


@dataclass
class FakeBlock:
    block_id: int
    ref_cnt: int = 1


class FakeBlockPool:
    def touch(self, blocks: list[FakeBlock]) -> None:
        for block in blocks:
            block.ref_cnt += 1

    def free_blocks(self, blocks: Iterable[FakeBlock]) -> None:
        for block in blocks:
            block.ref_cnt -= 1


class FakeManager:
    def __init__(self, blocks: list[FakeBlock]) -> None:
        self.req_to_blocks = {"dst": blocks}
        self.num_cached_block = {"dst": len(blocks)}


def test_replace_request_blocks_preserves_async_suffix() -> None:
    # old dst: 1, 5, 9, 4, 8, [10]
    old_blocks = [FakeBlock(i) for i in [1, 5, 9, 4, 8, 10]]
    # new source prefix: 1, 5, 6, 2, 3
    shared_prefix = [FakeBlock(i) for i in [1, 5, 6, 2, 3]]
    mgr = FakeManager(old_blocks)

    BeamSearchScheduler._replace_request_blocks(
        mgr=mgr,
        dst_id="dst",
        shared_blocks=shared_prefix,
        new_blocks=list(shared_prefix),
        block_pool=FakeBlockPool(),
        prefix_blocks=len(shared_prefix),
    )

    assert [block.block_id for block in mgr.req_to_blocks["dst"]] == [
        1,
        5,
        6,
        2,
        3,
        10,
    ]
    assert [block.ref_cnt for block in old_blocks[:5]] == [0, 0, 0, 0, 0]
    assert old_blocks[5].ref_cnt == 1
    assert [block.ref_cnt for block in shared_prefix] == [2, 2, 2, 2, 2]


def test_snapshot_source_prefix_keeps_partial_cow_computed() -> None:
    blocks = [FakeBlock(i) for i in [10, 20]]
    mgr = FakeManager([])
    mgr.req_to_blocks["src"] = blocks
    scheduler = BeamSearchScheduler.__new__(BeamSearchScheduler)
    scheduler.block_size = 4

    snapshot = scheduler._snapshot_source_prefix(
        src_id="src",
        kv_prefix_len=5,
        self_idxs=[0],
        mgrs=[mgr],
    )

    assert [block.block_id for block in snapshot.blocks_by_manager[0]] == [10]
    assert snapshot.num_computed_tokens == 5
