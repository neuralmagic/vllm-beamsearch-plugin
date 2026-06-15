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
