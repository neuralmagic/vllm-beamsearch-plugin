from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_beam_search.beam_types import _INIT_NEG, BeamRuntime
from vllm_beam_search.mrv2_sampler import (
    AsyncBeamTransitions,
    BeamSearchMRV2Sampler,
    _BeamGroupGpuState,
    _BeamGroupStatePool,
    _BeamTransitionsGpu,
    _TransitionBufferPool,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MRV2 beam sampler rewrite tests require CUDA",
)

DEVICE = torch.device("cuda")


class FakeTensor:
    def __init__(self, gpu: torch.Tensor) -> None:
        self.gpu = gpu


class FakeReqStates:
    def __init__(self) -> None:
        self.all_token_ids = FakeTensor(
            torch.tensor(
                [
                    [0, 0, 31806, 1979, 75, 99],
                    [0, 0, 38, 222, 45, 88],
                ],
                dtype=torch.int32,
                device=DEVICE,
            )
        )
        self.total_len = FakeTensor(
            torch.tensor([6, 6], dtype=torch.int32, device=DEVICE)
        )
        self.num_computed_tokens = FakeTensor(
            torch.tensor([6, 6], dtype=torch.int32, device=DEVICE)
        )
        self.prefill_len = SimpleNamespace(
            np=np.array([2, 2], dtype=np.int32)
        )


class FakeBlockTables:
    def __init__(self) -> None:
        self.block_sizes = [1]
        self.block_tables = [
            FakeTensor(
                torch.tensor(
                    [
                        [1, 2, 265, 269, 273, 501],
                        [67, 68, 301, 302, 303, 601],
                    ],
                    dtype=torch.int32,
                    device=DEVICE,
                )
            )
        ]


def _sampler() -> BeamSearchMRV2Sampler:
    sampler = BeamSearchMRV2Sampler.__new__(BeamSearchMRV2Sampler)
    sampler.base_sampler = SimpleNamespace(req_states=FakeReqStates())
    sampler.block_tables = FakeBlockTables()
    sampler.self_attn_group_indices = (0,)
    sampler._pending_computed_resets = []
    sampler.req_to_group = {
        "gid:beam:0": ("gid", 0),
        "gid:beam:1": ("gid", 1),
    }
    sampler.groups = {
        "gid": BeamRuntime(
            beam_width=2,
            eos_token_id=2,
            no_repeat_ngram_size=3,
            prompt_tokens=[0, 0],
        )
    }
    return sampler


def _pooled_state(
    *,
    pool: _BeamGroupStatePool,
    tokens: torch.Tensor,
    cum: torch.Tensor,
    length: int,
    prompt_len: int = 0,
) -> tuple[_BeamGroupGpuState, torch.Tensor]:
    slot = pool.allocate()
    pool.tokens[slot].zero_()
    pool.tokens[slot, :, :tokens.shape[1]].copy_(tokens)
    pool.cum[slot].copy_(cum)
    state = _BeamGroupGpuState(
        pool=pool,
        slot=slot,
        length=length,
        prompt_len=prompt_len,
    )
    return state, pool.tokens[slot]


def _gpu_transitions(
    *,
    gids: tuple[str, ...],
    prefix_lens: tuple[int, ...],
    active_counts: tuple[int, ...],
    req_idx_by_group: torch.Tensor,
    dst_slots: torch.Tensor,
    src_slots: torch.Tensor,
    tokens: torch.Tensor,
) -> _BeamTransitionsGpu:
    group_count, beam_width = req_idx_by_group.shape
    return _BeamTransitionsGpu(
        gids=gids,
        steps=tuple(0 for _gid in gids),
        prefix_lens=prefix_lens,
        prompt_lens=tuple(0 for _gid in gids),
        active_counts=active_counts,
        req_idx_by_group=req_idx_by_group,
        dst_slots=dst_slots,
        src_slots=src_slots,
        fork_src=torch.empty(
            (group_count, beam_width),
            dtype=torch.long,
            device=DEVICE,
        ),
        active_mask=torch.empty(
            (group_count, beam_width),
            dtype=torch.bool,
            device=DEVICE,
        ),
        tokens=tokens,
        cum=torch.zeros(
            (group_count, beam_width),
            dtype=torch.float32,
            device=DEVICE,
        ),
    )


def test_gpu_worker_rewrite_uses_worker_prefix_and_preserves_suffix() -> None:
    sampler = _sampler()
    # Simulate async CPU/worker token lag: the physical source row has not
    # caught up to the sampler's real-time token truth yet.
    sampler.base_sampler.req_states.all_token_ids.gpu[0, :5] = torch.tensor(
        [0, 0, 111, 112, 113],
        dtype=torch.int32,
        device=DEVICE,
    )
    transitions = _gpu_transitions(
        gids=("gid",),
        # Immediate MRV2 sampler transitions use full worker prefix length:
        # decoder prompt [0, 0] + generated [31806, 1979, 75].
        prefix_lens=(5,),
        active_counts=(1,),
        req_idx_by_group=torch.tensor([[0, 1]], dtype=torch.long, device=DEVICE),
        dst_slots=torch.tensor([[1, 0]], dtype=torch.long, device=DEVICE),
        src_slots=torch.tensor([[0, 0]], dtype=torch.long, device=DEVICE),
        tokens=torch.tensor(
            [[[0, 0, 31806, 1979, 75, 99],
              [0, 0, 31806, 1979, 75, 88]]],
            dtype=torch.int32,
            device=DEVICE,
        ),
    )
    sampler._apply_gpu_worker_rewrites(transitions)

    req_states = sampler.base_sampler.req_states
    assert req_states.all_token_ids.gpu.cpu().tolist() == [
        [0, 0, 111, 112, 113, 99],
        [0, 0, 31806, 1979, 75, 88],
    ]
    assert req_states.total_len.gpu.cpu().tolist() == [6, 5]
    assert req_states.num_computed_tokens.gpu.cpu().tolist() == [6, 5]

    block_table = sampler.block_tables.block_tables[0].gpu
    assert block_table.cpu().tolist() == [
        [1, 2, 265, 269, 273, 501],
        [1, 2, 265, 269, 273, 601],
    ]


def test_gpu_worker_rewrite_resets_zero_prefix_rows() -> None:
    sampler = _sampler()
    transitions = _gpu_transitions(
        gids=("gid",),
        prefix_lens=(0,),
        active_counts=(2,),
        req_idx_by_group=torch.tensor([[0, 1]], dtype=torch.long, device=DEVICE),
        dst_slots=torch.tensor([[0, 1]], dtype=torch.long, device=DEVICE),
        src_slots=torch.tensor([[0, 0]], dtype=torch.long, device=DEVICE),
        tokens=torch.zeros((1, 2, 1), dtype=torch.int32, device=DEVICE),
    )
    sampler._apply_gpu_worker_rewrites(transitions)

    req_states = sampler.base_sampler.req_states
    assert req_states.total_len.gpu.cpu().tolist() == [0, 0]
    assert req_states.num_computed_tokens.gpu.cpu().tolist() == [0, 0]
    assert sampler.block_tables.block_tables[0].gpu.cpu().tolist() == [
        [1, 2, 265, 269, 273, 501],
        [67, 68, 301, 302, 303, 601],
    ]


def test_gpu_worker_rewrite_batches_same_prefix_groups() -> None:
    sampler = BeamSearchMRV2Sampler.__new__(BeamSearchMRV2Sampler)
    sampler.base_sampler = SimpleNamespace(
        req_states=SimpleNamespace(
            all_token_ids=FakeTensor(
                torch.tensor(
                    [
                        [0, 0, 10, 100],
                        [0, 0, 11, 101],
                        [0, 0, 20, 102],
                        [0, 0, 21, 103],
                    ],
                    dtype=torch.int32,
                    device=DEVICE,
                )
            ),
            total_len=FakeTensor(
                torch.tensor([4, 4, 4, 4], dtype=torch.int32, device=DEVICE)
            ),
            num_computed_tokens=FakeTensor(
                torch.tensor([4, 4, 4, 4], dtype=torch.int32, device=DEVICE)
            ),
        )
    )
    sampler.block_tables = SimpleNamespace(
        block_sizes=[1],
        block_tables=[
            FakeTensor(
                torch.tensor(
                    [
                        [1, 2, 10, 100],
                        [4, 5, 11, 101],
                        [7, 8, 20, 102],
                        [9, 10, 21, 103],
                    ],
                    dtype=torch.int32,
                    device=DEVICE,
                )
            )
        ],
    )
    sampler.self_attn_group_indices = (0,)
    sampler._pending_computed_resets = []
    sampler.req_to_group = {
        "g0:beam:0": ("g0", 0),
        "g0:beam:1": ("g0", 1),
        "g1:beam:0": ("g1", 0),
        "g1:beam:1": ("g1", 1),
    }
    sampler.groups = {
        gid: BeamRuntime(
            beam_width=2,
            eos_token_id=None,
            no_repeat_ngram_size=0,
            prompt_tokens=[],
        )
        for gid in ("g0", "g1")
    }
    transitions = _gpu_transitions(
        gids=("g0", "g1"),
        prefix_lens=(3, 3),
        active_counts=(1, 1),
        req_idx_by_group=torch.tensor(
            [[0, 1], [2, 3]],
            dtype=torch.long,
            device=DEVICE,
        ),
        dst_slots=torch.tensor(
            [[1, 0], [1, 0]],
            dtype=torch.long,
            device=DEVICE,
        ),
        src_slots=torch.tensor(
            [[0, 0], [0, 0]],
            dtype=torch.long,
            device=DEVICE,
        ),
        tokens=torch.tensor(
            [
                [[0, 0, 10, 100], [0, 0, 10, 101]],
                [[0, 0, 20, 102], [0, 0, 20, 103]],
            ],
            dtype=torch.int32,
            device=DEVICE,
        ),
    )

    sampler._apply_gpu_worker_rewrites(transitions)

    req_states = sampler.base_sampler.req_states
    assert req_states.all_token_ids.gpu.cpu().tolist() == [
        [0, 0, 10, 100],
        [0, 0, 10, 101],
        [0, 0, 20, 102],
        [0, 0, 20, 103],
    ]
    assert req_states.total_len.gpu.cpu().tolist() == [4, 3, 4, 3]
    assert req_states.num_computed_tokens.gpu.cpu().tolist() == [4, 3, 4, 3]
    assert sampler.block_tables.block_tables[0].gpu.cpu().tolist() == [
        [1, 2, 10, 100],
        [1, 2, 10, 101],
        [7, 8, 20, 102],
        [7, 8, 20, 103],
    ]
    req_states.num_computed_tokens.gpu[:] = torch.tensor(
        [4, 4, 4, 4],
        dtype=torch.int32,
        device=DEVICE,
    )
    sampler._apply_pending_computed_resets()
    assert req_states.num_computed_tokens.gpu.cpu().tolist() == [4, 3, 4, 3]


def test_gpu_worker_rewrite_copies_partial_kv_block() -> None:
    sampler = BeamSearchMRV2Sampler.__new__(BeamSearchMRV2Sampler)
    sampler.base_sampler = SimpleNamespace(
        req_states=SimpleNamespace(
            all_token_ids=FakeTensor(
                torch.zeros((2, 8), dtype=torch.int32, device=DEVICE)
            ),
            total_len=FakeTensor(torch.zeros(2, dtype=torch.int32, device=DEVICE)),
            num_computed_tokens=FakeTensor(
                torch.zeros(2, dtype=torch.int32, device=DEVICE)
            ),
        )
    )
    sampler.block_tables = SimpleNamespace(
        block_sizes=[4],
        block_tables=[
            FakeTensor(
                torch.tensor(
                    [
                        [10, 20, 0],
                        [30, 40, 0],
                    ],
                    dtype=torch.int32,
                    device=DEVICE,
                )
            )
        ],
    )
    sampler.self_attn_group_indices = (0,)
    sampler._pending_computed_resets = []
    sampler.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["layer0"])]
    )
    kv_cache = torch.zeros((64, 2, 4, 1, 2), dtype=torch.float16, device=DEVICE)
    kv_cache[20] = torch.arange(
        kv_cache[20].numel(),
        dtype=torch.float16,
        device=DEVICE,
    ).reshape_as(kv_cache[20])
    sampler.kv_cache_forward_context = {
        "layer0": SimpleNamespace(kv_cache=kv_cache),
    }

    req_idx_by_group = torch.tensor([[0, 1]], dtype=torch.long, device=DEVICE)
    tokens = torch.tensor(
        [[[1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 6, 0]]],
        dtype=torch.int32,
        device=DEVICE,
    )
    transitions = _BeamTransitionsGpu(
        gids=("gid",),
        steps=(0,),
        prefix_lens=(5,),
        prompt_lens=(0,),
        active_counts=(1,),
        req_idx_by_group=req_idx_by_group,
        dst_slots=torch.tensor([[1, 0]], dtype=torch.long, device=DEVICE),
        src_slots=torch.tensor([[0, 0]], dtype=torch.long, device=DEVICE),
        fork_src=torch.empty((1, 2), dtype=torch.long, device=DEVICE),
        active_mask=torch.empty((1, 2), dtype=torch.bool, device=DEVICE),
        tokens=tokens,
        cum=torch.zeros((1, 2), dtype=torch.float32, device=DEVICE),
    )

    sampler._apply_gpu_worker_rewrites(transitions)

    assert sampler.block_tables.block_tables[0].gpu.cpu().tolist() == [
        [10, 20, 0],
        [10, 40, 0],
    ]
    assert torch.equal(kv_cache[40].cpu(), kv_cache[20].cpu())
    assert sampler.base_sampler.req_states.total_len.gpu.cpu().tolist() == [0, 5]
    assert sampler.base_sampler.req_states.num_computed_tokens.gpu.cpu().tolist() == [
        0,
        5,
    ]


def test_pending_computed_reset_restores_prefix_after_postprocess() -> None:
    sampler = _sampler()
    sampler._pending_computed_resets.append((
        torch.tensor([0, 1], dtype=torch.long, device=DEVICE),
        torch.tensor([0, 0], dtype=torch.long, device=DEVICE),
    ))
    req_states = sampler.base_sampler.req_states
    req_states.num_computed_tokens.gpu[:] = torch.tensor(
        [1, 1], dtype=torch.int32, device=DEVICE
    )

    sampler._apply_pending_computed_resets()

    assert req_states.num_computed_tokens.gpu.cpu().tolist() == [0, 0]
    assert sampler._pending_computed_resets == []


def test_select_masks_padded_inactive_beams() -> None:
    sampler = BeamSearchMRV2Sampler.__new__(BeamSearchMRV2Sampler)
    sampler.req_to_group = {
        "gid:beam:0": ("gid", 0),
        "gid:beam:2": ("gid", 2),
    }
    sampler.groups = {
        "gid": BeamRuntime(
            beam_width=4,
            eos_token_id=None,
            no_repeat_ngram_size=0,
            prompt_tokens=[],
        )
    }
    tokens_input = torch.tensor(
        [
            [0, 10, 0, 0],
            [0, 11, 0, 0],
            [0, 20, 0, 0],
            [0, 21, 0, 0],
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    pool = _BeamGroupStatePool(
        beam_width=4,
        max_len=4,
        device=DEVICE,
        capacity=2,
    )
    cum = torch.tensor(
        [0.0, _INIT_NEG, 0.0, _INIT_NEG],
        dtype=torch.float32,
        device=DEVICE,
    )
    state, tokens = _pooled_state(
        pool=pool,
        tokens=tokens_input,
        cum=cum,
        length=2,
    )
    sampler._gpu_group_state = {
        "gid": state,
    }
    logits = torch.full((2, 12), -100.0, dtype=torch.float32, device=DEVICE)
    logits[0, 5] = 10.0
    logits[0, 6] = 9.0
    logits[1, 7] = 11.0
    logits[1, 8] = 8.0
    sampled = torch.zeros(2, dtype=torch.int64, device=DEVICE)

    transitions = sampler._gpu_select_groups(
        processed_logits=logits,
        input_batch=SimpleNamespace(
            req_ids=["gid:beam:0", "gid:beam:2"],
            cu_num_logits_np=np.array([0, 1, 2], dtype=np.int32),
        ),
        sampled=sampled,
    )

    assert transitions is not None
    group_idx = transitions.gids.index("gid")
    active_count = transitions.active_counts[group_idx]
    assert transitions.dst_slots[group_idx, :active_count].cpu().tolist() == [0, 2]
    assert transitions.src_slots[group_idx, :active_count].cpu().tolist() == [2, 0]
    assert transitions.active_mask[group_idx].cpu().tolist() == [
        True,
        False,
        True,
        False,
    ]
    assert sampled.cpu().tolist() == [7, 5]
    assert tokens[:, :3].cpu().tolist() == [
        [0, 20, 7],
        [0, 11, 0],
        [0, 10, 5],
        [0, 21, 0],
    ]


def test_beam_view_reorders_interleaved_requests() -> None:
    sampler = BeamSearchMRV2Sampler.__new__(BeamSearchMRV2Sampler)
    sampler.req_to_group = {
        "g0:beam:0": ("g0", 0),
        "g0:beam:1": ("g0", 1),
        "g1:beam:0": ("g1", 0),
        "g1:beam:1": ("g1", 1),
    }
    sampler.groups = {
        gid: BeamRuntime(
            beam_width=2,
            eos_token_id=None,
            no_repeat_ngram_size=0,
            prompt_tokens=[],
        )
        for gid in ("g0", "g1")
    }

    view = sampler._beam_view_by_group(
        SimpleNamespace(
            req_ids=[
                "g0:beam:0",
                "g1:beam:0",
                "g0:beam:1",
                "g1:beam:1",
            ],
            cu_num_logits_np=np.array([0, 1, 2, 3, 4], dtype=np.int32),
            idx_mapping_np=np.array([10, 11, 12, 13], dtype=np.int32),
        ),
        num_logits=4,
    )

    assert view.gids == ("g0", "g1")
    assert view.beam_indices_cpu.tolist() == [[0, 1], [0, 1]]
    assert view.logit_indices_cpu.tolist() == [[0, 2], [1, 3]]
    assert view.slot_to_batch_cpu.tolist() == [[0, 2], [1, 3]]
    assert view.req_idx_cpu.tolist() == [[10, 12], [11, 13]]
    assert view.valid_beam_mask_cpu.tolist() == [
        [True, True],
        [True, True],
    ]
    assert view.active_slot_counts == (2, 2)


def test_select_handles_mixed_lengths_in_one_bucket() -> None:
    sampler = BeamSearchMRV2Sampler.__new__(BeamSearchMRV2Sampler)
    sampler.req_to_group = {
        "g0:beam:0": ("g0", 0),
        "g0:beam:1": ("g0", 1),
        "g1:beam:0": ("g1", 0),
        "g1:beam:1": ("g1", 1),
    }
    sampler.groups = {
        gid: BeamRuntime(
            beam_width=2,
            eos_token_id=None,
            no_repeat_ngram_size=2,
            prompt_tokens=[],
        )
        for gid in ("g0", "g1")
    }
    tokens_g0_input = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=torch.int32,
        device=DEVICE,
    )
    tokens_g1_input = torch.tensor(
        [[0, 7, 0, 0], [0, 4, 4, 0]],
        dtype=torch.int32,
        device=DEVICE,
    )
    pool = _BeamGroupStatePool(
        beam_width=2,
        max_len=4,
        device=DEVICE,
        capacity=4,
    )
    state_g0, tokens_g0 = _pooled_state(
        pool=pool,
        tokens=tokens_g0_input,
        cum=torch.tensor(
            [0.0, _INIT_NEG],
            dtype=torch.float32,
            device=DEVICE,
        ),
        length=1,
        prompt_len=1,
    )
    state_g1, tokens_g1 = _pooled_state(
        pool=pool,
        tokens=tokens_g1_input,
        cum=torch.tensor(
            [0.0, _INIT_NEG],
            dtype=torch.float32,
            device=DEVICE,
        ),
        length=3,
        prompt_len=1,
    )
    sampler._gpu_group_state = {
        "g0": state_g0,
        "g1": state_g1,
    }
    logits = torch.full((4, 12), -100.0, dtype=torch.float32, device=DEVICE)
    logits[0, 5] = 10.0
    logits[0, 6] = 9.0
    logits[2, 7] = 10.0
    logits[2, 9] = 8.0
    logits[2, 10] = 7.0
    sampled = torch.zeros(4, dtype=torch.int64, device=DEVICE)

    transitions = sampler._gpu_select_groups(
        processed_logits=logits,
        input_batch=SimpleNamespace(
            req_ids=[
                "g0:beam:0",
                "g0:beam:1",
                "g1:beam:0",
                "g1:beam:1",
            ],
            cu_num_logits_np=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        ),
        sampled=sampled,
    )

    assert transitions is not None
    assert set(transitions.gids) == {"g0", "g1"}
    assert sampled.cpu().tolist() == [5, 6, 9, 10]
    assert tokens_g0[:, :2].cpu().tolist() == [[0, 5], [0, 6]]
    assert tokens_g1[:, :4].cpu().tolist() == [
        [0, 7, 0, 9],
        [0, 7, 0, 10],
    ]

    staged = AsyncBeamTransitions(transitions).to_cpu_nonblocking()
    torch.cuda.synchronize()
    by_gid = {
        transition.group_id: transition
        for transition in staged.to_output()
    }
    assert by_gid["g0"].tokens == ((5,), (6,))
    assert by_gid["g1"].tokens == ((7, 0, 9), (7, 0, 10))


def test_select_materializes_eos_completions() -> None:
    sampler = BeamSearchMRV2Sampler.__new__(BeamSearchMRV2Sampler)
    sampler.req_to_group = {
        "g0:beam:0": ("g0", 0),
        "g0:beam:1": ("g0", 1),
        "g1:beam:0": ("g1", 0),
        "g1:beam:1": ("g1", 1),
    }
    sampler.groups = {
        gid: BeamRuntime(
            beam_width=2,
            eos_token_id=2,
            no_repeat_ngram_size=0,
            prompt_tokens=[],
        )
        for gid in ("g0", "g1")
    }
    pool = _BeamGroupStatePool(
        beam_width=2,
        max_len=5,
        device=DEVICE,
        capacity=4,
    )
    state_g0, _tokens_g0 = _pooled_state(
        pool=pool,
        tokens=torch.tensor(
            [[101, 11, 12, 0, 0], [101, 13, 14, 0, 0]],
            dtype=torch.int32,
            device=DEVICE,
        ),
        cum=torch.tensor([0.0, _INIT_NEG], dtype=torch.float32, device=DEVICE),
        length=3,
        prompt_len=1,
    )
    state_g1, _tokens_g1 = _pooled_state(
        pool=pool,
        tokens=torch.tensor(
            [[201, 21, 22, 0, 0], [201, 23, 24, 0, 0]],
            dtype=torch.int32,
            device=DEVICE,
        ),
        cum=torch.tensor([0.0, _INIT_NEG], dtype=torch.float32, device=DEVICE),
        length=3,
        prompt_len=1,
    )
    sampler._gpu_group_state = {
        "g0": state_g0,
        "g1": state_g1,
    }
    logits = torch.full((4, 12), -100.0, dtype=torch.float32, device=DEVICE)
    logits[0, 2] = 10.0
    logits[0, 5] = 9.0
    logits[0, 6] = 8.0
    logits[2, 2] = 11.0
    logits[2, 7] = 10.0
    logits[2, 8] = 9.0
    sampled = torch.zeros(4, dtype=torch.int64, device=DEVICE)

    transitions = sampler._gpu_select_groups(
        processed_logits=logits,
        input_batch=SimpleNamespace(
            req_ids=[
                "g0:beam:0",
                "g0:beam:1",
                "g1:beam:0",
                "g1:beam:1",
            ],
            cu_num_logits_np=np.array([0, 1, 2, 3, 4], dtype=np.int32),
        ),
        sampled=sampled,
    )

    assert transitions is not None
    g0_idx = transitions.gids.index("g0")
    g1_idx = transitions.gids.index("g1")
    assert transitions.completion_prefixes[g0_idx, 0, 1:3].cpu().tolist() == [
        11,
        12,
    ]
    assert transitions.completion_prefixes[g1_idx, 0, 1:3].cpu().tolist() == [
        21,
        22,
    ]
    assert transitions.completion_tokens[g0_idx, 0].item() == 2
    assert transitions.completion_tokens[g1_idx, 0].item() == 2
    assert transitions.completion_scores[g0_idx, 0].item() > _INIT_NEG / 2
    assert transitions.completion_scores[g1_idx, 0].item() > _INIT_NEG / 2

    staged = AsyncBeamTransitions(transitions).to_cpu_nonblocking()
    torch.cuda.synchronize()
    by_gid = {
        transition.group_id: transition
        for transition in staged.to_output()
    }
    assert by_gid["g0"].completions[0][0] == (11, 12, 2)
    assert by_gid["g1"].completions[0][0] == (21, 22, 2)


def test_async_beam_transitions_use_persistent_gpu_slot_buffer() -> None:
    pool = _TransitionBufferPool(num_slots=2)
    gpu_transitions = _BeamTransitionsGpu(
        gids=("g0", "g1"),
        steps=(3, 4),
        prefix_lens=(4, 4),
        prompt_lens=(1, 1),
        active_counts=(2, 2),
        dst_slots=torch.tensor(
            [[0, 1], [0, 1]],
            dtype=torch.long,
            device=DEVICE,
        ),
        src_slots=torch.tensor(
            [[1, 0], [0, 1]],
            dtype=torch.long,
            device=DEVICE,
        ),
        fork_src=torch.tensor(
            [[1, 0], [0, 1]],
            dtype=torch.long,
            device=DEVICE,
        ),
        active_mask=torch.tensor(
            [[True, True], [True, True]],
            device=DEVICE,
        ),
        tokens=torch.tensor(
            [
                [[0, 11, 12, 13], [0, 21, 22, 23]],
                [[0, 31, 32, 33], [0, 41, 42, 43]],
            ],
            dtype=torch.int32,
            device=DEVICE,
        ),
        cum=torch.tensor(
            [[0.1, 0.2], [0.3, 0.4]],
            dtype=torch.float32,
            device=DEVICE,
        ),
    )
    transitions = AsyncBeamTransitions(gpu_transitions, buffer_pool=pool)

    staged = transitions.to_cpu_nonblocking()
    torch.cuda.synchronize()
    assert staged.tensors is not None
    assert staged.tensors["tokens"].shape == (2, 2, 4)
    token_buffers = [
        buffer
        for key, buffer in pool._buffers.items()
        if key[0] == "tokens"
    ]
    assert len(token_buffers) == 1
    token_buffer_ptr = token_buffers[0].data_ptr()

    out = staged.to_output()
    assert [transition.group_id for transition in out] == ["g0", "g1"]
    assert out[0].tokens == ((11, 12, 13), (21, 22, 23))
    assert out[1].tokens == ((31, 32, 33), (41, 42, 43))

    staged_again = AsyncBeamTransitions(
        gpu_transitions,
        buffer_pool=pool,
    ).to_cpu_nonblocking()
    torch.cuda.synchronize()
    assert staged_again.tensors is not None
    token_buffers = [
        buffer
        for key, buffer in pool._buffers.items()
        if key[0] == "tokens"
    ]
    assert token_buffers[0].data_ptr() == token_buffer_ptr
