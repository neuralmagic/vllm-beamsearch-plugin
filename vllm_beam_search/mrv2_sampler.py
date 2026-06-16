"""MRV2 sampler wrapper for in-flight beam search."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import torch
import triton
import triton.language as tl

from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from .beam_types import _INIT_NEG, BeamRuntime, BeamTransition

BEAM_TRANSITIONS_OUTPUT = "vllm_beam_search.transitions"
_SYNC_CHECK = os.getenv("VLLM_BEAM_GPU_SYNC_CHECK")
_NEG_INF = float("-inf")
_GroupStateEntry = tuple[str, "_BeamGroupGpuState"]
_SelectKey = tuple[int, int, int | None]
_BeamRequest = tuple[int, str, int, BeamRuntime]
_BEAM_IDX = 0
_LOGIT_IDX = 1
_INPUT_BATCH_IDX = 2
_REQ_IDX = 3
_NUM_SLOT_INDEX_FIELDS = 4
_BEAM_STATE_BLOCK_N = 256


@dataclass
class BeamSamplerOutput(SamplerOutput):
    async_outputs: dict[str, Any] | None = None


@dataclass
class _BeamView:
    """CPU view of scheduler requests grouped by beam parent."""

    gids: tuple[str, ...]
    key: _SelectKey | None
    active_slot_counts: tuple[int, ...]
    # Shape: [field, group, slot], where field is one of the constants above.
    beam_metadata_cpu: torch.Tensor
    valid_beam_mask_cpu: torch.Tensor

    @property
    def beam_indices_cpu(self) -> torch.Tensor:
        return self.beam_metadata_cpu[_BEAM_IDX]

    @property
    def logit_indices_cpu(self) -> torch.Tensor:
        return self.beam_metadata_cpu[_LOGIT_IDX]

    @property
    def slot_to_batch_cpu(self) -> torch.Tensor:
        return self.beam_metadata_cpu[_INPUT_BATCH_IDX]

    @property
    def req_idx_cpu(self) -> torch.Tensor:
        return self.beam_metadata_cpu[_REQ_IDX]


def _beam_select_key(gr: BeamRuntime) -> _SelectKey:
    return (gr.beam_width, int(gr.no_repeat_ngram_size), gr.eos_token_id)


def _empty_beam_view() -> _BeamView:
    return _BeamView(
        gids=(),
        key=None,
        active_slot_counts=(),
        beam_metadata_cpu=torch.empty(
            (_NUM_SLOT_INDEX_FIELDS, 0, 0),
            dtype=torch.long,
            pin_memory=True,
        ),
        valid_beam_mask_cpu=torch.empty(
            (0, 0),
            dtype=torch.bool,
            pin_memory=True,
        ),
    )


@triton.jit
def _no_repeat_ngram_mask_kernel(
    tokens,
    logprobs,
    present_slots,
    state_slots,
    lengths,
    ngram: tl.constexpr,
    beam_width: tl.constexpr,
    token_stride0,
    token_stride1,
    token_stride2,
    logprobs_stride0,
    logprobs_stride1,
    logprobs_stride2,
) -> None:
    group = tl.program_id(0)
    row = tl.program_id(1)
    start = tl.program_id(2)
    slot = tl.load(present_slots + group * beam_width + row)
    state_slot = tl.load(state_slots + group)
    group_length = tl.load(lengths + group)
    valid_start = start < group_length - (ngram - 1)

    match = valid_start
    for offset in tl.static_range(0, ngram - 1):
        prev = tl.load(
            tokens
            + state_slot * token_stride0
            + slot * token_stride1
            + (start + offset) * token_stride2,
            mask=valid_start,
            other=0,
        )
        suffix = tl.load(
            tokens
            + state_slot * token_stride0
            + slot * token_stride1
            + (group_length - (ngram - 1) + offset) * token_stride2,
            mask=valid_start,
            other=0,
        )
        match = match & (prev == suffix)

    banned = tl.load(
        tokens
        + state_slot * token_stride0
        + slot * token_stride1
        + (start + ngram - 1) * token_stride2,
        mask=valid_start,
        other=0,
    )
    tl.store(
        logprobs
        + group * logprobs_stride0
        + row * logprobs_stride1
        + banned * logprobs_stride2,
        -float("inf"),
        mask=match & valid_start,
    )


@triton.jit
def _snapshot_beam_prefix_kernel(
    token_pool,
    token_prefix,
    src_slots,
    valid_beam_mask,
    state_slots,
    lengths,
    beam_width: tl.constexpr,
    pool_stride0,
    pool_stride1,
    pool_stride2,
    prefix_stride0,
    prefix_stride1,
    prefix_stride2,
    src_stride0,
    src_stride1,
    BLOCK_N: tl.constexpr,
) -> None:
    group = tl.program_id(0)
    row = tl.program_id(1)
    col_block = tl.program_id(2)
    offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    state_slot = tl.load(state_slots + group)
    length = tl.load(lengths + group)
    src_slot = tl.load(src_slots + group * src_stride0 + row * src_stride1)
    valid = tl.load(valid_beam_mask + group * beam_width + row)

    prefix_mask = offsets < length
    prefix_vals = tl.load(
        token_pool
        + state_slot * pool_stride0
        + src_slot * pool_stride1
        + offsets * pool_stride2,
        mask=prefix_mask,
        other=0,
    )
    tl.store(
        token_prefix
        + group * prefix_stride0
        + src_slot * prefix_stride1
        + offsets * prefix_stride2,
        prefix_vals,
        mask=valid & prefix_mask,
    )


@triton.jit
def _init_beam_transition_kernel(
    fork_src,
    active_mask,
    beam_width: tl.constexpr,
    fork_stride0,
    fork_stride1,
    active_stride0,
    active_stride1,
) -> None:
    group = tl.program_id(0)
    row = tl.program_id(1)
    tl.store(fork_src + group * fork_stride0 + row * fork_stride1, row)
    tl.store(active_mask + group * active_stride0 + row * active_stride1, False)


@triton.jit
def _snapshot_transition_state_kernel(
    token_pool,
    cum_pool,
    tokens_out,
    cum_out,
    state_slots,
    lengths,
    beam_width: tl.constexpr,
    pool_stride0,
    pool_stride1,
    pool_stride2,
    cum_pool_stride0,
    cum_pool_stride1,
    out_stride0,
    out_stride1,
    out_stride2,
    cum_out_stride0,
    cum_out_stride1,
    BLOCK_N: tl.constexpr,
) -> None:
    group = tl.program_id(0)
    row = tl.program_id(1)
    col_block = tl.program_id(2)
    offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    state_slot = tl.load(state_slots + group)
    length = tl.load(lengths + group) + 1
    mask = offsets < length
    values = tl.load(
        token_pool
        + state_slot * pool_stride0
        + row * pool_stride1
        + offsets * pool_stride2,
        mask=mask,
        other=0,
    )
    tl.store(
        tokens_out
        + group * out_stride0
        + row * out_stride1
        + offsets * out_stride2,
        values,
        mask=mask,
    )
    if col_block == 0:
        score = tl.load(
            cum_pool + state_slot * cum_pool_stride0 + row * cum_pool_stride1
        )
        tl.store(
            cum_out + group * cum_out_stride0 + row * cum_out_stride1,
            score,
        )


@triton.jit
def _set_num_computed_kernel(
    num_computed,
    req_indices,
    prefix_lens,
    n_items,
    req_stride0,
    lens_stride0,
    BLOCK_N: tl.constexpr,
) -> None:
    block = tl.program_id(0)
    offsets = block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n_items
    req_idx = tl.load(req_indices + offsets * req_stride0, mask=mask, other=-1)
    prefix_len = tl.load(prefix_lens + offsets * lens_stride0, mask=mask, other=0)
    tl.store(num_computed + req_idx, prefix_len, mask=mask & (req_idx >= 0))


@triton.jit
def _rewrite_worker_tokens_kernel(
    transition_tokens,
    req_tokens,
    total_len,
    num_computed,
    req_idx_by_group,
    dst_slots,
    prefix_lens,
    active_counts,
    pending_req_idx,
    pending_prefix_lens,
    beam_width: tl.constexpr,
    trans_stride0,
    trans_stride1,
    trans_stride2,
    req_token_stride0,
    req_token_stride1,
    req_idx_stride0,
    req_idx_stride1,
    dst_stride0,
    dst_stride1,
    pending_stride0,
    pending_stride1,
    BLOCK_N: tl.constexpr,
) -> None:
    group = tl.program_id(0)
    row = tl.program_id(1)
    col_block = tl.program_id(2)
    offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    active_count = tl.load(active_counts + group)
    valid = row < active_count
    dst_slot = tl.load(dst_slots + group * dst_stride0 + row * dst_stride1)
    req_idx = tl.load(
        req_idx_by_group + group * req_idx_stride0 + dst_slot * req_idx_stride1,
        mask=valid,
        other=-1,
    )
    prefix_len = tl.load(prefix_lens + group)
    mask = valid & (req_idx >= 0) & (offsets < prefix_len)
    values = tl.load(
        transition_tokens
        + group * trans_stride0
        + dst_slot * trans_stride1
        + offsets * trans_stride2,
        mask=mask,
        other=0,
    )
    tl.store(
        req_tokens + req_idx * req_token_stride0 + offsets * req_token_stride1,
        values,
        mask=mask,
    )
    if col_block == 0:
        tl.store(total_len + req_idx, prefix_len, mask=valid & (req_idx >= 0))
        tl.store(num_computed + req_idx, prefix_len, mask=valid & (req_idx >= 0))
        tl.store(
            pending_req_idx + group * pending_stride0 + row * pending_stride1,
            req_idx,
        )
        tl.store(
            pending_prefix_lens + group * pending_stride0 + row * pending_stride1,
            prefix_len,
        )


@triton.jit
def _snapshot_worker_block_prefix_kernel(
    table,
    block_prefix,
    req_idx_by_group,
    src_slots,
    prefix_lens,
    active_counts,
    block_size: tl.constexpr,
    beam_width: tl.constexpr,
    table_stride0,
    table_stride1,
    prefix_stride0,
    prefix_stride1,
    prefix_stride2,
    req_idx_stride0,
    req_idx_stride1,
    src_stride0,
    src_stride1,
    BLOCK_N: tl.constexpr,
) -> None:
    group = tl.program_id(0)
    row = tl.program_id(1)
    col_block = tl.program_id(2)
    offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    active_count = tl.load(active_counts + group)
    valid = row < active_count
    src_slot = tl.load(src_slots + group * src_stride0 + row * src_stride1)
    req_idx = tl.load(
        req_idx_by_group + group * req_idx_stride0 + src_slot * req_idx_stride1,
        mask=valid,
        other=-1,
    )
    prefix_blocks = tl.load(prefix_lens + group) // block_size
    mask = valid & (req_idx >= 0) & (offsets < prefix_blocks)
    values = tl.load(
        table + req_idx * table_stride0 + offsets * table_stride1,
        mask=mask,
        other=0,
    )
    tl.store(
        block_prefix
        + group * prefix_stride0
        + row * prefix_stride1
        + offsets * prefix_stride2,
        values,
        mask=mask,
    )


@triton.jit
def _rewrite_worker_block_prefix_kernel(
    table,
    block_prefix,
    req_idx_by_group,
    dst_slots,
    prefix_lens,
    active_counts,
    block_size: tl.constexpr,
    beam_width: tl.constexpr,
    table_stride0,
    table_stride1,
    prefix_stride0,
    prefix_stride1,
    prefix_stride2,
    req_idx_stride0,
    req_idx_stride1,
    dst_stride0,
    dst_stride1,
    BLOCK_N: tl.constexpr,
) -> None:
    group = tl.program_id(0)
    row = tl.program_id(1)
    col_block = tl.program_id(2)
    offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    active_count = tl.load(active_counts + group)
    valid = row < active_count
    dst_slot = tl.load(dst_slots + group * dst_stride0 + row * dst_stride1)
    req_idx = tl.load(
        req_idx_by_group + group * req_idx_stride0 + dst_slot * req_idx_stride1,
        mask=valid,
        other=-1,
    )
    prefix_blocks = tl.load(prefix_lens + group) // block_size
    mask = valid & (req_idx >= 0) & (offsets < prefix_blocks)
    values = tl.load(
        block_prefix
        + group * prefix_stride0
        + row * prefix_stride1
        + offsets * prefix_stride2,
        mask=mask,
        other=0,
    )
    tl.store(
        table + req_idx * table_stride0 + offsets * table_stride1,
        values,
        mask=mask,
    )


@triton.jit
def _partial_block_ids_kernel(
    table,
    req_idx_by_group,
    src_slots,
    dst_slots,
    prefix_lens,
    active_counts,
    src_block_ids,
    dst_block_ids,
    block_size: tl.constexpr,
    beam_width: tl.constexpr,
    table_stride0,
    table_stride1,
    req_idx_stride0,
    req_idx_stride1,
    src_stride0,
    src_stride1,
    dst_stride0,
    dst_stride1,
) -> None:
    group = tl.program_id(0)
    row = tl.program_id(1)
    out_idx = group * beam_width + row
    tl.store(src_block_ids + out_idx, -1)
    tl.store(dst_block_ids + out_idx, -1)

    active_count = tl.load(active_counts + group)
    prefix_len = tl.load(prefix_lens + group)
    has_partial = prefix_len % block_size != 0
    valid = (row < active_count) & has_partial
    src_slot = tl.load(
        src_slots + group * src_stride0 + row * src_stride1,
        mask=valid,
        other=-1,
    )
    dst_slot = tl.load(
        dst_slots + group * dst_stride0 + row * dst_stride1,
        mask=valid,
        other=-1,
    )
    valid = valid & (src_slot != dst_slot)
    src_req_idx = tl.load(
        req_idx_by_group + group * req_idx_stride0 + src_slot * req_idx_stride1,
        mask=valid,
        other=-1,
    )
    dst_req_idx = tl.load(
        req_idx_by_group + group * req_idx_stride0 + dst_slot * req_idx_stride1,
        mask=valid,
        other=-1,
    )
    partial_block = prefix_len // block_size
    valid = valid & (src_req_idx >= 0) & (dst_req_idx >= 0)
    src_block = tl.load(
        table + src_req_idx * table_stride0 + partial_block * table_stride1,
        mask=valid,
        other=-1,
    )
    dst_block = tl.load(
        table + dst_req_idx * table_stride0 + partial_block * table_stride1,
        mask=valid,
        other=-1,
    )
    valid = valid & (src_block >= 0) & (dst_block >= 0) & (src_block != dst_block)
    tl.store(src_block_ids + out_idx, src_block, mask=valid)
    tl.store(dst_block_ids + out_idx, dst_block, mask=valid)


@triton.jit
def _snapshot_kv_blocks_kernel(
    cache_pages,
    scratch,
    src_block_ids,
    page_elems: tl.constexpr,
    cache_stride0,
    scratch_stride0,
    BLOCK_N: tl.constexpr,
) -> None:
    pair = tl.program_id(0)
    col_block = tl.program_id(1)
    offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    src_block = tl.load(src_block_ids + pair)
    mask = (src_block >= 0) & (offsets < page_elems)
    values = tl.load(
        cache_pages + src_block * cache_stride0 + offsets,
        mask=mask,
        other=0,
    )
    tl.store(
        scratch + pair * scratch_stride0 + offsets,
        values,
        mask=mask,
    )


@triton.jit
def _rewrite_kv_blocks_kernel(
    cache_pages,
    scratch,
    dst_block_ids,
    page_elems: tl.constexpr,
    cache_stride0,
    scratch_stride0,
    BLOCK_N: tl.constexpr,
) -> None:
    pair = tl.program_id(0)
    col_block = tl.program_id(1)
    offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    dst_block = tl.load(dst_block_ids + pair)
    mask = (dst_block >= 0) & (offsets < page_elems)
    values = tl.load(
        scratch + pair * scratch_stride0 + offsets,
        mask=mask,
        other=0,
    )
    tl.store(
        cache_pages + dst_block * cache_stride0 + offsets,
        values,
        mask=mask,
    )


@triton.jit
def _beam_state_rewrite_kernel(
    token_prefix,
    token_pool,
    cum_pool,
    sampled,
    slot_to_batch,
    dst_slots,
    src_slots,
    selected_tokens,
    selected_scores,
    fork_src,
    active_mask,
    state_slots,
    lengths,
    valid_beam_mask,
    beam_width: tl.constexpr,
    prefix_stride0,
    prefix_stride1,
    prefix_stride2,
    token_width: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    group = tl.program_id(0)
    row = tl.program_id(1)
    col_block = tl.program_id(2)
    offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    group_slot = group * beam_width + row
    valid = tl.load(valid_beam_mask + group_slot)
    state_slot = tl.load(state_slots + group)
    length = tl.load(lengths + group)
    dst_slot = tl.load(dst_slots + group_slot)
    src_slot = tl.load(src_slots + group_slot)
    token = tl.load(selected_tokens + group_slot)

    prefix_mask = offsets < length
    prefix_vals = tl.load(
        token_prefix
        + group * prefix_stride0
        + src_slot * prefix_stride1
        + offsets * prefix_stride2,
        mask=valid & prefix_mask,
        other=0,
    )
    tl.store(
        token_pool
        + (state_slot * beam_width + dst_slot) * token_width
        + offsets,
        prefix_vals,
        mask=valid & prefix_mask,
    )
    tl.store(
        token_pool
        + (state_slot * beam_width + dst_slot) * token_width
        + length,
        token,
        mask=valid & (col_block == 0),
    )

    if col_block == 0:
        score = tl.load(selected_scores + group_slot)
        tl.store(
            cum_pool + state_slot * beam_width + dst_slot,
            score,
            mask=valid,
        )
        batch_row = tl.load(slot_to_batch + group * beam_width + dst_slot)
        tl.store(sampled + batch_row, token, mask=valid & (batch_row >= 0))
        tl.store(
            fork_src + group * beam_width + dst_slot,
            src_slot,
            mask=valid,
        )
        tl.store(
            active_mask + group * beam_width + dst_slot,
            True,
            mask=valid,
        )


def _copy_cpu_values_to_gpu(
    values: Any,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Stage small CPU metadata lists onto GPU without sync-debug trips."""
    cpu = torch.tensor(values, dtype=dtype, pin_memory=True)
    gpu = torch.empty(cpu.shape, dtype=dtype, device=device)
    gpu.copy_(cpu, non_blocking=True)
    return gpu


def _copy_cpu_tensor_to_gpu(
    cpu: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Copy a prebuilt CPU metadata tensor to GPU asynchronously."""
    if not cpu.is_pinned():
        cpu = cpu.pin_memory()
    gpu = torch.empty(cpu.shape, dtype=cpu.dtype, device=device)
    gpu.copy_(cpu, non_blocking=True)
    return gpu


def _kv_cache_pages(cache: torch.Tensor) -> tuple[torch.Tensor, int]:
    """View a KV cache as physical pages without changing storage order."""
    page_elems = int(cache.stride(0))
    return torch.as_strided(
        cache,
        size=(int(cache.shape[0]), page_elems),
        stride=(page_elems, 1),
    ), page_elems


def _copy_kv_pages(
    cache: torch.Tensor,
    src_block_ids: torch.Tensor,
    dst_block_ids: torch.Tensor,
) -> None:
    """Copy full KV pages through a temp buffer so swaps stay correct."""
    cache_pages, page_elems = _kv_cache_pages(cache)
    n_pairs = int(src_block_ids.numel())
    if n_pairs == 0 or page_elems <= 0:
        return

    scratch = torch.empty(
        (n_pairs, page_elems),
        dtype=cache.dtype,
        device=cache.device,
    )
    block_n = min(triton.next_power_of_2(page_elems), 1024)
    grid = (n_pairs, triton.cdiv(page_elems, block_n))
    _snapshot_kv_blocks_kernel[grid](
        cache_pages,
        scratch,
        src_block_ids.reshape(-1),
        page_elems,
        cache_pages.stride(0),
        scratch.stride(0),
        BLOCK_N=block_n,
    )
    _rewrite_kv_blocks_kernel[grid](
        cache_pages,
        scratch,
        dst_block_ids.reshape(-1),
        page_elems,
        cache_pages.stride(0),
        scratch.stride(0),
        BLOCK_N=block_n,
    )


class _BeamGroupStatePool:
    """Contiguous per-group beam state for sampler rewrites."""

    def __init__(
        self,
        *,
        beam_width: int,
        max_len: int,
        device: torch.device,
        capacity: int | None = None,
    ) -> None:
        if capacity is None:
            capacity = int(os.getenv("VLLM_BEAM_GROUP_STATE_CAPACITY", "2048"))
        self.beam_width = beam_width
        self.max_len = max_len
        self.capacity = max(1, capacity)
        self.tokens = torch.empty(
            (self.capacity, beam_width, max_len),
            dtype=torch.int32,
            device=device,
        )
        self.cum = torch.empty(
            (self.capacity, beam_width),
            dtype=torch.float32,
            device=device,
        )
        self._next_slot = 0
        self._free_slots: list[int] = []

    def allocate(self) -> int:
        if self._free_slots:
            return self._free_slots.pop()
        if self._next_slot >= self.capacity:
            raise RuntimeError(
                "Beam group state capacity exceeded; increase "
                "VLLM_BEAM_GROUP_STATE_CAPACITY"
            )
        slot = self._next_slot
        self._next_slot += 1
        return slot

    def release(self, slot: int) -> None:
        self._free_slots.append(slot)

    def initialize(self, slot: int, prompt_tokens: list[int]) -> int:
        self.tokens[slot].zero_()
        prompt = torch.as_tensor(
            prompt_tokens,
            dtype=self.tokens.dtype,
            device=self.tokens.device,
        )
        prompt_len = int(prompt.numel())
        if prompt_len:
            self.tokens[slot, :, :prompt_len] = prompt.unsqueeze(0)
        self.cum[slot].fill_(_INIT_NEG)
        self.cum[slot, 0:1].fill_(0.0)
        return prompt_len


@dataclass
class _BeamGroupGpuState:
    """Per-group slot in the shared GPU beam state pool."""

    pool: _BeamGroupStatePool
    slot: int
    length: int
    prompt_len: int
    step: int = 0


@dataclass
class _BeamTransitionsGpu:
    """GPU transition state for worker rewrite and async output."""

    gids: tuple[str, ...]
    steps: tuple[int, ...]
    prefix_lens: tuple[int, ...]
    prompt_lens: tuple[int, ...]
    active_counts: tuple[int, ...]
    dst_slots: torch.Tensor
    src_slots: torch.Tensor
    fork_src: torch.Tensor
    active_mask: torch.Tensor
    tokens: torch.Tensor
    cum: torch.Tensor
    req_idx_by_group: torch.Tensor | None = None
    completion_scores: torch.Tensor | None = None
    completion_slots: torch.Tensor | None = None
    completion_tokens: torch.Tensor | None = None
    completion_prefixes: torch.Tensor | None = None
    completion_lens: tuple[int, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.gids)

    def async_tensors(self) -> dict[str, torch.Tensor | None]:
        """Return only tensors needed by scheduler finalization."""
        return {
            "fork_src": self.fork_src,
            "active_mask": self.active_mask,
            "tokens": self.tokens,
            "cum": self.cum,
            "completion_scores": self.completion_scores,
            "completion_slots": self.completion_slots,
            "completion_tokens": self.completion_tokens,
            "completion_prefixes": self.completion_prefixes,
        }


def _rewrite_beam_states(
    *,
    group_count: int,
    pool: _BeamGroupStatePool,
    state_slots_by_group: torch.Tensor,
    lengths_cpu: list[int],
    lengths: torch.Tensor,
    sampled: torch.Tensor,
    slot_to_batch: torch.Tensor,
    valid_beam_mask: torch.Tensor,
    snapshot_slots_by_group: torch.Tensor,
    dst_slots_by_group: torch.Tensor,
    src_slots_by_group: torch.Tensor,
    selected_tokens_by_group: torch.Tensor,
    selected_scores_by_group: torch.Tensor,
    beam_width: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Rewrite shared-pool beam state with fixed sampler launch count."""
    device = sampled.device
    max_length = max(lengths_cpu)
    token_prefix = torch.empty(
        (group_count, beam_width, max_length),
        dtype=pool.tokens.dtype,
        device=device,
    )
    fork_src = torch.empty(
        (group_count, beam_width),
        dtype=torch.long,
        device=device,
    )
    active_mask = torch.empty(
        (group_count, beam_width),
        dtype=torch.bool,
        device=device,
    )

    block_n = _BEAM_STATE_BLOCK_N
    prefix_grid = (group_count, beam_width, triton.cdiv(max_length, block_n))
    _snapshot_beam_prefix_kernel[prefix_grid](
        pool.tokens,
        token_prefix,
        snapshot_slots_by_group,
        valid_beam_mask,
        state_slots_by_group,
        lengths,
        beam_width,
        pool.tokens.stride(0),
        pool.tokens.stride(1),
        pool.tokens.stride(2),
        token_prefix.stride(0),
        token_prefix.stride(1),
        token_prefix.stride(2),
        src_slots_by_group.stride(0),
        src_slots_by_group.stride(1),
        BLOCK_N=block_n,
    )

    init_grid = (group_count, beam_width)
    _init_beam_transition_kernel[init_grid](
        fork_src,
        active_mask,
        beam_width,
        fork_src.stride(0),
        fork_src.stride(1),
        active_mask.stride(0),
        active_mask.stride(1),
    )

    n_cols = max_length + 1
    block_n = _BEAM_STATE_BLOCK_N
    rewrite_grid = (group_count, beam_width, triton.cdiv(n_cols, block_n))
    selected_tokens = selected_tokens_by_group.to(torch.int64)
    _beam_state_rewrite_kernel[rewrite_grid](
        token_prefix,
        pool.tokens,
        pool.cum,
        sampled,
        slot_to_batch,
        dst_slots_by_group,
        src_slots_by_group,
        selected_tokens,
        selected_scores_by_group,
        fork_src,
        active_mask,
        state_slots_by_group,
        lengths,
        valid_beam_mask,
        beam_width,
        token_prefix.stride(0),
        token_prefix.stride(1),
        token_prefix.stride(2),
        token_width=pool.max_len,
        BLOCK_N=block_n,
    )
    return fork_src, active_mask, token_prefix


def _snapshot_transition_state(
    *,
    pool: _BeamGroupStatePool,
    state_slots_by_group: torch.Tensor,
    lengths: torch.Tensor,
    max_length: int,
    beam_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Snapshot rewritten group state for async CPU finalization."""
    group_count = state_slots_by_group.shape[0]
    tokens = torch.empty(
        (group_count, beam_width, max_length + 1),
        dtype=pool.tokens.dtype,
        device=pool.tokens.device,
    )
    cum = torch.empty(
        (group_count, beam_width),
        dtype=pool.cum.dtype,
        device=pool.cum.device,
    )
    block_n = _BEAM_STATE_BLOCK_N
    grid = (group_count, beam_width, triton.cdiv(max_length + 1, block_n))
    _snapshot_transition_state_kernel[grid](
        pool.tokens,
        pool.cum,
        tokens,
        cum,
        state_slots_by_group,
        lengths,
        beam_width,
        pool.tokens.stride(0),
        pool.tokens.stride(1),
        pool.tokens.stride(2),
        pool.cum.stride(0),
        pool.cum.stride(1),
        tokens.stride(0),
        tokens.stride(1),
        tokens.stride(2),
        cum.stride(0),
        cum.stride(1),
        BLOCK_N=block_n,
    )
    return tokens, cum


def _completion_lens(
    lengths_cpu: list[int],
    prompt_lens_cpu: list[int],
) -> tuple[int, ...]:
    """Return generated EOS candidate lengths by group."""
    return tuple(
        length - prompt_len + 1
        for length, prompt_len in zip(lengths_cpu, prompt_lens_cpu)
    )


@contextmanager
def _gpu_sync_check():
    if _SYNC_CHECK not in {"warn", "error"} or not torch.cuda.is_available():
        yield
        return

    prev_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode(_SYNC_CHECK)
    try:
        yield
    except RuntimeError as exc:
        if str(exc) == "called a synchronizing CUDA operation":
            raise RuntimeError(
                "GPU<->CPU sync detected in beam MRV2 sampler"
            ) from exc
        raise
    finally:
        torch.cuda.set_sync_debug_mode(prev_mode)


class BeamSearchMRV2Sampler:
    """Beam-aware wrapper around MRV2's GPU sampler."""

    def __init__(
        self,
        base_sampler: Any,
        vllm_config: VllmConfig,
        device: torch.device,
        pin_memory: bool = False,
    ) -> None:
        self.base_sampler = base_sampler
        self.groups: dict[str, BeamRuntime] = {}
        self.req_to_group: dict[str, tuple[str, int]] = {}
        self.block_tables: Any | None = None
        self.self_attn_group_indices: tuple[int, ...] = ()
        self.kv_cache_config: Any | None = None
        self.kv_cache_forward_context: dict[str, Any] | None = None
        self._pending_computed_resets: list[
            tuple[torch.Tensor, torch.Tensor]
        ] = []
        self._gpu_group_state: dict[str, _BeamGroupGpuState] = {}
        self._gpu_state_pools: dict[int, _BeamGroupStatePool] = {}
        self._transition_buffer_pool = _TransitionBufferPool()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_sampler, name)

    def add_request(
        self,
        req_idx: int,
        prompt_len: int,
        sampling_params: SamplingParams,
    ) -> None:
        self.base_sampler.add_request(req_idx, prompt_len, sampling_params)

    def apply_staged_writes(self) -> None:
        self.base_sampler.apply_staged_writes()

    def set_block_tables(
        self,
        block_tables: Any,
        self_attn_group_indices: tuple[int, ...],
    ) -> None:
        self.block_tables = block_tables
        self.self_attn_group_indices = self_attn_group_indices

    def set_kv_caches(
        self,
        kv_cache_config: Any,
        forward_context: dict[str, Any],
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self.kv_cache_forward_context = forward_context

    def register_request(
        self,
        req_id: str,
        sampling_params: SamplingParams | None,
        prompt_token_ids: list[int] | None,
    ) -> None:
        if sampling_params is None:
            return
        extra = sampling_params.extra_args or {}
        gid = extra.get("_beam_group_id")
        if gid is None:
            return

        beam_idx = int(extra["_beam_index"])
        beam_width = int(extra["_beam_width"])
        eos = extra.get("_beam_eos_token_id")
        no_repeat = int(extra.get("no_repeat_ngram_size", 0))

        gr = self.groups.get(gid)
        if gr is None:
            gr = BeamRuntime(
                beam_width=beam_width,
                eos_token_id=int(eos) if eos is not None else None,
                no_repeat_ngram_size=no_repeat,
                prompt_tokens=list(prompt_token_ids or []),
            )
            self.groups[gid] = gr
            self._init_gpu_group_state(gid, gr)

        self.req_to_group[req_id] = (gid, beam_idx)

    def _init_gpu_group_state(self, gid: str, gr: BeamRuntime) -> None:
        if gid in self._gpu_group_state:
            return
        pool = self._get_gpu_state_pool(gr.beam_width)
        slot = pool.allocate()
        prompt_len = pool.initialize(slot, gr.prompt_tokens)
        self._gpu_group_state[gid] = _BeamGroupGpuState(
            pool=pool,
            slot=slot,
            length=prompt_len,
            prompt_len=prompt_len,
        )

    def _get_gpu_state_pool(self, beam_width: int) -> _BeamGroupStatePool:
        pool = self._gpu_state_pools.get(beam_width)
        if pool is not None:
            return pool
        pool = _BeamGroupStatePool(
            beam_width=beam_width,
            max_len=int(self.base_sampler.req_states.max_model_len),
            device=self.base_sampler.req_states.all_token_ids.gpu.device,
        )
        self._gpu_state_pools[beam_width] = pool
        return pool

    def remove_request(self, req_id: str) -> None:
        entry = self.req_to_group.pop(req_id, None)
        if entry is None:
            return
        gid, _beam_idx = entry
        gr = self.groups.get(gid)
        if gr is None:
            return
        if not any(group_id == gid for group_id, _ in self.req_to_group.values()):
            self.groups.pop(gid, None)
            state = self._gpu_group_state.pop(gid, None)
            if state is not None:
                state.pool.release(state.slot)

    def __call__(self, logits: torch.Tensor, input_batch: InputBatch) -> SamplerOutput:
        if not self._has_beam_requests(input_batch):
            return self.base_sampler(logits, input_batch)

        with _gpu_sync_check():
            return self._sample_with_beam_logits(logits, input_batch)

    def _has_beam_requests(self, input_batch: InputBatch) -> bool:
        return any(req_id in self.req_to_group for req_id in input_batch.req_ids)

    def _sample_with_beam_logits(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        sampler = self.base_sampler
        expanded_idx_mapping = input_batch.expanded_idx_mapping
        idx_mapping_np = input_batch.idx_mapping_np
        expanded_local_pos = input_batch.expanded_local_pos
        pos = input_batch.positions[input_batch.logits_indices]
        input_ids = input_batch.input_ids[input_batch.logits_indices]

        num_nans = get_num_nans(logits) if sampler.compute_nans else None

        processed_logits = sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            skip_top_k_top_p=True,
        )

        return self._sample_with_gpu_beam_select(
            processed_logits,
            input_batch,
            num_nans,
        )

    def _sample_with_gpu_beam_select(
        self,
        processed_logits: torch.Tensor,
        input_batch: InputBatch,
        num_nans: torch.Tensor | None,
    ) -> SamplerOutput:
        sampled = torch.zeros(
            input_batch.num_reqs,
            dtype=torch.int64,
            device=processed_logits.device,
        )
        transitions = self._gpu_select_groups(
            processed_logits=processed_logits,
            input_batch=input_batch,
            sampled=sampled,
        )

        if transitions:
            self._apply_gpu_worker_rewrites(transitions)
            async_outputs = {
                BEAM_TRANSITIONS_OUTPUT: AsyncBeamTransitions(
                    transitions,
                    buffer_pool=self._transition_buffer_pool,
                )
            }
        else:
            async_outputs = None

        num_sampled, num_rejected = get_num_sampled_and_rejected(
            input_batch.seq_lens.new_ones(input_batch.num_reqs),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.base_sampler.req_states.prefill_len.gpu,
        )
        return BeamSamplerOutput(
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=None,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            async_outputs=async_outputs,
        )

    def _gpu_select_groups(
        self,
        *,
        processed_logits: torch.Tensor,
        input_batch: InputBatch,
        sampled: torch.Tensor,
    ) -> _BeamTransitionsGpu | None:
        beam_view = self._beam_view_by_group(
            input_batch,
            processed_logits.shape[0],
        )
        key, group_states = self._collect_group_states(beam_view)
        if key is None:
            return None

        bw, no_repeat, eos = key
        device = processed_logits.device
        group_count = len(group_states)
        lengths_cpu = [state.length for _gid, state in group_states]
        prompt_lens_cpu = [state.prompt_len for _gid, state in group_states]
        pools = {state.pool for _gid, state in group_states}
        assert len(pools) == 1
        pool = pools.pop()
        state_slots_cpu = [state.slot for _gid, state in group_states]
        max_length = max(lengths_cpu)
        beam_metadata = _copy_cpu_tensor_to_gpu(
            beam_view.beam_metadata_cpu,
            device=device,
        )
        beam_indices = beam_metadata[_BEAM_IDX]
        logit_indices = beam_metadata[_LOGIT_IDX]
        slot_to_batch = beam_metadata[_INPUT_BATCH_IDX]
        req_idx_by_group = beam_metadata[_REQ_IDX]
        valid_beam_mask = _copy_cpu_tensor_to_gpu(
            beam_view.valid_beam_mask_cpu,
            device=device,
        )
        lengths = _copy_cpu_values_to_gpu(
            lengths_cpu,
            dtype=torch.long,
            device=device,
        )
        state_slots_by_group = _copy_cpu_values_to_gpu(
            state_slots_cpu,
            dtype=torch.long,
            device=device,
        )

        flat_logit_idx = logit_indices.reshape(-1)
        vocab_size = processed_logits.shape[1]
        logprobs = torch.log_softmax(
            processed_logits[flat_logit_idx].float(),
            dim=-1,
        ).view(group_count, bw, vocab_size)

        if no_repeat > 0 and max_length >= no_repeat:
            grid = (group_count, bw, max_length - (no_repeat - 1))
            _no_repeat_ngram_mask_kernel[grid](
                pool.tokens,
                logprobs,
                beam_indices,
                state_slots_by_group,
                lengths,
                no_repeat,
                bw,
                pool.tokens.stride(0),
                pool.tokens.stride(1),
                pool.tokens.stride(2),
                logprobs.stride(0),
                logprobs.stride(1),
                logprobs.stride(2),
            )

        k = min(vocab_size, 2 * bw)
        vals, ids = torch.topk(logprobs, k, dim=-1)
        bases = pool.cum[
            state_slots_by_group[:, None],
            beam_indices,
        ].unsqueeze(2)
        scores = vals + bases
        scores = scores.masked_fill(~valid_beam_mask.unsqueeze(2), _NEG_INF)

        eos_completion_scores = None
        completion_slots = None
        completion_tokens = None
        if eos is not None:
            flat_scores_all, flat_idx_all = torch.topk(
                scores.reshape(group_count, -1),
                min(2 * bw, scores.shape[1] * scores.shape[2]),
                dim=-1,
            )
            src_pos_all = flat_idx_all // k
            tok_all = ids.reshape(group_count, -1).gather(1, flat_idx_all)
            eos_mask = tok_all == int(eos)
            rank = torch.arange(
                flat_scores_all.shape[1],
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)
            completion_mask = eos_mask & (rank < bw)
            eos_completion_scores = flat_scores_all.masked_fill(
                ~completion_mask, _INIT_NEG
            )
            completion_slots = beam_indices.gather(1, src_pos_all)
            completion_tokens = tok_all
            scores = scores.masked_fill(ids == int(eos), _NEG_INF)

        # Global per-group top-k over beam slots x token candidates.
        flat_scores, flat_idx = torch.topk(
            scores.reshape(group_count, -1),
            bw,
            dim=-1,
        )
        # Decode flattened candidate indices back to source slot and token rank.
        src_pos, _tok_pos = flat_idx // k, flat_idx % k
        src_slots_by_group = beam_indices.gather(1, src_pos)
        selected_tokens_by_group = ids.reshape(group_count, -1).gather(1, flat_idx)
        dst_slots_by_group = beam_indices[:, :bw]

        completion_lens = _completion_lens(
            lengths_cpu=lengths_cpu,
            prompt_lens_cpu=prompt_lens_cpu,
        )

        (
            fork_src_by_group,
            active_mask_by_group,
            completion_prefixes,
        ) = _rewrite_beam_states(
            group_count=group_count,
            pool=pool,
            state_slots_by_group=state_slots_by_group,
            lengths_cpu=lengths_cpu,
            lengths=lengths,
            sampled=sampled,
            slot_to_batch=slot_to_batch,
            valid_beam_mask=valid_beam_mask,
            snapshot_slots_by_group=beam_indices,
            dst_slots_by_group=dst_slots_by_group,
            src_slots_by_group=src_slots_by_group,
            selected_tokens_by_group=selected_tokens_by_group,
            selected_scores_by_group=flat_scores,
            beam_width=bw,
        )
        transition_tokens, transition_cum = _snapshot_transition_state(
            pool=pool,
            state_slots_by_group=state_slots_by_group,
            lengths=lengths,
            max_length=max_length,
            beam_width=bw,
        )

        gids: list[str] = []
        steps: list[int] = []
        for group_idx, (gid, state) in enumerate(group_states):
            length = lengths_cpu[group_idx]
            gids.append(gid)
            steps.append(state.step)
            state.length = length + 1
            state.step += 1

        return _BeamTransitionsGpu(
            gids=tuple(gids),
            steps=tuple(steps),
            prefix_lens=tuple(lengths_cpu),
            prompt_lens=tuple(prompt_lens_cpu),
            active_counts=beam_view.active_slot_counts,
            req_idx_by_group=req_idx_by_group,
            dst_slots=dst_slots_by_group,
            src_slots=src_slots_by_group,
            fork_src=fork_src_by_group,
            active_mask=active_mask_by_group,
            tokens=transition_tokens,
            cum=transition_cum,
            completion_scores=eos_completion_scores,
            completion_slots=completion_slots,
            completion_tokens=completion_tokens,
            completion_prefixes=(
                completion_prefixes if eos_completion_scores is not None else None
            ),
            completion_lens=completion_lens,
        )

    def _collect_group_states(
        self,
        beam_view: _BeamView,
    ) -> tuple[_SelectKey | None, list[_GroupStateEntry]]:
        """Collect one homogeneous selection group for the MRV2 beam path."""
        key = beam_view.key
        group_states: list[_GroupStateEntry] = []
        if key is None:
            return None, group_states
        for gid in beam_view.gids:
            gr = self.groups.get(gid)
            if gr is None:
                continue
            state = self._gpu_group_state.get(gid)
            if state is None:
                self._init_gpu_group_state(gid, gr)
                state = self._gpu_group_state[gid]
            group_states.append((gid, state))
        return key, group_states

    def _beam_view_by_group(
        self,
        input_batch: InputBatch,
        num_logits: int,
    ) -> _BeamView:
        idx_mapping_np = getattr(input_batch, "idx_mapping_np", None)
        req_to_group = self.req_to_group
        groups = self.groups
        beam_requests: list[_BeamRequest] = [
            (batch_idx, gid, beam_idx, gr)
            for batch_idx, req_id in enumerate(input_batch.req_ids)
            if (entry := req_to_group.get(req_id)) is not None
            for gid, beam_idx in (entry,)
            if (gr := groups.get(gid)) is not None
            if gr.beam_width > 1
            if 0 <= beam_idx < gr.beam_width
        ]

        if not beam_requests:
            return _empty_beam_view()

        batch_idx_np = np.fromiter(
            (request[0] for request in beam_requests),
            dtype=np.int64,
            count=len(beam_requests),
        )
        beam_idx_np = np.fromiter(
            (request[2] for request in beam_requests),
            dtype=np.int64,
            count=len(beam_requests),
        )
        cu_num_logits = input_batch.cu_num_logits_np
        cu_start = cu_num_logits[batch_idx_np]
        cu_end = cu_num_logits[batch_idx_np + 1]
        logit_idx_np = cu_end - 1
        valid_np = (
            (cu_end > cu_start)
            & (logit_idx_np >= 0)
            & (logit_idx_np < num_logits)
        )
        valid_request_pos_np = np.nonzero(valid_np)[0]
        if valid_request_pos_np.size == 0:
            return _empty_beam_view()

        _batch_idx, _gid, _beam_idx, first_group = beam_requests[
            int(valid_request_pos_np[0])
        ]
        key = _beam_select_key(first_group)
        beam_width = key[0]
        gids: list[str] = []
        group_idx_by_gid: dict[str, int] = {}
        padding_request_pos: list[int] = []
        slot_counts_by_group: list[int] = []
        group_idx_np = np.empty(valid_request_pos_np.size, dtype=np.int64)
        slot_pos_np = np.empty(valid_request_pos_np.size, dtype=np.int64)
        for compact_idx, request_pos in enumerate(valid_request_pos_np):
            _batch_idx, gid, _beam_idx, gr = beam_requests[int(request_pos)]
            if _beam_select_key(gr) != key:
                raise RuntimeError(
                    "BeamSearchMRV2Sampler requires homogeneous beam "
                    "settings within one scheduler step."
                )
            group_idx = group_idx_by_gid.get(gid)
            if group_idx is None:
                group_idx = len(gids)
                group_idx_by_gid[gid] = group_idx
                gids.append(gid)
                padding_request_pos.append(compact_idx)
                slot_counts_by_group.append(0)
            group_idx_np[compact_idx] = group_idx
            slot_pos_np[compact_idx] = slot_counts_by_group[group_idx]
            slot_counts_by_group[group_idx] += 1

        group_count = len(gids)
        padding_request_pos_np = np.asarray(padding_request_pos, dtype=np.int64)
        batch_idx_np = batch_idx_np[valid_request_pos_np]
        beam_idx_np = beam_idx_np[valid_request_pos_np]
        logit_idx_np = logit_idx_np[valid_request_pos_np].astype(
            np.int64,
            copy=False,
        )
        if idx_mapping_np is None:
            req_idx_np = batch_idx_np
        else:
            req_idx_np = idx_mapping_np[batch_idx_np].astype(
                np.int64,
                copy=False,
            )

        valid_beam_entries_np = slot_pos_np < beam_width
        valid_group_idx_np = group_idx_np[valid_beam_entries_np]
        valid_slot_pos_np = slot_pos_np[valid_beam_entries_np]
        valid_beam_idx_np = beam_idx_np[valid_beam_entries_np]
        valid_logit_idx_np = logit_idx_np[valid_beam_entries_np]
        valid_batch_idx_np = batch_idx_np[valid_beam_entries_np]
        valid_req_idx_np = req_idx_np[valid_beam_entries_np]

        active_slot_counts_np = np.bincount(
            valid_group_idx_np,
            minlength=group_count,
        )
        valid_beam_mask_np = np.zeros(
            (group_count, beam_width),
            dtype=np.bool_,
        )
        beam_idx_by_slot_np = np.broadcast_to(
            beam_idx_np[padding_request_pos_np, None],
            (group_count, beam_width),
        ).copy()
        logit_idx_by_slot_np = np.broadcast_to(
            logit_idx_np[padding_request_pos_np, None],
            (group_count, beam_width),
        ).copy()
        valid_beam_mask_np[valid_group_idx_np, valid_slot_pos_np] = True
        beam_idx_by_slot_np[valid_group_idx_np, valid_slot_pos_np] = (
            valid_beam_idx_np
        )
        logit_idx_by_slot_np[valid_group_idx_np, valid_slot_pos_np] = (
            valid_logit_idx_np
        )

        input_batch_idx_by_beam_np = np.full(
            (group_count, beam_width),
            -1,
            dtype=np.int64,
        )
        req_idx_by_beam_np = np.full(
            (group_count, beam_width),
            -1,
            dtype=np.int64,
        )
        input_batch_idx_by_beam_np[
            valid_group_idx_np,
            valid_beam_idx_np,
        ] = valid_batch_idx_np
        req_idx_by_beam_np[
            valid_group_idx_np,
            valid_beam_idx_np,
        ] = valid_req_idx_np

        beam_metadata_np = np.empty(
            (_NUM_SLOT_INDEX_FIELDS, group_count, beam_width),
            dtype=np.int64,
        )
        beam_metadata_np[_BEAM_IDX] = beam_idx_by_slot_np
        beam_metadata_np[_LOGIT_IDX] = logit_idx_by_slot_np
        beam_metadata_np[_INPUT_BATCH_IDX] = input_batch_idx_by_beam_np
        beam_metadata_np[_REQ_IDX] = req_idx_by_beam_np

        beam_metadata_cpu = torch.empty(
            beam_metadata_np.shape,
            dtype=torch.long,
            pin_memory=True,
        )
        beam_metadata_cpu.copy_(torch.from_numpy(beam_metadata_np))
        valid_beam_mask_cpu = torch.empty(
            valid_beam_mask_np.shape,
            dtype=torch.bool,
            pin_memory=True,
        )
        valid_beam_mask_cpu.copy_(torch.from_numpy(valid_beam_mask_np))
        active_slot_counts = tuple(int(count) for count in active_slot_counts_np)

        return _BeamView(
            gids=tuple(gids),
            key=key,
            active_slot_counts=active_slot_counts,
            beam_metadata_cpu=beam_metadata_cpu,
            valid_beam_mask_cpu=valid_beam_mask_cpu,
        )

    def _apply_gpu_worker_rewrites(
        self,
        transitions: _BeamTransitionsGpu,
    ) -> None:
        """Apply transition rewrites with fixed plugin launch count."""
        if self.block_tables is None or not self.self_attn_group_indices:
            return
        req_idx_by_group = transitions.req_idx_by_group
        if req_idx_by_group is None:
            raise RuntimeError("MRV2 beam transitions must include request indices")
        req_states = self.base_sampler.req_states
        group_count = len(transitions.gids)
        if group_count == 0:
            return

        device = req_idx_by_group.device
        beam_width = int(transitions.dst_slots.shape[1])
        prefix_lens = _copy_cpu_values_to_gpu(
            transitions.prefix_lens,
            dtype=torch.long,
            device=device,
        )
        active_counts = _copy_cpu_values_to_gpu(
            transitions.active_counts,
            dtype=torch.long,
            device=device,
        )
        max_prefix_len = max(transitions.prefix_lens, default=0)
        pending_req_idx = torch.empty(
            (group_count, beam_width),
            dtype=torch.long,
            device=device,
        )
        pending_prefix_lens = torch.empty(
            (group_count, beam_width),
            dtype=torch.long,
            device=device,
        )

        n_cols = max(max_prefix_len, 1)
        block_n = _BEAM_STATE_BLOCK_N
        grid = (group_count, beam_width, triton.cdiv(n_cols, block_n))
        _rewrite_worker_tokens_kernel[grid](
            transitions.tokens,
            req_states.all_token_ids.gpu,
            req_states.total_len.gpu,
            req_states.num_computed_tokens.gpu,
            req_idx_by_group,
            transitions.dst_slots,
            prefix_lens,
            active_counts,
            pending_req_idx,
            pending_prefix_lens,
            beam_width,
            transitions.tokens.stride(0),
            transitions.tokens.stride(1),
            transitions.tokens.stride(2),
            req_states.all_token_ids.gpu.stride(0),
            req_states.all_token_ids.gpu.stride(1),
            req_idx_by_group.stride(0),
            req_idx_by_group.stride(1),
            transitions.dst_slots.stride(0),
            transitions.dst_slots.stride(1),
            pending_req_idx.stride(0),
            pending_req_idx.stride(1),
            BLOCK_N=block_n,
        )
        self._pending_computed_resets.append((
            pending_req_idx.reshape(-1),
            pending_prefix_lens.reshape(-1),
        ))

        for group_idx in self.self_attn_group_indices:
            block_size = int(self.block_tables.block_sizes[group_idx])
            max_prefix_blocks = max_prefix_len // block_size
            table = self.block_tables.block_tables[group_idx].gpu
            if max_prefix_blocks > 0:
                block_prefix = torch.empty(
                    (group_count, beam_width, max_prefix_blocks),
                    dtype=table.dtype,
                    device=table.device,
                )
                block_n = _BEAM_STATE_BLOCK_N
                grid = (
                    group_count,
                    beam_width,
                    triton.cdiv(max_prefix_blocks, block_n),
                )
                _snapshot_worker_block_prefix_kernel[grid](
                    table,
                    block_prefix,
                    req_idx_by_group,
                    transitions.src_slots,
                    prefix_lens,
                    active_counts,
                    block_size,
                    beam_width,
                    table.stride(0),
                    table.stride(1),
                    block_prefix.stride(0),
                    block_prefix.stride(1),
                    block_prefix.stride(2),
                    req_idx_by_group.stride(0),
                    req_idx_by_group.stride(1),
                    transitions.src_slots.stride(0),
                    transitions.src_slots.stride(1),
                    BLOCK_N=block_n,
                )
                _rewrite_worker_block_prefix_kernel[grid](
                    table,
                    block_prefix,
                    req_idx_by_group,
                    transitions.dst_slots,
                    prefix_lens,
                    active_counts,
                    block_size,
                    beam_width,
                    table.stride(0),
                    table.stride(1),
                    block_prefix.stride(0),
                    block_prefix.stride(1),
                    block_prefix.stride(2),
                    req_idx_by_group.stride(0),
                    req_idx_by_group.stride(1),
                    transitions.dst_slots.stride(0),
                    transitions.dst_slots.stride(1),
                    BLOCK_N=block_n,
                )
            if block_size > 1:
                self._copy_partial_worker_kv_blocks(
                    group_idx=group_idx,
                    table=table,
                    req_idx_by_group=req_idx_by_group,
                    transitions=transitions,
                    prefix_lens=prefix_lens,
                    active_counts=active_counts,
                    block_size=block_size,
                    beam_width=beam_width,
                )

    def _copy_partial_worker_kv_blocks(
        self,
        *,
        group_idx: int,
        table: torch.Tensor,
        req_idx_by_group: torch.Tensor,
        transitions: _BeamTransitionsGpu,
        prefix_lens: torch.Tensor,
        active_counts: torch.Tensor,
        block_size: int,
        beam_width: int,
    ) -> None:
        """Copy source partial KV pages into destination-owned pages."""
        if not any(length % block_size for length in transitions.prefix_lens):
            return

        group_count = len(transitions.gids)
        src_block_ids = torch.empty(
            (group_count, beam_width),
            dtype=table.dtype,
            device=table.device,
        )
        dst_block_ids = torch.empty_like(src_block_ids)
        _partial_block_ids_kernel[(group_count, beam_width)](
            table,
            req_idx_by_group,
            transitions.src_slots,
            transitions.dst_slots,
            prefix_lens,
            active_counts,
            src_block_ids,
            dst_block_ids,
            block_size,
            beam_width,
            table.stride(0),
            table.stride(1),
            req_idx_by_group.stride(0),
            req_idx_by_group.stride(1),
            transitions.src_slots.stride(0),
            transitions.src_slots.stride(1),
            transitions.dst_slots.stride(0),
            transitions.dst_slots.stride(1),
        )

        for cache in self._self_attn_kv_caches(group_idx):
            _copy_kv_pages(cache, src_block_ids, dst_block_ids)

    def _self_attn_kv_caches(self, group_idx: int) -> list[torch.Tensor]:
        kv_cache_config = self.kv_cache_config
        forward_context = self.kv_cache_forward_context
        if kv_cache_config is None or forward_context is None:
            return []
        if group_idx >= len(kv_cache_config.kv_cache_groups):
            return []

        caches: list[torch.Tensor] = []
        seen: set[int] = set()
        group = kv_cache_config.kv_cache_groups[group_idx]
        for layer_name in group.layer_names:
            attn_layer = forward_context.get(layer_name)
            cache = getattr(attn_layer, "kv_cache", None)
            if not isinstance(cache, torch.Tensor):
                continue
            key = cache.data_ptr()
            if key in seen:
                continue
            seen.add(key)
            caches.append(cache)
        return caches

    def apply_pending_rewrites(self) -> None:
        self._apply_pending_computed_resets()

    def _apply_pending_computed_resets(self) -> None:
        if not self._pending_computed_resets:
            return
        num_computed = self.base_sampler.req_states.num_computed_tokens.gpu
        for req_idx, prefix_len in self._pending_computed_resets:
            n_items = req_idx.numel()
            block_n = 256
            _set_num_computed_kernel[(triton.cdiv(n_items, block_n),)](
                num_computed,
                req_idx,
                prefix_len,
                n_items,
                req_idx.stride(0),
                prefix_len.stride(0),
                BLOCK_N=block_n,
            )
        self._pending_computed_resets.clear()


class _TransitionBufferPool:
    """Persistent GPU slots for async transition snapshots."""

    def __init__(self, num_slots: int | None = None) -> None:
        if num_slots is None:
            num_slots = int(os.getenv("VLLM_BEAM_TRANSITION_BUFFER_SLOTS", "8"))
        self.num_slots = max(1, num_slots)
        self._next_slot = 0
        self._buffers: dict[
            tuple[str, tuple[int, ...], torch.dtype, torch.device],
            torch.Tensor,
        ] = {}

    def snapshot(
        self,
        transitions: _BeamTransitionsGpu,
    ) -> dict[str, torch.Tensor | None]:
        """Copy GPU transition tensors into persistent slots."""
        slot = self._next_slot
        self._next_slot = (self._next_slot + 1) % self.num_slots

        tensors: dict[str, torch.Tensor | None] = {}
        for key, value in transitions.async_tensors().items():
            if value is None:
                tensors[key] = None
                continue
            out = self._slot_view(key, value, slot)
            out.copy_(value)
            tensors[key] = out

        return tensors

    def _slot_view(
        self,
        key: str,
        value: torch.Tensor,
        slot: int,
    ) -> torch.Tensor:
        buffer_key = (key, tuple(value.shape[1:]), value.dtype, value.device)
        buffer = self._buffers.get(buffer_key)
        group_count = value.shape[0]
        if buffer is None or buffer.shape[1] < group_count:
            capacity = group_count
            if buffer is not None:
                capacity = max(capacity, buffer.shape[1] * 2)
            self._buffers[buffer_key] = torch.empty(
                (self.num_slots, capacity, *value.shape[1:]),
                dtype=value.dtype,
                device=value.device,
            )
            buffer = self._buffers[buffer_key]
        return buffer[slot, :group_count]


class AsyncBeamTransitions:
    def __init__(
        self,
        transitions: _BeamTransitionsGpu | None = None,
        *,
        tensors: dict[str, torch.Tensor | None] | None = None,
        gids: tuple[str, ...] = (),
        steps: tuple[int, ...] = (),
        prefix_lens: tuple[int, ...] = (),
        prompt_lens: tuple[int, ...] = (),
        completion_lens: tuple[int, ...] = (),
        gpu_refs: tuple[torch.Tensor, ...] = (),
        buffer_pool: _TransitionBufferPool | None = None,
    ) -> None:
        """Hold sampler transitions until vLLM async output finalization."""
        self.gpu_transitions = transitions
        if transitions is not None:
            self.gids = transitions.gids
            self.steps = transitions.steps
            self.prefix_lens = transitions.prefix_lens
            self.prompt_lens = transitions.prompt_lens
            self.completion_lens = transitions.completion_lens
        else:
            self.gids = gids
            self.steps = steps
            self.prefix_lens = prefix_lens
            self.prompt_lens = prompt_lens
            self.completion_lens = completion_lens
        self.tensors = tensors
        self.gpu_refs = gpu_refs
        self._buffer_pool = buffer_pool or _TransitionBufferPool()

    def to_cpu_nonblocking(self) -> "AsyncBeamTransitions":
        """Stage transition tensors for CPU finalization."""
        with _gpu_sync_check():
            tensors = self.tensors
            if tensors is None:
                assert self.gpu_transitions is not None
                tensors = self._buffer_pool.snapshot(self.gpu_transitions)

            cpu_tensors: dict[str, torch.Tensor | None] = {}
            gpu_refs: list[torch.Tensor] = []
            for key, value in tensors.items():
                if value is None or value.device.type == "cpu":
                    cpu_tensors[key] = value
                    continue
                cpu_tensors[key] = value.to("cpu", non_blocking=True)
                gpu_refs.append(value)
            return AsyncBeamTransitions(
                tensors=cpu_tensors,
                gids=self.gids,
                steps=self.steps,
                prefix_lens=self.prefix_lens,
                prompt_lens=self.prompt_lens,
                completion_lens=self.completion_lens,
                gpu_refs=tuple(gpu_refs),
                buffer_pool=self._buffer_pool,
            )

    def to_output(self) -> tuple[BeamTransition, ...]:
        """Convert staged tensors into CPU BeamTransition records."""
        tensors = self.tensors
        if tensors is None:
            if self.gpu_transitions is None:
                return ()
            tensors = self.gpu_transitions.async_tensors()
        return tuple(
            _materialize_transition(
                self._transition_data(group_idx, gid, tensors)
            )
            for group_idx, gid in enumerate(self.gids)
        )

    def _transition_data(
        self,
        group_idx: int,
        gid: str,
        tensors: dict[str, torch.Tensor | None],
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "gid": gid,
            "step": self.steps[group_idx],
            "prefix_len": self.prefix_lens[group_idx],
            "prompt_len": self.prompt_lens[group_idx],
        }
        if self.completion_lens:
            data["completion_len"] = self.completion_lens[group_idx]
        for key, value in tensors.items():
            data[key] = None if value is None else value[group_idx]
        return data


def _materialize_transition(data: dict[str, Any]) -> BeamTransition:
    """Turn one async sampler sidecar into a scheduler transition."""
    prompt_len = int(data["prompt_len"])
    prefix_len = int(data["prefix_len"])
    length = prefix_len + 1
    token_sequences = data["tokens"][:, :length].tolist()
    generated = [
        tuple(int(tok) for tok in sequence[prompt_len:length])
        for sequence in token_sequences
    ]
    cum = tuple(float(x) for x in data["cum"].tolist())
    fork_src = tuple(int(x) for x in data["fork_src"].tolist())
    active_mask = data["active_mask"].tolist()
    active_slots = tuple(
        slot
        for slot, active in enumerate(active_mask)
        if active
    )

    completions: list[tuple[tuple[int, ...], float]] = []
    completion_scores = data.get("completion_scores")
    completion_prefixes = data.get("completion_prefixes")
    completion_slots = data.get("completion_slots")
    completion_tokens = data.get("completion_tokens")
    if (
        completion_scores is not None
        and completion_scores.numel() > 0
    ):
        if (
            completion_prefixes is not None
            and completion_slots is not None
            and completion_tokens is not None
        ):
            scores = completion_scores.tolist()
            slots = completion_slots.tolist()
            tokens = completion_tokens.tolist()
            for slot, token, score in zip(slots, tokens, scores):
                if score <= _INIT_NEG / 2:
                    continue
                prefix = completion_prefixes[
                    int(slot), prompt_len:prefix_len
                ].tolist()
                completions.append(
                    (
                        tuple(int(tok) for tok in (*prefix, int(token))),
                        float(score),
                    )
                )

    return BeamTransition(
        group_id=str(data["gid"]),
        step_id=int(data["step"]),
        prefix_len=int(data["prefix_len"]) - prompt_len,
        active_slots=active_slots,
        fork_src=fork_src,
        tokens=tuple(generated),
        cum=cum,
        completions=tuple(completions),
    )
