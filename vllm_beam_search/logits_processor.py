"""BeamSearchLogitsProcessor — V1 logits processor that drives in-flight
beam search, matching the HF / V0 algorithm.

The LP is the per-step decision-maker. It owns the canonical per-beam
state (token sequences, cumulative logprobs) and, each forward pass:

  1. Pools `beam_width * 2*beam_width` candidates across all live beams.
  2. Sorts by RAW cumulative logprob (cum + token_logprob), desc.
  3. Walks the top `2*beam_width`: an EOS at rank < beam_width becomes a
     completed hypothesis (not a live slot); otherwise the candidate
     becomes a surviving beam. Stops at `beam_width` survivors.
  4. Assigns survivors to physical slots, minimizing KV moves by default
     (a survivor whose parent is its own slot stays put; the rest fork).
     `VLLM_BEAM_RANK_ORDER=1` is a diagnostic mode that instead rebuilds
     active physical slots in survivor rank order, closer to V0's
     `sequence_group.seqs = new_seqs` replacement.
  5. Masks each slot's logits to its assigned token so the engine samples
     it, and records a fork plan + completions for the scheduler.

Beam-init trick (matches HF `beam_scores[:, 1:] = -1e9`): beams start with
cum[0]=0 and cum[1:]=-inf so the first real selection draws `beam_width`
*distinct* tokens from beam 0 alone; without it every slot would get the
same top token.

The KV side of forks (sharing a parent's prefix blocks, allocating one new
block) is executed by BeamSearchScheduler reading `fork_src`/`completions`.
"""
from __future__ import annotations

import os
import builtins
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


_DEBUG = bool(int(os.getenv("VLLM_BEAM_DEBUG", "0")))
_TRACE = bool(int(os.getenv("VLLM_BEAM_TRACE", "0")))
_TRACE_STEPS = int(os.getenv("VLLM_BEAM_TRACE_STEPS", "12"))
_RANK_ORDER = bool(int(os.getenv("VLLM_BEAM_RANK_ORDER", "0")))
_TENSOR_POOL_TOPK = bool(int(os.getenv("VLLM_BEAM_TENSOR_POOL_TOPK", "0")))
_HOTPATH_TIMING = bool(int(os.getenv("VLLM_V1_HOTPATH_TIMING", "0")))
_HOTPATH_MIN_MS = float(os.getenv("VLLM_V1_HOTPATH_TIMING_MIN_MS", "0.0"))
_NEG_INF = float("-inf")
_INIT_NEG = -1e9  # HF beam-init sentinel (finite so arithmetic stays sane)
_RUNTIME_ATTR = "_vllm_beam_search_runtime_groups"


def _hotpath_log(component: str, dt_s: float, **fields: object) -> None:
    if not _HOTPATH_TIMING:
        return
    dt_ms = dt_s * 1000.0
    if dt_ms < _HOTPATH_MIN_MS:
        return
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    print(
        f"[V1_HOTPATH] component={component} dt_ms={dt_ms:.3f}"
        + (f" {extra}" if extra else ""),
        flush=True,
    )


def _banned_ngram_tokens(seq: list[int], n: int) -> set[int]:
    """Tokens that would complete a repeated n-gram given the suffix of seq."""
    if n <= 0 or len(seq) < n:
        return set()
    prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
    banned: set[int] = set()
    for i in range(len(seq) - (n - 1)):
        if tuple(seq[i:i + n - 1]) == prefix:
            j = i + n - 1
            if j < len(seq):
                banned.add(seq[j])
    return banned


@dataclass
class _BeamGroupRuntime:
    """Per-group beam state, owned by the LP (single source of truth)."""

    beam_width: int
    eos_token_id: int | None
    no_repeat_ngram_size: int = 0
    # Decoder prompt tokens (e.g. [decoder_start]); part of the no_repeat
    # n-gram window to match V0's prompt+generated scan.
    prompt_tokens: list[int] = field(default_factory=list)

    # Physical-slot bookkeeping.
    batch_indices: list[int | None] = field(default_factory=list)

    # Canonical per-beam state (index = physical slot).
    tokens: list[list[int]] = field(default_factory=list)  # generated tokens
    cum: list[float] = field(default_factory=list)
    decode_step: int = 0

    # Per-step plan, drained by the scheduler in update_from_output.
    #   fork_src[slot] = parent slot this beam's KV must follow this step.
    #     == slot  -> beam continued in place, no KV move.
    #     != slot  -> rebase this slot's KV onto fork_src[slot]'s prefix.
    fork_src: list[int] = field(default_factory=list)
    # Completed EOS hypotheses: (generated_tokens_incl_eos, cum_logprob).
    completions: list[tuple[list[int], float]] = field(default_factory=list)
    # Physical slots still participating in V0's active beam set. Slots can
    # shrink when EOS candidates occupy the top beam ranks.
    active: list[bool] = field(default_factory=list)
    # Slots that became inactive during the most recent apply(); the scheduler
    # drains this list and finishes the corresponding child requests.
    inactive_slots: list[int] = field(default_factory=list)
    # Snapshot of each beam's pre-step token length (= L), for KV rebase.
    prefix_len: int = 0

    def present_slots(self, batch_size: int) -> list[tuple[int, int]]:
        out = []
        for slot, bi in enumerate(self.batch_indices):
            if (
                bi is not None
                and bi < batch_size
                and (not self.active or self.active[slot])
            ):
                out.append((slot, bi))
        return out


class BeamSearchLogitsProcessor(LogitsProcessor):
    """V1 LogitsProcessor implementing HF/V0 in-flight beam search.

    Beam-children carry in `sampling_params.extra_args`:
        _beam_group_id, _beam_index, _beam_width, _beam_eos_token_id
    """

    _is_beam_search_logitsproc = True
    _singleton: "BeamSearchLogitsProcessor | None" = None

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        self.device = device
        self.pin_memory = is_pin_memory
        BeamSearchLogitsProcessor._singleton = self

        self.groups: dict[str, _BeamGroupRuntime] = {}
        self.row_to_group: dict[int, tuple[str, int]] = {}
        setattr(builtins, _RUNTIME_ATTR, self.groups)

    def _publish_runtime(self) -> None:
        setattr(builtins, _RUNTIME_ATTR, self.groups)

    def is_argmax_invariant(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Batch tracking
    # ------------------------------------------------------------------

    def _register_added(
        self, batch_idx: int, sampling_params, prompt_tok_ids
    ) -> None:
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
            gr = _BeamGroupRuntime(
                beam_width=beam_width,
                eos_token_id=int(eos) if eos is not None else None,
                no_repeat_ngram_size=no_repeat,
                prompt_tokens=list(prompt_tok_ids or []),
                batch_indices=[None] * beam_width,
                tokens=[[] for _ in range(beam_width)],
                cum=[_INIT_NEG] * beam_width,
                fork_src=list(range(beam_width)),
                active=[True] * beam_width,
            )
            gr.cum[0] = 0.0  # beam-init trick
            self.groups[gid] = gr
        gr.batch_indices[beam_idx] = batch_idx
        self.row_to_group[batch_idx] = (gid, beam_idx)
        self._publish_runtime()

    def _unregister_row(self, batch_idx: int) -> None:
        entry = self.row_to_group.pop(batch_idx, None)
        if entry is None:
            return
        gid, beam_idx = entry
        gr = self.groups.get(gid)
        if gr is None:
            return
        if 0 <= beam_idx < len(gr.batch_indices):
            gr.batch_indices[beam_idx] = None
        if all(bi is None for bi in gr.batch_indices):
            self.groups.pop(gid, None)
        self._publish_runtime()

    def _move_row(self, src: int, dst: int, swap: bool) -> None:
        src_entry = self.row_to_group.get(src)
        dst_entry = self.row_to_group.get(dst) if swap else None
        if src_entry is not None:
            gid, beam_idx = src_entry
            self.groups[gid].batch_indices[beam_idx] = dst
            self.row_to_group[dst] = src_entry
        else:
            self.row_to_group.pop(dst, None)
        if swap:
            if dst_entry is not None:
                gid2, beam_idx2 = dst_entry
                self.groups[gid2].batch_indices[beam_idx2] = src
                self.row_to_group[src] = dst_entry
            else:
                self.row_to_group.pop(src, None)
        elif src_entry is not None:
            self.row_to_group.pop(src, None)
        self._publish_runtime()

    def update_state(self, batch_update: Optional[BatchUpdate]) -> None:
        if batch_update is None:
            return
        for idx in batch_update.removed:
            self._unregister_row(idx)
        for added in batch_update.added:
            idx, params, prompt_ids, _output_tok_ids = added
            self._register_added(idx, params, prompt_ids)
        for adx, bdx, direct in batch_update.moved:
            self._move_row(adx, bdx, swap=(direct.name == "SWAP"))

    # ------------------------------------------------------------------
    # Per-step apply()
    # ------------------------------------------------------------------

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.groups:
            return logits
        t0 = time.perf_counter() if _HOTPATH_TIMING else 0.0
        groups_seen = 0
        slots_seen = 0
        self._publish_runtime()
        batch_size = logits.shape[0]
        select_jobs: list[tuple[_BeamGroupRuntime, list[tuple[int, int]]]] = []
        for gid, gr in self.groups.items():
            present = gr.present_slots(batch_size)
            if not present:
                continue
            groups_seen += 1
            slots_seen += len(present)
            select_jobs.append((gr, present))

        if select_jobs:
            t_stage = time.perf_counter() if _HOTPATH_TIMING else 0.0
            all_batch_indices: list[int] = []
            row_meta: list[tuple[_BeamGroupRuntime, int]] = []
            max_k = 0
            for gr, present in select_jobs:
                max_k = max(max_k, 2 * gr.beam_width)
                for slot, bi in present:
                    row_meta.append((gr, slot))
                    all_batch_indices.append(bi)
            if _HOTPATH_TIMING:
                _hotpath_log(
                    "beam_logits_prepare_rows",
                    time.perf_counter() - t_stage,
                    groups=len(select_jobs),
                    rows=len(row_meta),
                )

            t_stage = time.perf_counter() if _HOTPATH_TIMING else 0.0
            logprobs_all = torch.log_softmax(
                logits[all_batch_indices].float(), dim=-1
            )
            if _HOTPATH_TIMING:
                _hotpath_log(
                    "beam_logits_log_softmax",
                    time.perf_counter() - t_stage,
                    rows=len(row_meta),
                )

            t_stage = time.perf_counter() if _HOTPATH_TIMING else 0.0
            for row_idx, (gr, slot) in enumerate(row_meta):
                if gr.no_repeat_ngram_size <= 0:
                    continue
                # V0's NoRepeatNgramLogitsProcessor scans prompt + generated
                # (i.e. includes the decoder prompt / decoder_start), so use
                # the same window here.
                window = gr.prompt_tokens + gr.tokens[slot]
                for t in _banned_ngram_tokens(window, gr.no_repeat_ngram_size):
                    logprobs_all[row_idx, t] = _NEG_INF
            if _HOTPATH_TIMING:
                _hotpath_log(
                    "beam_logits_no_repeat_mask",
                    time.perf_counter() - t_stage,
                    rows=len(row_meta),
                )

            t_stage = time.perf_counter() if _HOTPATH_TIMING else 0.0
            k = min(logprobs_all.shape[1], max_k)
            vals_all, ids_all = torch.topk(logprobs_all, k, dim=-1)
            if _HOTPATH_TIMING:
                _hotpath_log(
                    "beam_logits_topk",
                    time.perf_counter() - t_stage,
                    rows=len(row_meta),
                    k=k,
                )

            t_stage = time.perf_counter() if _HOTPATH_TIMING else 0.0
            row_offset = 0
            if _TENSOR_POOL_TOPK:
                vals_rows = None
                ids_rows = None
            else:
                vals_rows = vals_all.tolist()
                ids_rows = ids_all.tolist()
                if _HOTPATH_TIMING:
                    _hotpath_log(
                        "beam_logits_topk_transfer",
                        time.perf_counter() - t_stage,
                        rows=len(row_meta),
                        k=k,
                    )
                t_stage = time.perf_counter() if _HOTPATH_TIMING else 0.0

            forced_tokens: list[tuple[int, int]] = []
            for gr, present in select_jobs:
                row_count = len(present)
                top_candidates = None
                if _TENSOR_POOL_TOPK:
                    top_candidates = self._pooled_top_candidates(
                        gr,
                        present,
                        vals_all[row_offset:row_offset + row_count],
                        ids_all[row_offset:row_offset + row_count],
                    )
                self._apply_select(
                    logits,
                    gr,
                    present,
                    (
                        None
                        if vals_rows is None
                        else vals_rows[row_offset:row_offset + row_count]
                    ),
                    (
                        None
                        if ids_rows is None
                        else ids_rows[row_offset:row_offset + row_count]
                    ),
                    top_candidates,
                    forced_tokens,
                )
                gr.decode_step += 1
                row_offset += row_count
            if _HOTPATH_TIMING:
                _hotpath_log(
                    "beam_logits_select_python",
                    time.perf_counter() - t_stage,
                    groups=len(select_jobs),
                    rows=len(row_meta),
                )

            if forced_tokens:
                t_stage = time.perf_counter() if _HOTPATH_TIMING else 0.0
                rows = torch.as_tensor(
                    [bi for bi, _tok in forced_tokens],
                    dtype=torch.long,
                    device=logits.device,
                )
                toks = torch.as_tensor(
                    [tok for _bi, tok in forced_tokens],
                    dtype=torch.long,
                    device=logits.device,
                )
                logits.index_fill_(0, rows, _NEG_INF)
                logits[rows, toks] = 0.0
                if _HOTPATH_TIMING:
                    _hotpath_log(
                        "beam_logits_mask_selected",
                        time.perf_counter() - t_stage,
                        rows=len(forced_tokens),
                    )

        if _HOTPATH_TIMING and groups_seen:
            _hotpath_log(
                "beam_logits_apply",
                time.perf_counter() - t0,
                groups=groups_seen,
                slots=slots_seen,
                batch=batch_size,
            )
        return logits

    def _apply_select(
        self,
        logits,
        gr,
        present,
        vals_all: list[list[float]] | torch.Tensor | None = None,
        ids_all: list[list[int]] | torch.Tensor | None = None,
        top_candidates: list[tuple[float, int, int, float]] | None = None,
        forced_tokens: list[tuple[int, int]] | None = None,
    ) -> None:
        bw = gr.beam_width
        eos = gr.eos_token_id
        gr.prefix_len = len(gr.tokens[present[0][0]])  # L (lockstep)
        gr.completions = []
        gr.inactive_slots = []

        present_slots = [slot for slot, _bi in present]
        if top_candidates is None and (vals_all is None or ids_all is None):
            present_batch_indices = [bi for _slot, bi in present]
            logprobs_all = torch.log_softmax(
                logits[present_batch_indices].float(), dim=-1
            )
            if gr.no_repeat_ngram_size > 0:
                for row_idx, slot in enumerate(present_slots):
                    window = gr.prompt_tokens + gr.tokens[slot]
                    for t in _banned_ngram_tokens(
                        window, gr.no_repeat_ngram_size
                    ):
                        logprobs_all[row_idx, t] = _NEG_INF
            k = min(logprobs_all.shape[1], 2 * bw)
            vals_all, ids_all = torch.topk(logprobs_all, k, dim=-1)
            vals_all = vals_all.tolist()
            ids_all = ids_all.tolist()

        if top_candidates is None:
            # Gather candidates from each live beam. `apply()` normally
            # computes top-k logprobs for all active groups in one batched
            # tensor op; keep a local fallback for direct/unit calls.
            k = 2 * bw
            candidates = []
            for row_idx, slot in enumerate(present_slots):
                vals = vals_all[row_idx][:k]
                ids = ids_all[row_idx][:k]
                for lp, tok in zip(vals, ids):
                    if lp == _NEG_INF:
                        continue
                    candidates.append((gr.cum[slot] + lp, slot, tok, lp))

            # Stable score-only sort preserves the row-major tie behavior of
            # the candidate collection order.
            candidates.sort(key=lambda x: x[0], reverse=True)
            top = candidates[:2 * bw]
        else:
            top = top_candidates

        survivors: list[tuple[int, int, float]] = []  # (src_slot, tok, cum2)
        for rank, (cum2, slot, tok, lp) in enumerate(top):
            if eos is not None and tok == eos:
                if rank < bw:
                    gr.completions.append((gr.tokens[slot] + [tok], cum2))
            else:
                survivors.append((slot, tok, cum2))
            if len(survivors) >= bw:
                break

        if _TRACE and gr.decode_step <= _TRACE_STEPS:
            print(
                f"[BEAM_TRACE plugin] step={gr.decode_step} "
                f"L={gr.prefix_len} present={present} "
                f"pre_cum={[round(c, 6) for c in gr.cum]}",
                flush=True,
            )
            for rank, (cum2, slot, tok, lp) in enumerate(top):
                is_eos = eos is not None and tok == eos
                seq = gr.tokens[slot] + [tok]
                print(
                    f"[BEAM_TRACE plugin] cand rank={rank} src={slot} "
                    f"tok={tok} lp={lp:.6f} cum={cum2:.6f} "
                    f"eos={is_eos} seq={seq}",
                    flush=True,
                )
            for idx, (slot, tok, cum2) in enumerate(survivors):
                print(
                    f"[BEAM_TRACE plugin] keep idx={idx} src={slot} "
                    f"tok={tok} cum={cum2:.6f} "
                    f"seq={gr.tokens[slot] + [tok]}",
                    flush=True,
                )
            for idx, (tokens, cum2) in enumerate(gr.completions):
                print(
                    f"[BEAM_TRACE plugin] complete idx={idx} "
                    f"tok={tokens[-1] if tokens else None} "
                    f"cum={cum2:.6f} len={len(tokens)} seq={tokens}",
                    flush=True,
                )

        if not survivors:
            # V0 finishes the group when no continuing beam survives.
            for slot, bi in present:
                gr.active[slot] = False
                gr.inactive_slots.append(slot)
                tok = eos if eos is not None else 0
                self._force_token(logits, bi, tok, forced_tokens)
            gr.fork_src = list(range(bw))
            return

        # Assign survivors to physical slots. The default minimizes KV moves;
        # rank-order mode is a fidelity diagnostic for V0's list replacement.
        assign: list[tuple[int, int, float] | None] = [None] * bw
        active_slots = {slot for slot, _ in present}
        if _RANK_ORDER:
            for s, slot in zip(survivors, sorted(active_slots)):
                assign[slot] = s
        else:
            free = set(active_slots)
            leftover = []
            for s in survivors:
                src = s[0]
                if src in free:
                    assign[src] = s
                    free.discard(src)
                else:
                    leftover.append(s)
            free_slots = sorted(free)
            for s, slot in zip(leftover, free_slots):
                assign[slot] = s

        new_tokens = [list(tokens) for tokens in gr.tokens]
        new_cum = list(gr.cum)
        fork_src = list(range(bw))
        for slot in range(bw):
            if assign[slot] is None:
                if slot in active_slots:
                    gr.active[slot] = False
                    gr.inactive_slots.append(slot)
                    bi = gr.batch_indices[slot]
                    if bi is not None and bi < logits.shape[0]:
                        tok = eos if eos is not None else 0
                        self._force_token(logits, bi, tok, forced_tokens)
                continue

            src, tok, cum2 = assign[slot]
            gr.active[slot] = True
            new_tokens[slot] = gr.tokens[src] + [tok]
            new_cum[slot] = cum2
            fork_src[slot] = src
            bi = gr.batch_indices[slot]
            if bi is not None and bi < logits.shape[0]:
                self._force_token(logits, bi, tok, forced_tokens)

        if _DEBUG:
            plan = " ".join(
                f"s{slot}<-b{fork_src[slot]}:{new_tokens[slot][-1]}"
                f"(lp={new_cum[slot] - gr.cum[assign[slot][0]]:+.3f})"
                + ("*" if fork_src[slot] != slot else "")
                for slot in range(bw)
                if assign[slot] is not None
            )
            inactive = (
                f" inactive={gr.inactive_slots}" if gr.inactive_slots else ""
            )
            print(f"[BEAM] gid step={gr.decode_step} L={gr.prefix_len} "
                  f"{plan} completions={len(gr.completions)} "
                  f"cum={['%.2f' % c for c in new_cum]}{inactive}",
                  flush=True)

        gr.tokens = new_tokens
        gr.cum = new_cum
        gr.fork_src = fork_src

    def _force_token(
        self,
        logits: torch.Tensor,
        batch_idx: int,
        token_id: int,
        forced_tokens: list[tuple[int, int]] | None,
    ) -> None:
        if forced_tokens is None:
            logits[batch_idx, :].fill_(_NEG_INF)
            logits[batch_idx, token_id] = 0.0
            return
        forced_tokens.append((batch_idx, token_id))

    def _pooled_top_candidates(
        self,
        gr: _BeamGroupRuntime,
        present: list[tuple[int, int]],
        vals_all: torch.Tensor,
        ids_all: torch.Tensor,
    ) -> list[tuple[float, int, int, float]]:
        bw = gr.beam_width
        row_count = len(present)
        k = min(vals_all.shape[1], 2 * bw)
        vals = vals_all[:, :k]
        ids = ids_all[:, :k]
        bases = torch.tensor(
            [gr.cum[slot] for slot, _bi in present],
            dtype=vals.dtype,
            device=vals.device,
        ).unsqueeze(1)
        scores = vals + bases
        top_n = min(2 * bw, scores.numel())
        top_scores, flat_indices = torch.topk(scores.flatten(), top_n)
        flat_list = flat_indices.tolist()
        score_list = top_scores.tolist()
        vals_rows = vals.tolist()
        ids_rows = ids.tolist()
        out: list[tuple[float, int, int, float]] = []
        for score, flat_idx in zip(score_list, flat_list):
            row_idx = flat_idx // k
            col_idx = flat_idx % k
            if row_idx >= row_count:
                continue
            slot = present[row_idx][0]
            lp = vals_rows[row_idx][col_idx]
            if lp == _NEG_INF:
                continue
            tok = ids_rows[row_idx][col_idx]
            out.append((score, slot, tok, lp))
        return out
