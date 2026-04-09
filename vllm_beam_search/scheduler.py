"""Beam search scheduler plugin for vLLM v1.

Uses copy-on-write block table forking via prefix caching
to avoid KV cache recomputation when beams fork. Sharing is at
block granularity — at most block_size-1 tokens are recomputed per fork.

Usage:
    vllm serve <model> \
        --scheduler-cls vllm_beam_search.scheduler.BeamSearchScheduler \
        --enable-prefix-caching

    client:
        SamplingParams(extra_args={"beam_width": 4}, max_tokens=128)
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Iterable

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler, check_stop
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import Request, RequestStatus

from vllm_beam_search.beam_state import BeamGroup, BeamState

if TYPE_CHECKING:
    from vllm.multimodal.registry import MultiModalRegistry
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


class BeamSearchScheduler(Scheduler):

    def __init__(
        self,
        vllm_config: Any,
        kv_cache_config: "KVCacheConfig",
        structured_output_manager: "StructuredOutputManager",
        block_size: int,
        mm_registry: "MultiModalRegistry | None" = None,
        include_finished_set: bool = False,
        log_stats: bool = False,
        **kwargs: Any,
    ) -> None:
        if mm_registry is None:
            from vllm.multimodal import MULTIMODAL_REGISTRY
            mm_registry = MULTIMODAL_REGISTRY
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
            **kwargs,
        )

        if not self.cache_config.enable_prefix_caching:
            raise ValueError(
                "BeamSearchScheduler requires prefix caching. "
                "Set --enable-prefix-caching in the vLLM server config."
            )

        # original_request_id -> BeamGroup
        self.beam_groups: dict[str, BeamGroup] = {}
        # beam_child_req_id -> original_request_id
        self.beam_to_group: dict[str, str] = {}
        # Block hasher captured from first request (shared across all).
        self._block_hasher: Any = None

    # ------------------------------------------------------------------
    # add_request
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        beam_width = self._get_beam_width(request)
        if beam_width is None:
            super().add_request(request)
            return

        if self._block_hasher is None:
            self._block_hasher = request._block_hasher

        length_penalty = float(
            (request.sampling_params.extra_args or {}).get("length_penalty", 1.0)
        )
        group = BeamGroup(
            original_request_id=request.request_id,
            original_request=request,
            beam_width=beam_width,
            length_penalty=length_penalty,
        )
        self.beam_groups[request.request_id] = group

        # Create ONE initial child. After its first decode we expand to
        # beam_width beams via the top-k logprobs.
        child = self._make_beam_child(
            group=group,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            client_index=request.client_index,
        )
        beam = BeamState(request=child, score=0.0)
        group.active_beams[child.request_id] = beam
        self.beam_to_group[child.request_id] = request.request_id
        super().add_request(child)

    # ------------------------------------------------------------------
    # finish_requests — cascade abort to all beam children
    # ------------------------------------------------------------------

    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
    ) -> list[tuple[str, int]]:
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        if request_ids is not None:
            expanded: list[str] = []
            for rid in request_ids:
                expanded.append(rid)
                group = self.beam_groups.get(rid)
                if group is not None:
                    expanded.extend(group.active_beams.keys())
            request_ids = expanded
        result = super().finish_requests(request_ids, finished_status)
        if request_ids is not None:
            for rid in request_ids:
                self._cleanup_group_if_needed(rid)
        return result

    # ------------------------------------------------------------------
    # update_from_output — core beam search logic
    # ------------------------------------------------------------------

    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, EngineCoreOutputs]:
        # Pull beam children out of num_scheduled_tokens so super()
        # does not emit EngineCoreOutputs for them.
        beam_scheduled: dict[str, int] = {}
        for req_id in list(scheduler_output.num_scheduled_tokens):
            if req_id in self.beam_to_group:
                beam_scheduled[req_id] = (
                    scheduler_output.num_scheduled_tokens.pop(req_id)
                )

        # Let the parent handle non-beam requests.
        outputs = super().update_from_output(scheduler_output, model_runner_output)

        # Restore so metrics / assertions aren't confused.
        scheduler_output.num_scheduled_tokens.update(beam_scheduled)

        if beam_scheduled:
            self._process_beam_outputs(
                model_runner_output, beam_scheduled, outputs
            )
        return outputs

    # ------------------------------------------------------------------
    # Beam output processing
    # ------------------------------------------------------------------

    def _process_beam_outputs(
        self,
        model_runner_output: "ModelRunnerOutput",
        beam_scheduled: dict[str, int],
        outputs: dict[int, EngineCoreOutputs],
    ) -> None:
        sampled = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs

        # Bucket scheduled beams by group.
        groups_in_step: dict[str, list[str]] = defaultdict(list)
        for beam_id in beam_scheduled:
            orig_id = self.beam_to_group.get(beam_id)
            if orig_id is not None:
                groups_in_step[orig_id].append(beam_id)

        for orig_id, sched_ids in groups_in_step.items():
            group = self.beam_groups.get(orig_id)
            if group is None:
                continue

            # Gather (source_beam_id, candidate_token, logprob) tuples.
            candidates: list[tuple[str, int, float]] = []
            prefill_only = True

            for beam_id in sched_ids:
                beam = group.active_beams.get(beam_id)
                if beam is None or beam.finished:
                    continue

                req_idx = model_runner_output.req_id_to_index.get(beam_id)
                if req_idx is None:
                    continue

                gen_tokens = sampled[req_idx] if sampled else []
                num_sched = beam_scheduled[beam_id]
                is_prefill = num_sched > 1

                if is_prefill:
                    # Still prefilling — just append the generated token
                    # (if any) and skip beam scoring this step.
                    if gen_tokens:
                        tok = gen_tokens[0]
                        beam.request.append_output_token_ids(tok)
                        beam.token_ids.append(tok)
                    continue

                prefill_only = False
                if not gen_tokens:
                    continue

                # Decode step: collect candidates from logprobs.
                if logprobs is not None:
                    lp = logprobs.slice_request(req_idx, 1)
                    seen = set()
                    for j in range(lp.logprob_token_ids.shape[1]):
                        tok = int(lp.logprob_token_ids[0, j])
                        if tok in seen:
                            continue
                        seen.add(tok)
                        candidates.append(
                            (beam_id, tok, float(lp.logprobs[0, j]))
                        )
                else:
                    candidates.append((beam_id, gen_tokens[0], 0.0))

            if prefill_only or not candidates:
                continue

            if len(group.active_beams) == 1:
                self._expand_initial(group, candidates, outputs)
            else:
                self._beam_step(group, candidates, outputs)

    # ------------------------------------------------------------------
    # Initial expansion (1 → beam_width)
    # ------------------------------------------------------------------

    def _expand_initial(
        self,
        group: BeamGroup,
        candidates: list[tuple[str, int, float]],
        outputs: dict[int, EngineCoreOutputs],
    ) -> None:
        candidates.sort(key=lambda c: c[2], reverse=True)
        top_k = candidates[: group.beam_width]

        src_id = top_k[0][0]
        src_beam = group.active_beams[src_id]

        # First candidate: the initial beam continues with its sampled token.
        tok0, lp0 = top_k[0][1], top_k[0][2]
        src_beam.request.append_output_token_ids(tok0)
        src_beam.token_ids.append(tok0)
        src_beam.score = lp0
        self._check_beam_stop(group, src_id)

        # Remaining candidates: fork from the initial beam's prompt.
        orig_prompt = src_beam.request.prompt_token_ids
        remaining_tokens = src_beam.request.max_tokens - 1

        for _, tok, lp in top_k[1:]:
            child_prompt = list(orig_prompt) + [tok]
            child = self._make_beam_child(
                group, child_prompt, remaining_tokens,
                group.original_request.client_index,
            )
            bs = BeamState(request=child, score=lp, token_ids=[tok])
            group.active_beams[child.request_id] = bs
            self.beam_to_group[child.request_id] = group.original_request_id
            super().add_request(child)
            self._check_beam_stop(group, child.request_id)

        self._maybe_complete_group(group, outputs)

    # ------------------------------------------------------------------
    # Regular beam step (score K*K → keep top K)
    # ------------------------------------------------------------------

    def _beam_step(
        self,
        group: BeamGroup,
        candidates: list[tuple[str, int, float]],
        outputs: dict[int, EngineCoreOutputs],
    ) -> None:
        # Build scored list: (src_beam_id, token, raw_lp, norm_score).
        scored: list[tuple[str, int, float, float]] = []
        for src_id, tok, lp in candidates:
            beam = group.active_beams.get(src_id)
            if beam is None:
                continue
            cum = beam.score + lp
            ns = group.normalized_score(cum, len(beam.token_ids) + 1)
            scored.append((src_id, tok, lp, ns))

        scored.sort(key=lambda c: c[3], reverse=True)
        selected = scored[: group.beam_width]

        # Figure out which beams continue in-place vs. need forking.
        # "top-1 token" per beam = the first candidate for that beam in the
        # sorted candidates list (= the sampled token with temp=0).
        top1_per_beam: dict[str, int] = {}
        for src_id, tok, _lp in sorted(
            candidates, key=lambda c: c[2], reverse=True
        ):
            if src_id not in top1_per_beam:
                top1_per_beam[src_id] = tok

        continuing: dict[str, tuple[int, float]] = {}  # beam_id -> (tok, lp)
        forks: list[tuple[str, int, float]] = []       # (src_id, tok, lp)

        for src_id, tok, lp, _ in selected:
            if (
                src_id not in continuing
                and top1_per_beam.get(src_id) == tok
            ):
                continuing[src_id] = (tok, lp)
            else:
                forks.append((src_id, tok, lp))

        # --- 1. Append sampled token to continuing beams. ----
        for beam_id, (tok, lp) in continuing.items():
            beam = group.active_beams[beam_id]
            beam.request.append_output_token_ids(tok)
            beam.token_ids.append(tok)
            beam.score += lp

        # --- 2. Create forks. ---
        orig_prompt = group.original_request.prompt_token_ids
        orig_max = group.original_request.max_tokens
        new_fork_ids: set[str] = set()

        for src_id, tok, lp in forks:
            src_beam = group.active_beams.get(src_id)
            if src_beam is None:
                continue
            child_prompt = (
                list(orig_prompt) + list(src_beam.token_ids) + [tok]
            )
            already = len(src_beam.token_ids) + 1
            child_max = orig_max - already
            if child_max <= 0:
                continue

            child = self._make_beam_child(
                group, child_prompt, child_max,
                group.original_request.client_index,
            )
            new_fork_ids.add(child.request_id)
            new_tokens = list(src_beam.token_ids) + [tok]
            bs = BeamState(
                request=child,
                score=src_beam.score + lp,
                token_ids=new_tokens,
            )
            group.active_beams[child.request_id] = bs
            self.beam_to_group[child.request_id] = group.original_request_id
            super().add_request(child)

        # --- 3. Kill unselected beams. ---
        keep = set(continuing.keys()) | new_fork_ids
        # Also keep beams that are finished (tracked for best_completed).
        keep |= {
            bid for bid, bs in group.active_beams.items() if bs.finished
        }
        for bid in [b for b in group.active_beams if b not in keep]:
            self._kill_beam(group, bid)

        # --- 4. Check stop conditions. ---
        for bid in list(continuing.keys()):
            self._check_beam_stop(group, bid)
        for bid in new_fork_ids:
            self._check_beam_stop(group, bid)

        self._maybe_complete_group(group, outputs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_beam_width(request: Request) -> int | None:
        if request.sampling_params is None:
            return None
        extra = request.sampling_params.extra_args
        if extra is None:
            return None
        bw = extra.get("beam_width")
        return int(bw) if bw is not None else None

    def _make_beam_child(
        self,
        group: BeamGroup,
        prompt_token_ids: list[int],
        max_tokens: int,
        client_index: int,
    ) -> Request:
        beam_id = group.next_beam_id()
        p = copy.copy(group.original_request.sampling_params)
        p.logprobs = group.beam_width - 1
        p.temperature = 0.0
        p.n = 1
        p.max_tokens = max(max_tokens, 1)
        if p.extra_args:
            p.extra_args = {
                k: v for k, v in p.extra_args.items()
                if k not in ("beam_width", "length_penalty")
            }
            if not p.extra_args:
                p.extra_args = None

        return Request(
            request_id=beam_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=p,
            pooling_params=None,
            client_index=client_index,
            block_hasher=self._block_hasher,
        )

    def _check_beam_stop(self, group: BeamGroup, beam_id: str) -> None:
        beam = group.active_beams.get(beam_id)
        if beam is None or beam.finished:
            return
        # check_stop reads output_token_ids[-1]; skip if no output yet
        # (newly forked children have the candidate in their prompt, not output).
        if len(beam.request._output_token_ids) == 0:
            return
        if not check_stop(beam.request, self.max_model_len):
            return

        beam.finished = True
        ns = group.normalized_score(beam.score, len(beam.token_ids))
        if group.best_completed is None or ns > group.normalized_score(
            group.best_completed.score, len(group.best_completed.token_ids)
        ):
            group.best_completed = beam

        # Remove from running so it's not scheduled again, free blocks.
        self._kill_beam(group, beam_id, remove_from_group=False)

    def _kill_beam(
        self,
        group: BeamGroup,
        beam_id: str,
        remove_from_group: bool = True,
    ) -> None:
        if remove_from_group:
            group.active_beams.pop(beam_id, None)
        self.beam_to_group.pop(beam_id, None)
        req = self.requests.get(beam_id)
        if req is not None and not req.is_finished():
            # Use the parent's finish_requests which properly removes
            # from BOTH running and waiting queues before freeing.
            Scheduler.finish_requests(
                self, beam_id, RequestStatus.FINISHED_STOPPED
            )

    def _maybe_complete_group(
        self,
        group: BeamGroup,
        outputs: dict[int, EngineCoreOutputs],
    ) -> None:
        all_done = group.all_finished or group.num_active == 0
        if not all_done and not group.can_early_stop():
            return

        # Pick the best beam.
        best = group.best_completed
        for bs in group.active_beams.values():
            if bs.finished:
                continue
            ns = group.normalized_score(bs.score, len(bs.token_ids))
            if best is None or ns > group.normalized_score(
                best.score, len(best.token_ids)
            ):
                best = bs

        if best is None:
            logger.warning(
                "Beam group %s completed with no valid beams.",
                group.original_request_id,
            )
            return

        # Emit result for the original (user-facing) request.
        ci = group.original_request.client_index
        eco = outputs.get(ci)
        if eco is None:
            eco = EngineCoreOutputs()
            outputs[ci] = eco
        eco.outputs.append(
            EngineCoreOutput(
                request_id=group.original_request_id,
                new_token_ids=list(best.token_ids),
                finish_reason=FinishReason.STOP,
            )
        )

        # Clean up remaining beams.
        for bid in list(group.active_beams.keys()):
            self._kill_beam(group, bid)
        self._cleanup_group(group.original_request_id)

    def _cleanup_group_if_needed(self, request_id: str) -> None:
        orig_id = self.beam_to_group.pop(request_id, None)
        if orig_id is not None:
            group = self.beam_groups.get(orig_id)
            if group is not None:
                group.active_beams.pop(request_id, None)

    def _cleanup_group(self, original_request_id: str) -> None:
        group = self.beam_groups.pop(original_request_id, None)
        if group is None:
            return
        for bid in list(group.active_beams.keys()):
            self.beam_to_group.pop(bid, None)
        group.active_beams.clear()

    # ------------------------------------------------------------------
    # Request counting — account for virtual beam group requests
    # ------------------------------------------------------------------

    def get_num_unfinished_requests(self) -> int:
        return super().get_num_unfinished_requests() + len(self.beam_groups)

    def has_requests(self) -> bool:
        return super().has_requests() or bool(self.beam_groups)
