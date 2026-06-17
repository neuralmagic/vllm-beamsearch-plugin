"""BeamSearchScheduler — MRV2 async plugin doing in-flight beam search.

Matches the HF / V0 algorithm. At `add_request` we materialize
`beam_width` sibling Request objects (one per beam) and add them all to
the base scheduler. The MRV2 sampler makes per-step beam decisions and
emits `BeamTransition` records; this scheduler reconciles the CPU-side
request/KV bookkeeping when those async outputs arrive.

Per step (`update_from_output`):
  1. Suppress the beam-children outputs from the engine stream.
  2. Drain the LP's completed EOS hypotheses into the BeamGroup.
  3. If `beam_width` hypotheses are complete -> early stop (HF
     early_stopping=True): emit one EngineCoreOutput for the original
     request id and finish all sibling children.
  4. Otherwise execute the sampler's fork plan: for each slot whose
     `fork_src != slot`, rebase that slot's decoder (self-attention)
     KV onto the parent's shareable full prefix blocks. For larger blocks,
     the worker copies the source partial block into the destination's
     private partial block. Cross-attention (encoder) KV is identical
     across beams and left untouched.
"""
from __future__ import annotations

import copy
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import CrossAttentionManager
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import Request, RequestStatus

from .beam_types import BeamTransition
from .beam_state import BeamGroup, CompletedBeam

_BEAM_TRANSITIONS_OUTPUT = "vllm_beam_search.transitions"

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)


def _install_worker_history_rewrite_hooks() -> None:
    _patch_flash_attn_hopper_block_size_one()
    _patch_mrv2_model_state_beam_sampler()
    _patch_mrv2_gpu_model_runner_history_rewrites()


def _is_beam_scheduler_config(vllm_config: Any) -> bool:
    scheduler_cls = getattr(vllm_config.scheduler_config, "scheduler_cls", None)
    if scheduler_cls is None:
        return False
    if isinstance(scheduler_cls, str):
        return scheduler_cls == "vllm_beam_search.scheduler.BeamSearchScheduler"
    return (
        getattr(scheduler_cls, "__module__", None) == __name__
        and getattr(scheduler_cls, "__qualname__", "") == "BeamSearchScheduler"
    )


def _patch_mrv2_model_state_beam_sampler() -> None:
    """Install the beam sampler on any MRV2 model state using BeamScheduler."""
    try:
        import vllm.v1.worker.gpu.model_runner as model_runner_module
        import vllm.v1.worker.gpu.model_states as model_states_module
    except ImportError:
        return

    if getattr(model_runner_module, "_vllm_beam_model_state_patched", False):
        return

    original_init_model_state = model_runner_module.init_model_state

    def patched_init_model_state(vllm_config, model, encoder_cache, device):
        model_state = original_init_model_state(
            vllm_config,
            model,
            encoder_cache,
            device,
        )
        _attach_beam_sampler_to_model_state(model_state, vllm_config, device)
        return model_state

    model_runner_module.init_model_state = patched_init_model_state
    model_states_module.init_model_state = patched_init_model_state
    model_runner_module._vllm_beam_model_state_patched = True


def _attach_beam_sampler_to_model_state(
    model_state: Any,
    vllm_config: Any,
    device: Any,
) -> None:
    if getattr(model_state, "_vllm_beam_sampler_patched", False):
        return
    if not _is_beam_scheduler_config(vllm_config):
        return

    original_add_request = model_state.add_request
    original_remove_request = model_state.remove_request
    original_custom_sampler = model_state.custom_sampler
    original_postprocess_state = model_state.postprocess_state

    def custom_sampler(sampler: Any) -> tuple[Any, Any] | None:
        custom = original_custom_sampler(sampler)
        rejection_sampler = None
        if custom is not None:
            sampler, rejection_sampler = custom

        from vllm_beam_search.mrv2_sampler import BeamSearchMRV2Sampler

        beam_sampler = BeamSearchMRV2Sampler(sampler, vllm_config, device)
        setattr(model_state, "_vllm_beam_sampler", beam_sampler)
        _configure_beam_sampler_from_model_state(model_state, beam_sampler)
        return beam_sampler, rejection_sampler

    def add_request(req_index: int, new_req_data: Any) -> None:
        original_add_request(req_index, new_req_data)
        beam_sampler = getattr(model_state, "_vllm_beam_sampler", None)
        if beam_sampler is None:
            return
        prompt_token_ids = (
            new_req_data.prefill_token_ids
            if new_req_data.prefill_token_ids is not None
            else new_req_data.prompt_token_ids
        )
        beam_sampler.register_request(
            new_req_data.req_id,
            new_req_data.sampling_params,
            prompt_token_ids,
        )

    def remove_request(req_id: str) -> None:
        original_remove_request(req_id)
        beam_sampler = getattr(model_state, "_vllm_beam_sampler", None)
        if beam_sampler is not None:
            beam_sampler.remove_request(req_id)

    def postprocess_state(idx_mapping: Any, num_sampled: Any) -> None:
        original_postprocess_state(idx_mapping, num_sampled)
        beam_sampler = getattr(model_state, "_vllm_beam_sampler", None)
        if beam_sampler is not None:
            beam_sampler.apply_pending_rewrites()

    model_state.custom_sampler = custom_sampler
    model_state.add_request = add_request
    model_state.remove_request = remove_request
    model_state.postprocess_state = postprocess_state
    model_state._vllm_beam_sampler_patched = True


def _configure_beam_sampler_from_model_state(
    model_state: Any,
    beam_sampler: Any | None = None,
) -> None:
    beam_sampler = beam_sampler or getattr(model_state, "_vllm_beam_sampler", None)
    if beam_sampler is None:
        return

    block_tables = getattr(model_state, "_vllm_beam_block_tables", None)
    self_attn_groups = getattr(model_state, "_vllm_beam_self_attn_groups", ())
    if block_tables is not None:
        beam_sampler.set_block_tables(block_tables, self_attn_groups)

    kv_cache_config = getattr(model_state, "_vllm_beam_kv_cache_config", None)
    forward_context = getattr(model_state, "_vllm_beam_forward_context", None)
    if kv_cache_config is not None and forward_context is not None:
        beam_sampler.set_kv_caches(kv_cache_config, forward_context)


def _flash_attn_hopper_block_size_one_enabled() -> bool:
    try:
        from vllm.platforms import current_platform
        from vllm.platforms.interface import DeviceCapability
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

        capability = current_platform.get_device_capability()
        return (
            current_platform.is_cuda()
            and capability is not None
            and capability >= DeviceCapability(9, 0)
            and capability < DeviceCapability(10, 0)
            and get_flash_attn_version() == 3
        )
    except Exception:
        return False


def _patch_flash_attn_hopper_block_size_one() -> None:
    """Advertise FA3 page-size-1 support on Hopper for beam experiments."""
    try:
        from vllm.v1.attention.backend import MultipleOf
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    except Exception:
        return

    if getattr(FlashAttentionBackend, "_vllm_beam_fa3_bs1_patched", False):
        return

    original_supported = FlashAttentionBackend.get_supported_kernel_block_sizes
    original_shape = FlashAttentionBackend.get_kv_cache_shape

    def patched_supported() -> list[int | MultipleOf]:
        try:
            supported = list(original_supported())
        except AssertionError:
            supported = [MultipleOf(16)]
        if _flash_attn_hopper_block_size_one_enabled() and 1 not in supported:
            supported.insert(0, 1)
        return supported

    def patched_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size == 1 and _flash_attn_hopper_block_size_one_enabled():
            return (num_blocks, 2, block_size, num_kv_heads, head_size)
        return original_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            cache_dtype_str,
        )

    FlashAttentionBackend.get_supported_kernel_block_sizes = staticmethod(
        patched_supported
    )
    FlashAttentionBackend.get_kv_cache_shape = staticmethod(patched_shape)
    FlashAttentionBackend._vllm_beam_fa3_bs1_patched = True


def _patch_mrv2_gpu_model_runner_history_rewrites() -> None:
    """Give MRV2 custom samplers access to worker KV/block state."""
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.v1.kv_cache_interface import CrossAttentionSpec
    except ImportError:
        return

    if getattr(GPUModelRunner, "_vllm_beam_history_rewrite_patched", False):
        return

    original_initialize_kv_cache = GPUModelRunner.initialize_kv_cache

    def _bind_beam_worker_state(self) -> None:
        model_state = getattr(self, "model_state", None)
        block_tables = getattr(self, "block_tables", None)
        if model_state is None or block_tables is None:
            return

        self_attn_groups = tuple(
            idx
            for idx, group in enumerate(self.kv_cache_config.kv_cache_groups)
            if not isinstance(group.kv_cache_spec, CrossAttentionSpec)
        )
        setattr(model_state, "_vllm_beam_block_tables", block_tables)
        setattr(model_state, "_vllm_beam_self_attn_groups", self_attn_groups)
        setattr(model_state, "_vllm_beam_kv_cache_config", self.kv_cache_config)
        setattr(
            model_state,
            "_vllm_beam_forward_context",
            self.compilation_config.static_forward_context,
        )
        _configure_beam_sampler_from_model_state(model_state)

    def patched_initialize_kv_cache(self, kv_cache_config):
        original_initialize_kv_cache(self, kv_cache_config)
        _bind_beam_worker_state(self)

    GPUModelRunner.initialize_kv_cache = patched_initialize_kv_cache
    GPUModelRunner._vllm_beam_history_rewrite_patched = True


@dataclass(frozen=True)
class _PrefixSnapshot:
    """Reusable source KV prefix captured before any destination is rebased."""

    num_computed_tokens: int
    blocks_by_manager: dict[int, list[Any]]


class BeamSearchScheduler(Scheduler):
    """V1 scheduler running `beam_width` sibling requests per beam group."""

    def __init__(
        self,
        vllm_config: Any,
        kv_cache_config: Any,
        structured_output_manager: Any,
        block_size: int,
        mm_registry: Any = None,
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
        if not self.use_v2_model_runner or not self.scheduler_config.async_scheduling:
            raise RuntimeError(
                "BeamSearchScheduler requires MRV2 with async scheduling enabled."
            )

        self._prefix_caching_enabled = bool(self.cache_config.enable_prefix_caching)
        self._block_hasher: Any = None

        self.beam_groups: dict[str, BeamGroup] = {}
        self.beam_to_group: dict[str, str] = {}
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self.pp_size = self.parallel_config.pipeline_parallel_size

        # Indices of the decoder (self-attention) KV managers — everything
        # that is not cross-attention. Computed lazily on first use.
        self._self_attn_mgr_idxs: list[int] | None = None

    def _update_after_schedule(self, scheduler_output: "SchedulerOutput") -> None:
        super()._update_after_schedule(scheduler_output)

        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output
                and request.num_output_placeholders > 0
            )
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            request.spec_token_ids = self._spec_token_placeholders

            request.next_decode_eligible_step = self.current_step + self.pp_size

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        if request.async_tokens_to_discard > 0:
            request.async_tokens_to_discard -= 1
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request,
                request.num_computed_tokens - request.num_output_placeholders,
            )
        return new_token_ids, stopped

    # ------------------------------------------------------------------
    # add_request: detect beam, pre-create children
    # ------------------------------------------------------------------

    @staticmethod
    def _get_beam_width(request: Request) -> int | None:
        sp = request.sampling_params
        if sp is None or sp.extra_args is None:
            return None
        bw = sp.extra_args.get("beam_width")
        if bw is None:
            return None
        bw = int(bw)
        return bw if bw > 1 else None

    def add_request(self, request: Request) -> None:
        beam_width = self._get_beam_width(request)
        if beam_width is None:
            super().add_request(request)
            return

        if self._prefix_caching_enabled and self._block_hasher is None:
            ghn = getattr(request, "get_hash_new_full_blocks", None)
            self._block_hasher = getattr(ghn, "func", None) or getattr(
                request, "_block_hasher", None
            )

        extra = request.sampling_params.extra_args or {}
        group = BeamGroup(
            orig_request_id=request.request_id,
            orig_request=request,
            beam_width=beam_width,
            length_penalty=float(extra.get("length_penalty", 1.0)),
        )
        self.beam_groups[request.request_id] = group

        for i in range(beam_width):
            child = self._make_beam_child(request, i, beam_width)
            group.beam_request_ids.append(child.request_id)
            group.beam_requests.append(child)
            self.beam_to_group[child.request_id] = request.request_id
            super().add_request(child)

    def _make_beam_child(
        self, orig: Request, beam_index: int, beam_width: int
    ) -> Request:
        sp = copy.copy(orig.sampling_params)
        sp.temperature = 0.0
        sp.n = 1
        sp.logprobs = None
        # Beam children are internal engine-core requests. Their outputs are
        # suppressed and the final parent response is emitted separately, so
        # avoid per-child text detokenization work when V1 observes this flag.
        sp.detokenize = False

        extra = dict(sp.extra_args or {})
        for k in ("beam_width", "length_penalty"):
            extra.pop(k, None)
        extra["_beam_group_id"] = orig.request_id
        extra["_beam_index"] = beam_index
        extra["_beam_width"] = beam_width
        extra["_beam_suppress_core_output"] = True
        eos_token_id = (
            orig.sampling_params.eos_token_id
            if orig.sampling_params is not None
            else None
        )
        if eos_token_id is not None:
            extra["_beam_eos_token_id"] = eos_token_id
        sp.extra_args = extra

        mm = ([copy.copy(f) for f in orig.mm_features]
              if orig.mm_features else None)

        child = Request(
            request_id=f"{orig.request_id}:beam:{beam_index}",
            prompt_token_ids=(
                list(orig.prompt_token_ids)
                if orig.prompt_token_ids is not None
                else None
            ),
            sampling_params=sp,
            pooling_params=None,
            client_index=orig.client_index,
            prompt_embeds=orig.prompt_embeds,
            prompt_is_token_ids=orig.prompt_is_token_ids,
            mm_features=mm,
            lora_request=orig.lora_request,
            cache_salt=orig.cache_salt,
            block_hasher=self._block_hasher,
            priority=orig.priority,
            trace_headers=orig.trace_headers,
            resumable=orig.resumable,
        )
        if self._prefix_caching_enabled and self._block_hasher is not None:
            child.block_hashes = []
            child.update_block_hashes()
        return child

    # ------------------------------------------------------------------
    # finish_requests: expand orig id -> children
    # ------------------------------------------------------------------

    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
    ) -> list[tuple[str, int]]:
        request_ids = self._normalize_finish_ids(request_ids)
        cleanup_groups, child_to_orig, expanded_ids = (
            self._expand_finish_request_ids(request_ids)
        )

        finished = super().finish_requests(expanded_ids, finished_status)
        if not cleanup_groups:
            return finished

        # External aborts target the user-facing request id, but only the
        # beam-child requests live in the base scheduler. Free the children
        # through the base scheduler, then remove all plugin-owned group state
        # so has_requests()/get_num_unfinished_requests() cannot stay true
        # forever after a client disconnect.
        public_finished: dict[str, int] = {}
        kept: list[tuple[str, int]] = []
        for req_id, client_index in finished:
            gid = child_to_orig.get(req_id)
            if gid is None:
                kept.append((req_id, client_index))
                continue
            group = cleanup_groups.get(gid)
            if group is not None:
                public_finished[group.orig_request_id] = (
                    group.orig_request.client_index
                )

        for group in cleanup_groups.values():
            self._cleanup_group(group)

        kept.extend(public_finished.items())
        return kept

    @staticmethod
    def _normalize_finish_ids(
        request_ids: str | Iterable[str] | None,
    ) -> tuple[str, ...] | None:
        if request_ids is None:
            return None
        if isinstance(request_ids, str):
            return (request_ids,)
        return tuple(request_ids)

    def _expand_finish_request_ids(
        self,
        request_ids: tuple[str, ...] | None,
    ) -> tuple[dict[str, BeamGroup], dict[str, str], list[str] | None]:
        if request_ids is None:
            cleanup_groups = dict(self.beam_groups)
            child_to_orig = {
                child_id: group_id
                for group_id, group in cleanup_groups.items()
                for child_id in group.beam_request_ids
            }
            return cleanup_groups, child_to_orig, None

        cleanup_groups: dict[str, BeamGroup] = {}
        child_to_orig: dict[str, str] = {}
        expanded_ids: list[str] = []

        for request_id in request_ids:
            group = self.beam_groups.get(request_id)
            if group is not None:
                cleanup_groups[request_id] = group
                expanded_ids.extend(group.beam_request_ids)
                child_to_orig.update({
                    child_id: request_id for child_id in group.beam_request_ids
                })
                continue

            group_id = self.beam_to_group.get(request_id)
            if group_id is not None:
                group = self.beam_groups.get(group_id)
                if group is not None:
                    cleanup_groups[group_id] = group
                    child_to_orig.update({
                        child_id: group_id for child_id in group.beam_request_ids
                    })
            expanded_ids.append(request_id)

        return cleanup_groups, child_to_orig, expanded_ids

    # ------------------------------------------------------------------
    # update_from_output
    # ------------------------------------------------------------------

    def schedule(self) -> "SchedulerOutput":
        scheduler_output = super().schedule()
        self._clamp_async_beam_decode_chunks(scheduler_output)
        return scheduler_output

    def _clamp_async_beam_decode_chunks(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        for req_id, scheduled in list(
            scheduler_output.num_scheduled_tokens.items()
        ):
            if scheduled <= 1 or req_id not in self.beam_to_group:
                continue
            request = self.requests.get(req_id)
            if request is None:
                continue

            # The initial decoder prompt is scheduled as a normal prefill
            # chunk. Only clamp decode chunks, where beam search has selected
            # exactly one next token for MRV2 to consume.
            prev_computed = request.num_computed_tokens - scheduled
            if prev_computed < request.num_prompt_tokens:
                continue

            trimmed = scheduled - 1
            scheduler_output.num_scheduled_tokens[req_id] = 1
            scheduler_output.total_num_scheduled_tokens -= trimmed
            request.num_computed_tokens -= trimmed
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )

    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[int, EngineCoreOutputs]:
        outputs = super().update_from_output(scheduler_output, model_runner_output)

        finished_children = self._suppress_child_outputs(outputs)
        if not self.beam_groups:
            return outputs

        transitions = self._beam_transitions_from_output(model_runner_output)

        for group in list(self.beam_groups.values()):
            if group.finalized:
                continue

            transition = transitions.get(group.orig_request_id)
            if transition is not None:
                self._drain_beam_transition(group, transition)

            active_slots = self._active_slots(group, transition)
            if self._should_finalize_group(
                group, transition, active_slots, finished_children
            ):
                self._finalize_group(group, outputs, transition)
                continue

            self._apply_fork_plan(group, transition, active_slots)

        return outputs

    def _suppress_child_outputs(
        self,
        outputs: dict[int, EngineCoreOutputs],
    ) -> set[str]:
        """Drop internal child outputs and return children that hit a finish."""
        finished_children: set[str] = set()
        for engine_outputs in outputs.values():
            kept = []
            for out in engine_outputs.outputs:
                group_id = self.beam_to_group.get(out.request_id)
                if group_id is not None or self._core_output_suppressed(
                    out.request_id
                ):
                    if group_id is not None and out.finish_reason is not None:
                        finished_children.add(out.request_id)
                    continue
                kept.append(out)
            engine_outputs.outputs = kept
        return finished_children

    def _core_output_suppressed(self, request_id: str) -> bool:
        request = self.requests.get(request_id)
        if request is None or request.sampling_params is None:
            return False
        extra = request.sampling_params.extra_args
        return bool(extra and extra.get("_beam_suppress_core_output"))

    @staticmethod
    def _beam_transitions_from_output(
        model_runner_output: "ModelRunnerOutput",
    ) -> dict[str, BeamTransition]:
        custom_outputs = getattr(model_runner_output, "custom_outputs", None)
        if not custom_outputs:
            return {}
        transitions = custom_outputs.get(_BEAM_TRANSITIONS_OUTPUT)
        if not transitions:
            return {}
        return {transition.group_id: transition for transition in transitions}

    def _drain_beam_transition(
        self,
        group: BeamGroup,
        transition: BeamTransition,
    ) -> None:
        for tokens, cum in transition.completions:
            self._add_completion(group, list(tokens), cum)

    def _should_finalize_group(
        self,
        group: BeamGroup,
        transition: BeamTransition | None,
        active_slots: list[int],
        finished_children: set[str],
    ) -> bool:
        # HF/V0 early_stopping=True finalizes once beam_width hypotheses exist.
        if len(group.completed) >= group.beam_width:
            return True

        if transition is None:
            return False

        if not active_slots:
            return True

        return all(
            group.beam_request_ids[slot] in finished_children
            for slot in active_slots
        )

    def _add_completion(
        self, group: BeamGroup, tokens: list[int], cum: float
    ) -> None:
        # V0 scores list(seq.data.get_token_ids()) with
        # exclude_last_from_length=True, so the denominator includes decoder
        # prompt tokens and excludes the final generated token (EOS here).
        group.add_completed(
            CompletedBeam(
                tokens=list(tokens),
                cum_score=cum,
                length=self._beam_length(group, tokens),
                finish_reason="stop",
            )
        )

    @staticmethod
    def _prompt_len(group: BeamGroup) -> int:
        return len(group.orig_request.prompt_token_ids or [])

    def _beam_length(self, group: BeamGroup, tokens: list[int]) -> int:
        return max(self._prompt_len(group) + len(tokens) - 1, 1)

    # ------------------------------------------------------------------
    # Fork plan execution (KV rebase)
    # ------------------------------------------------------------------

    def _self_attn_managers(self) -> list[int]:
        if self._self_attn_mgr_idxs is None:
            mgrs = self.kv_cache_manager.coordinator.single_type_managers
            self._self_attn_mgr_idxs = [
                i for i, m in enumerate(mgrs)
                if not isinstance(m, CrossAttentionManager)
            ]
        return self._self_attn_mgr_idxs

    def _active_slots(
        self,
        group: BeamGroup,
        transition: BeamTransition | None,
    ) -> list[int]:
        if transition is None:
            return list(range(group.beam_width))
        return [
            slot for slot in transition.active_slots[:group.beam_width]
            if slot not in group.finished_beam_indices
        ]

    def _apply_fork_plan(
        self,
        group: BeamGroup,
        transition: BeamTransition | None,
        active_slots: list[int],
    ) -> None:
        if transition is None:
            return
        rebases = self._fork_rebases(group, transition, active_slots)
        if not rebases:
            return

        mgrs = self.kv_cache_manager.coordinator.single_type_managers
        self_idxs = self._self_attn_managers()
        snapshots = self._snapshot_fork_sources(
            group, transition, rebases, self_idxs, mgrs
        )

        for dst, src in rebases:
            self._rebase_slot(
                group,
                transition,
                dst,
                snapshots[src],
                self_idxs,
                mgrs,
            )

    @staticmethod
    def _fork_rebases(
        group: BeamGroup,
        transition: BeamTransition,
        active_slots: list[int],
    ) -> list[tuple[int, int]]:
        fork_src = transition.fork_src
        active_slot_set = set(active_slots)
        return [
            (dst, fork_src[dst])
            for dst in range(group.beam_width)
            if (
                dst in active_slot_set
                and dst < len(fork_src)
                and fork_src[dst] != dst
            )
        ]

    def _snapshot_fork_sources(
        self,
        group: BeamGroup,
        transition: BeamTransition,
        rebases: list[tuple[int, int]],
        self_idxs: list[int],
        mgrs: Any,
    ) -> dict[int, _PrefixSnapshot]:
        snapshots: dict[int, _PrefixSnapshot] = {}
        output_prefix_len = int(transition.prefix_len)

        for _dst, src in rebases:
            if src in snapshots:
                continue
            src_req = group.beam_requests[src]
            kv_prefix_len = len(src_req.prompt_token_ids or []) + output_prefix_len
            snapshots[src] = self._snapshot_source_prefix(
                group.beam_request_ids[src],
                kv_prefix_len,
                self_idxs,
                mgrs,
            )

        return snapshots

    def _snapshot_source_prefix(
        self,
        src_id: str,
        kv_prefix_len: int,
        self_idxs: list[int],
        mgrs: Any,
    ) -> _PrefixSnapshot:
        # Snapshot every source's prefix blocks BEFORE mutating anything.
        blocks_by_manager: dict[int, list[Any]] = {}
        num_computed_tokens = 0

        for manager_index in self_idxs:
            blocks = list(mgrs[manager_index].req_to_blocks.get(src_id, []))
            n_full_blocks = kv_prefix_len // self.block_size
            shared_blocks = blocks[:n_full_blocks]
            shareable_tokens = len(shared_blocks) * self.block_size
            has_partial_block = kv_prefix_len % self.block_size != 0

            blocks_by_manager[manager_index] = shared_blocks
            if has_partial_block:
                num_computed_tokens = max(num_computed_tokens, kv_prefix_len)
            else:
                num_computed_tokens = max(num_computed_tokens, shareable_tokens)

        return _PrefixSnapshot(
            num_computed_tokens=num_computed_tokens,
            blocks_by_manager=blocks_by_manager,
        )

    def _rebase_slot(
        self,
        group: BeamGroup,
        transition: BeamTransition,
        dst: int,
        src_snap: _PrefixSnapshot,
        self_idxs: list[int],
        mgrs: Any,
    ) -> None:
        block_pool = self.kv_cache_manager.block_pool
        dst_id = group.beam_request_ids[dst]
        dst_req = group.beam_requests[dst]
        n_prefix = src_snap.num_computed_tokens

        for manager_index in self_idxs:
            mgr = mgrs[manager_index]
            shared_blocks = list(src_snap.blocks_by_manager[manager_index])
            new_blocks = list(shared_blocks)

            self._replace_request_blocks(
                mgr=mgr,
                dst_id=dst_id,
                shared_blocks=shared_blocks,
                new_blocks=new_blocks,
                block_pool=block_pool,
                prefix_blocks=len(new_blocks),
            )

        self._rewrite_rebased_request(dst_req, transition, dst, n_prefix)
        self.prev_step_scheduled_req_ids.discard(dst_id)

    @staticmethod
    def _replace_request_blocks(
        mgr: Any,
        dst_id: str,
        shared_blocks: list[Any],
        new_blocks: list[Any],
        block_pool: Any,
        prefix_blocks: int,
    ) -> None:
        old_blocks = mgr.req_to_blocks.get(dst_id, [])
        suffix_blocks = list(old_blocks[prefix_blocks:])
        stale_prefix_blocks = list(old_blocks[:prefix_blocks])
        new_blocks.extend(suffix_blocks)
        # Touch new prefix (ref++) before freeing old (ref--) so blocks
        # shared between them never transiently hit ref 0.
        if shared_blocks:
            block_pool.touch(shared_blocks)
        if stale_prefix_blocks:
            block_pool.free_blocks(reversed(stale_prefix_blocks))
        mgr.req_to_blocks[dst_id] = list(new_blocks)
        num_cached = getattr(mgr, "num_cached_block", None)
        if num_cached is not None:
            num_cached[dst_id] = len(new_blocks)

    def _rewrite_rebased_request(
        self,
        dst_req: Request,
        transition: BeamTransition,
        dst: int,
        num_computed_tokens: int,
    ) -> None:
        new_output = list(transition.tokens[dst])
        dst_req._output_token_ids.clear()
        dst_req._output_token_ids.extend(new_output)
        dst_req._all_token_ids.clear()
        dst_req._all_token_ids.extend(list(dst_req.prompt_token_ids) + new_output)
        dst_req.num_computed_tokens = num_computed_tokens

        # Token state was rewritten in-place, bypassing append_token_id's
        # incremental hashing; recompute block_hashes so the base scheduler's
        # cache_full_blocks assertion stays satisfied.
        if getattr(dst_req, "_block_hasher", None) is not None:
            dst_req.block_hashes = []
            dst_req.update_block_hashes()

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def _finalize_group(
        self,
        group: BeamGroup,
        outputs: dict[int, EngineCoreOutputs],
        transition: BeamTransition | None,
    ) -> None:
        best, finish_reason = self._select_final_beam(group, transition)
        if best is not None:
            final_tokens = self._tokens_with_eos(group, best.tokens)
            self._emit_parent_output(group, outputs, final_tokens, finish_reason)
        else:
            logger.warning("Beam group %s finalized with no beams.",
                           group.orig_request_id)

        self._finish_group_children(group)
        self._cleanup_group(group)

    def _select_final_beam(
        self,
        group: BeamGroup,
        transition: BeamTransition | None,
    ) -> tuple[CompletedBeam | None, FinishReason]:
        best = group.best_completed()
        if best is not None:
            return best, FinishReason.STOP

        live = self._best_live_beam(group, transition)
        if live is not None:
            return live, FinishReason.LENGTH

        return None, FinishReason.STOP

    def _best_live_beam(
        self,
        group: BeamGroup,
        transition: BeamTransition | None,
    ) -> CompletedBeam | None:
        if transition is None:
            return None

        best_slot, best_norm = None, float("-inf")
        for slot in self._active_slots(group, transition):
            toks = transition.tokens[slot] if slot < len(transition.tokens) else []
            if not toks:
                continue
            length = self._beam_length(group, toks)
            norm = group.normalized(transition.cum[slot], length)
            if norm > best_norm:
                best_norm, best_slot = norm, slot

        if best_slot is None:
            return None

        tokens = list(transition.tokens[best_slot])
        return CompletedBeam(
            tokens=tokens,
            cum_score=transition.cum[best_slot],
            length=self._beam_length(group, tokens),
            finish_reason="length",
        )

    @staticmethod
    def _tokens_with_eos(group: BeamGroup, tokens: list[int]) -> list[int]:
        final_tokens = list(tokens)
        # The V1 incremental detokenizer holds the most-recent token pending a
        # lookahead. EOS is skipped as a special token and flushes the last word.
        eos = (
            group.orig_request.sampling_params.eos_token_id
            if group.orig_request.sampling_params is not None
            else None
        )
        if eos is not None and (not final_tokens or final_tokens[-1] != eos):
            final_tokens.append(eos)
        return final_tokens

    @staticmethod
    def _emit_parent_output(
        group: BeamGroup,
        outputs: dict[int, EngineCoreOutputs],
        final_tokens: list[int],
        finish_reason: FinishReason,
    ) -> None:
        client_index = group.orig_request.client_index
        engine_outputs = outputs.get(client_index)
        if engine_outputs is None:
            engine_outputs = EngineCoreOutputs()
            outputs[client_index] = engine_outputs
        engine_outputs.outputs.append(
            EngineCoreOutput(
                request_id=group.orig_request_id,
                new_token_ids=final_tokens,
                finish_reason=finish_reason,
            )
        )

    def _finish_group_children(self, group: BeamGroup) -> None:
        Scheduler.finish_requests(
            self, group.beam_request_ids, RequestStatus.FINISHED_STOPPED
        )

    def _cleanup_group(self, group: BeamGroup) -> None:
        group.finalized = True
        for bid in group.beam_request_ids:
            self.beam_to_group.pop(bid, None)
        self.beam_groups.pop(group.orig_request_id, None)

    def get_num_unfinished_requests(self) -> int:
        return super().get_num_unfinished_requests() + len(self.beam_groups)

    def has_requests(self) -> bool:
        return super().has_requests() or bool(self.beam_groups)


_install_worker_history_rewrite_hooks()
