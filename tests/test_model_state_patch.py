from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from vllm_beam_search.scheduler import _attach_beam_sampler_to_model_state


class _FakeBeamSampler:
    def __init__(self, base_sampler: Any, vllm_config: Any, device: Any) -> None:
        self.base_sampler = base_sampler
        self.vllm_config = vllm_config
        self.device = device
        self.calls: list[tuple[Any, ...]] = []

    def set_block_tables(self, block_tables: Any, self_attn_groups: Any) -> None:
        self.calls.append(("set_block_tables", block_tables, self_attn_groups))

    def set_kv_caches(self, kv_cache_config: Any, forward_context: Any) -> None:
        self.calls.append(("set_kv_caches", kv_cache_config, forward_context))

    def register_request(
        self,
        req_id: str,
        sampling_params: Any,
        prompt_token_ids: list[int] | None,
    ) -> None:
        self.calls.append(
            ("register_request", req_id, sampling_params, prompt_token_ids)
        )

    def remove_request(self, req_id: str) -> None:
        self.calls.append(("remove_request", req_id))

    def apply_pending_rewrites(self) -> None:
        self.calls.append(("apply_pending_rewrites",))


class _FakeModelState:
    def __init__(self) -> None:
        self.original_calls: list[tuple[Any, ...]] = []

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        self.original_calls.append(("add_request", req_index, new_req_data.req_id))

    def remove_request(self, req_id: str) -> None:
        self.original_calls.append(("remove_request", req_id))

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        self.original_calls.append(("custom_sampler", sampler))
        return None

    def postprocess_state(self, idx_mapping: Any, num_sampled: Any) -> None:
        self.original_calls.append(("postprocess_state", idx_mapping, num_sampled))


def _beam_config() -> SimpleNamespace:
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            scheduler_cls="vllm_beam_search.scheduler.BeamSearchScheduler"
        )
    )


def test_beam_scheduler_wraps_generic_model_state(monkeypatch):
    fake_module = ModuleType("vllm_beam_search.mrv2_sampler")
    fake_module.BeamSearchMRV2Sampler = _FakeBeamSampler
    monkeypatch.setitem(sys.modules, "vllm_beam_search.mrv2_sampler", fake_module)

    model_state = _FakeModelState()
    model_state._vllm_beam_block_tables = "block-tables"
    model_state._vllm_beam_self_attn_groups = (0,)
    model_state._vllm_beam_kv_cache_config = "kv-cache-config"
    model_state._vllm_beam_forward_context = "forward-context"

    _attach_beam_sampler_to_model_state(model_state, _beam_config(), "cuda")
    sampler, rejection_sampler = model_state.custom_sampler("base-sampler")

    assert isinstance(sampler, _FakeBeamSampler)
    assert rejection_sampler is None
    assert sampler.calls == [
        ("set_block_tables", "block-tables", (0,)),
        ("set_kv_caches", "kv-cache-config", "forward-context"),
    ]

    request = SimpleNamespace(
        req_id="req",
        prefill_token_ids=[1, 2],
        prompt_token_ids=[3, 4],
        sampling_params="sampling-params",
    )
    model_state.add_request(7, request)
    model_state.remove_request("req")
    model_state.postprocess_state("idx", "num")

    assert model_state.original_calls == [
        ("custom_sampler", "base-sampler"),
        ("add_request", 7, "req"),
        ("remove_request", "req"),
        ("postprocess_state", "idx", "num"),
    ]
    assert sampler.calls[-3:] == [
        ("register_request", "req", "sampling-params", [1, 2]),
        ("remove_request", "req"),
        ("apply_pending_rewrites",),
    ]


def test_default_scheduler_does_not_wrap_model_state():
    model_state = _FakeModelState()

    _attach_beam_sampler_to_model_state(
        model_state,
        SimpleNamespace(scheduler_config=SimpleNamespace(scheduler_cls=None)),
        "cuda",
    )

    assert not hasattr(model_state, "_vllm_beam_sampler_patched")
