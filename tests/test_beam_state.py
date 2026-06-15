from __future__ import annotations

from types import SimpleNamespace

from vllm_beam_search.beam_state import BeamGroup, CompletedBeam


def _group(*, length_penalty: float = 1.0) -> BeamGroup:
    request = SimpleNamespace(
        request_id="orig",
        prompt_token_ids=[101, 102],
    )
    return BeamGroup(
        orig_request_id="orig",
        orig_request=request,
        beam_width=2,
        length_penalty=length_penalty,
    )


def test_normalized_respects_length_penalty() -> None:
    group = _group(length_penalty=1.0)

    assert group.normalized(-6.0, 3) == -2.0
    assert group.normalized(-6.0, 0) == -6.0


def test_best_completed_prefers_eos_over_length_fallback() -> None:
    group = _group()
    group.add_completed(
        CompletedBeam(
            tokens=[1, 2, 3],
            cum_score=-9.0,
            length=3,
            finish_reason="length",
        )
    )
    group.add_completed(
        CompletedBeam(
            tokens=[4, 5],
            cum_score=-8.0,
            length=2,
            finish_reason="stop",
        )
    )

    assert group.best_completed().tokens == [4, 5]


def test_best_completed_uses_length_fallback_when_no_eos() -> None:
    group = _group()
    group.add_completed(
        CompletedBeam(
            tokens=[1],
            cum_score=-4.0,
            length=1,
            finish_reason="length",
        )
    )
    group.add_completed(
        CompletedBeam(
            tokens=[2, 3],
            cum_score=-5.0,
            length=2,
            finish_reason="length",
        )
    )

    assert group.best_completed().tokens == [2, 3]
