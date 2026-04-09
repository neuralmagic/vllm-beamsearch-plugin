"""Unit tests for BeamGroup / BeamState."""

from unittest.mock import MagicMock

from vllm_beam_search.beam_state import BeamGroup, BeamState


def _mock_request(request_id: str = "req_0") -> MagicMock:
    req = MagicMock()
    req.request_id = request_id
    return req


def test_beam_group_next_id():
    group = BeamGroup(
        original_request_id="orig",
        original_request=_mock_request("orig"),
        beam_width=4,
    )
    ids = [group.next_beam_id() for _ in range(5)]
    assert ids == [
        "orig:beam:0",
        "orig:beam:1",
        "orig:beam:2",
        "orig:beam:3",
        "orig:beam:4",
    ]


def test_normalized_score_no_penalty():
    group = BeamGroup(
        original_request_id="orig",
        original_request=_mock_request(),
        beam_width=4,
        length_penalty=0.0,
    )
    assert group.normalized_score(-5.0, 10) == -5.0


def test_normalized_score_with_penalty():
    group = BeamGroup(
        original_request_id="orig",
        original_request=_mock_request(),
        beam_width=4,
        length_penalty=1.0,
    )
    score = group.normalized_score(-6.0, 7)
    # penalty = ((5 + 7) / 6) ^ 1.0 = 2.0
    assert abs(score - (-6.0 / 2.0)) < 1e-6


def test_early_stop():
    group = BeamGroup(
        original_request_id="orig",
        original_request=_mock_request(),
        beam_width=2,
        length_penalty=0.0,
    )
    # Add two active beams.
    b0 = BeamState(request=_mock_request("b0"), score=-3.0, token_ids=[1, 2, 3])
    b1 = BeamState(request=_mock_request("b1"), score=-5.0, token_ids=[4, 5])
    group.active_beams = {"b0": b0, "b1": b1}

    # No completed beam yet — can't early stop.
    assert not group.can_early_stop()

    # Complete b0.
    b0.finished = True
    group.best_completed = b0
    # b1 has score -5.0 which is worse than b0's -3.0.
    # Since logprobs are <= 0, b1 can't improve. Early stop!
    assert group.can_early_stop()

    # But if b1 has a better score than the completed beam, no early stop.
    b1.score = -2.0
    assert not group.can_early_stop()
