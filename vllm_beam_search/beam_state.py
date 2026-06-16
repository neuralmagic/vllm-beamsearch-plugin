"""Beam-search data classes for the V1 plugin's in-flight scheduler.

These live on the scheduler side. Per-step beam decisions arrive from the
MRV2 sampler as BeamTransition records; the scheduler keeps group identity
and completed-beam records.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from vllm.v1.request import Request


@dataclass
class CompletedBeam:
    """A beam that has hit EOS, max_tokens, or another stop criterion."""

    tokens: list[int]
    cum_score: float
    length: int  # length used for length-penalty normalization (excludes EOS)
    finish_reason: str = "stop"  # "stop" | "length"


@dataclass
class BeamGroup:
    """One user request's worth of beams.

    Lives in BeamSearchScheduler.beam_groups, keyed by the original
    user-facing request_id. The actual generation is done by
    `beam_width` sibling Request objects (one per beam) with ids
    `f"{orig_id}:beam:{i}"`.
    """

    orig_request_id: str
    orig_request: Request
    beam_width: int

    # Sampling-time params mirrored from the public request.
    length_penalty: float = 1.0

    beam_request_ids: list[str] = field(default_factory=list)
    # Refs to the beam-child Request objects; held so we can read their
    # final `output_token_ids` even after the base scheduler removes them
    # from `self.requests` on finish.
    beam_requests: list[Request] = field(default_factory=list)
    completed: list[CompletedBeam] = field(default_factory=list)
    # Set of beam_indices that have finished (so we don't double-record).
    finished_beam_indices: set[int] = field(default_factory=set)

    # True after we've emitted the final EngineCoreOutput for this group;
    # the scheduler uses this to suppress duplicate emissions.
    finalized: bool = False

    def normalized(self, score: float, length: int) -> float:
        """HF-compatible length normalization: score / length**length_penalty."""
        if self.length_penalty == 0.0 or length <= 0:
            return score
        return score / (float(length) ** self.length_penalty)

    def add_completed(self, beam: CompletedBeam) -> None:
        # V0 keeps a plain completed list during generation and sorts it at
        # finalization; it does not maintain a capped hypothesis heap.
        self.completed.append(beam)

    def best_completed(self) -> CompletedBeam | None:
        if not self.completed:
            return None
        # Prefer beams that hit EOS naturally — beams that ran to
        # max_tokens ("length") are fallback only.
        eos_beams = [b for b in self.completed if b.finish_reason == "stop"]
        pool = eos_beams if eos_beams else self.completed
        return max(
            pool,
            key=lambda b: self.normalized(b.cum_score, b.length),
        )
