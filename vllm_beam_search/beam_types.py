from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

_INIT_NEG = -1e9


@dataclass(frozen=True)
class BeamTransition:
    group_id: str
    step_id: int
    prefix_len: int
    active_slots: tuple[int, ...]
    fork_src: tuple[int, ...]
    tokens: tuple[tuple[int, ...], ...]
    cum: tuple[float, ...]
    completions: tuple[tuple[tuple[int, ...], float], ...] = ()
    inactive_slots: tuple[int, ...] = ()


@dataclass
class BeamRuntime:
    beam_width: int
    eos_token_id: int | None
    no_repeat_ngram_size: int = 0
    prompt_tokens: list[int] = field(default_factory=list)
    tokens: list[list[int]] = field(default_factory=list)
    cum: list[float] = field(default_factory=list)
    decode_step: int = 0
    fork_src: list[int] = field(default_factory=list)
    active: list[bool] = field(default_factory=list)
    prefix_len: int = 0
    transitions: deque[BeamTransition] = field(default_factory=deque)
