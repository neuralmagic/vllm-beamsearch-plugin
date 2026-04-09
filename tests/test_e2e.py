"""End-to-end test comparing plugin beam search against existing vLLM beam search.

Run with a GPU:
    PYTHONPATH=. pytest tests/test_e2e.py -v -s
"""

from __future__ import annotations

import time

import pytest

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPTS = [
    "The future of artificial intelligence is",
]
BEAM_WIDTH = 4
MAX_TOKENS = 32
DTYPE = "half"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def baseline_outputs(tokenizer):
    """Run existing vLLM beam search (client-side iterative)."""
    from vllm import LLM
    from vllm.sampling_params import BeamSearchParams

    llm = LLM(
        model=MODEL,
        dtype=DTYPE,
        enforce_eager=True,
        max_num_seqs=16,
    )

    params = BeamSearchParams(
        beam_width=BEAM_WIDTH,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        length_penalty=1.0,
    )

    t0 = time.perf_counter()
    results = llm.beam_search(
        [{"prompt": p} for p in PROMPTS],
        params,
    )
    elapsed = time.perf_counter() - t0

    del llm

    outputs = []
    for result in results:
        beams = []
        for seq in result.sequences:
            prompt_len = len(tokenizer.encode(PROMPTS[0]))
            generated_ids = seq.tokens[prompt_len:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            beams.append({
                "token_ids": generated_ids,
                "text": text,
                "score": seq.cum_logprob,
            })
        outputs.append(beams)

    return outputs, elapsed


@pytest.fixture(scope="module")
def plugin_outputs(tokenizer):
    """Run our beam search scheduler plugin."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype=DTYPE,
        enforce_eager=True,
        max_num_seqs=16,
        enable_prefix_caching=True,
        scheduler_cls="vllm_beam_search.scheduler.BeamSearchScheduler",
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        extra_args={"beam_width": BEAM_WIDTH, "length_penalty": 1.0},
    )

    t0 = time.perf_counter()
    results = llm.generate(PROMPTS, sampling_params)
    elapsed = time.perf_counter() - t0

    del llm

    outputs = []
    for result in results:
        text = result.outputs[0].text
        token_ids = list(result.outputs[0].token_ids)
        outputs.append({
            "token_ids": token_ids,
            "text": text,
        })

    return outputs, elapsed


def test_plugin_produces_output(plugin_outputs):
    """Basic sanity: plugin produces non-empty output."""
    outputs, elapsed = plugin_outputs
    assert len(outputs) == len(PROMPTS)
    for out in outputs:
        assert len(out["token_ids"]) > 0, "Plugin produced no tokens"
        assert len(out["text"]) > 0, "Plugin produced empty text"
    print(f"\n[Plugin] Time: {elapsed:.2f}s")
    print(f"[Plugin] Output: {outputs[0]['text']!r}")


def test_baseline_produces_output(baseline_outputs):
    """Sanity: baseline beam search works."""
    outputs, elapsed = baseline_outputs
    assert len(outputs) == len(PROMPTS)
    assert len(outputs[0]) == BEAM_WIDTH, (
        f"Expected {BEAM_WIDTH} beams, got {len(outputs[0])}"
    )
    print(f"\n[Baseline] Time: {elapsed:.2f}s")
    for i, beam in enumerate(outputs[0]):
        print(f"[Baseline] Beam {i}: {beam['text']!r} (score={beam['score']:.4f})")


def test_comparison(baseline_outputs, plugin_outputs):
    """Compare both methods: output quality and timing."""
    baseline, b_time = baseline_outputs
    plugin, p_time = plugin_outputs

    baseline_top1 = baseline[0][0]  # First prompt, first (best) beam.
    plugin_result = plugin[0]

    print(f"\n{'='*60}")
    print(f"BEAM SEARCH COMPARISON (beam_width={BEAM_WIDTH}, max_tokens={MAX_TOKENS})")
    print(f"{'='*60}")
    print(f"\n[Baseline (iterative, 2*beam_width candidates)]")
    print(f"  Top-1: {baseline_top1['text']!r}")
    print(f"  Time:  {b_time:.2f}s")
    print(f"\n[Plugin (scheduler-level, beam_width-1 candidates)]")
    print(f"  Output: {plugin_result['text']!r}")
    print(f"  Time:   {p_time:.2f}s")
    print(f"\n[Speedup] {b_time / p_time:.2f}x")
    print(f"{'='*60}")

    # Both should produce non-trivial output.
    assert len(baseline_top1["token_ids"]) > 0
    assert len(plugin_result["token_ids"]) > 0

    # Note: exact match is NOT expected because the two methods use
    # different candidate pool sizes (2*beam_width vs beam_width-1)
    # and different length penalty formulas (HF-style vs Wu et al.).
    # Both are valid beam search implementations.
