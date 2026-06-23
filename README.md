# vLLM Beam Search Plugin

MRV2 beam-search scheduler and sampler plugin for vLLM V1.

This package provides:

- `vllm_beam_search.scheduler.BeamSearchScheduler`
- an MRV2 custom sampler wrapper installed through a plugin-local `ModelState`
  hook
- plugin-local runtime hooks for MRV2 worker history rewrites

The current production path targets MRV2 generate models with async scheduling.
The sampler hook is model-state generic; BART-family models still need the
companion `vllm-bart-plugin` for encoder-decoder model support.

For BART-family encoder-decoder serving, see
[`BART_BEAM_SEARCH.md`](BART_BEAM_SEARCH.md).

## Install

```bash
uv pip install -e .
```

For stress tooling:

```bash
uv pip install -e '.[stress]'
```

## Server

```bash
MODEL=${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}
SERVED_MODEL=${SERVED_MODEL:-llama3-8b}

CUDA_VISIBLE_DEVICES=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name "${SERVED_MODEL}" \
  --dtype bfloat16 \
  --port 8005 \
  --scheduler-cls vllm_beam_search.scheduler.BeamSearchScheduler
```

## Request Shape

```json
{
  "model": "llama3-8b",
  "prompt": "Write a concise summary of why beam search is useful:",
  "max_tokens": 128,
  "temperature": 0,
  "add_special_tokens": false,
  "vllm_xargs": {
    "beam_width": 4,
    "no_repeat_ngram_size": 3
  }
}
```

## Validation

Run unit tests:

```bash
python -m pytest tests -q
```

Run sustained stress plus memory sampling against a running server:

```bash
vllm-beam-stress \
  --base-url http://localhost:8005 \
  --model llama3-8b \
  --rounds 100 \
  --requests-per-round 32 \
  --concurrency 64 \
  --abort-rounds 3
```

The stress tool writes CSV samples with request count, RSS, and GPU memory.

## Runtime Knobs

- `VLLM_BEAM_GROUP_STATE_CAPACITY` controls GPU beam-state pool capacity.
- `VLLM_BEAM_TRANSITION_BUFFER_SLOTS` controls async transition buffer slots.
