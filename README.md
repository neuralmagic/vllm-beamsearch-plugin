# vLLM Beam Search Plugin

MRV2 beam-search scheduler and sampler plugin for vLLM V1.

This package provides:

- `vllm_beam_search.scheduler.BeamSearchScheduler`
- an MRV2 custom sampler wrapper installed through a plugin-local `ModelState`
  hook
- plugin-local runtime hooks for MRV2 worker history rewrites
- Hopper FA3 block-size-1 enablement for the beam-search path

The current production path targets MRV2 generate models with async scheduling.
The sampler hook is model-state generic; BART-family models still need the
companion `vllm-bart-plugin` for encoder-decoder model support.

## Install

```bash
uv pip install -e /home/LucasWilkinson/local/vllm-beamsearch-plugin
```

For stress tooling:

```bash
uv pip install -e '/home/LucasWilkinson/local/vllm-beamsearch-plugin[stress]'
```

## Server

```bash
MODEL=${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}
SERVED_MODEL=${SERVED_MODEL:-llama3-8b}

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_CACHE=${HF_HUB_CACHE:-/mnt/data/engine/hub_cache} \
VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
PYTHONPATH=/home/LucasWilkinson/local/vllm-beamsearch-plugin:${PYTHONPATH:-} \
/home/LucasWilkinson/local/vllm/.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name "${SERVED_MODEL}" \
  --dtype bfloat16 \
  --port 8005 \
  --block-size 16 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --scheduler-cls vllm_beam_search.scheduler.BeamSearchScheduler \
  --max-num-seqs 512 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.9 \
  --disable-log-stats \
  --no-enable-log-requests \
  --disable-uvicorn-access-log
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
/home/LucasWilkinson/local/vllm/.venv/bin/python -m pytest \
  /home/LucasWilkinson/local/vllm-beamsearch-plugin/tests -q
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

- `VLLM_BEAM_FA3_BLOCK_SIZE_ONE=0` disables the Hopper FA3 block-size-1 patch.
- `VLLM_BEAM_GPU_SYNC_CHECK=warn|error` enables PyTorch CUDA sync debugging
  around beam sampler hot paths.
- `VLLM_BEAM_GROUP_STATE_CAPACITY` controls GPU beam-state pool capacity.
- `VLLM_BEAM_TRANSITION_BUFFER_SLOTS` controls async transition buffer slots.
