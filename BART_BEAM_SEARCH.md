# BART + Beam Search

This guide runs BART-family encoder-decoder models with the beam-search
scheduler. It requires both plugins in the same vLLM environment:

- `vllm-bart-plugin` for BART model and prompt support
- `vllm-beam-search` for the MRV2 beam scheduler and sampler

## Install

Install vLLM first, then install both plugins into that environment:

```bash
cd <bart-plugin-checkout>
uv pip install -e .

cd <vllm-beamsearch-plugin-checkout>
uv pip install -e '.[stress]'
```

By default vLLM loads all installed `vllm.general_plugins`. If you restrict
plugin loading with `VLLM_PLUGINS`, include both plugin entry points:

```bash
export VLLM_PLUGINS=bart,beam_search
```

## One-Shot Nightly

To try the latest vLLM nightly without installing into the current environment,
run the server with both plugin checkouts installed into an isolated `uv run`
environment:

```bash
BART_PLUGIN=${BART_PLUGIN:-<bart-plugin-checkout>}
BEAM_PLUGIN=${BEAM_PLUGIN:-<vllm-beamsearch-plugin-checkout>}
MODEL=${MODEL:-facebook/bart-large-cnn}
SERVED_MODEL=${SERVED_MODEL:-bart}

CUDA_VISIBLE_DEVICES=0 \
VLLM_PLUGINS=bart,beam_search \
VLLM_USE_FLASHINFER_SAMPLER=0 \
UV_TORCH_BACKEND=auto \
uv run --isolated --python 3.12 \
  --with vllm \
  --with 'tokenizers==0.22.1' \
  --with-editable "${BART_PLUGIN}" \
  --with-editable "${BEAM_PLUGIN}" \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --prerelease allow \
  -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name "${SERVED_MODEL}" \
  --dtype float16 \
  --port 8005 \
  --scheduler-cls vllm_beam_search.scheduler.BeamSearchScheduler
```

For pinned GitHub refs, use `--with` instead of `--with-editable`.
Editable installs require a local checkout; Git refs are installed as normal
package sources. These defaults use the beam-search plugin `main` branch and
the BART plugin MRV2 cleanup branch; replace either ref with a commit hash for
an immutable pin:

```bash
BART_PLUGIN_REF=${BART_PLUGIN_REF:-codex/bart-mrv2-plugin-cleanup}
BEAM_PLUGIN_REF=${BEAM_PLUGIN_REF:-main}
MODEL=${MODEL:-facebook/bart-large-cnn}
SERVED_MODEL=${SERVED_MODEL:-bart}

CUDA_VISIBLE_DEVICES=0 \
VLLM_PLUGINS=bart,beam_search \
VLLM_USE_FLASHINFER_SAMPLER=0 \
UV_TORCH_BACKEND=auto \
uv run --isolated --python 3.12 \
  --with vllm \
  --with 'tokenizers==0.22.1' \
  --with "vllm-bart-plugin @ git+https://github.com/vllm-project/bart-plugin.git@${BART_PLUGIN_REF}" \
  --with "vllm-beam-search @ git+https://github.com/neuralmagic/vllm-beamsearch-plugin.git@${BEAM_PLUGIN_REF}" \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --prerelease allow \
  -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name "${SERVED_MODEL}" \
  --dtype float16 \
  --port 8005 \
  --scheduler-cls vllm_beam_search.scheduler.BeamSearchScheduler
```

## Start Server

```bash
MODEL=${MODEL:-facebook/bart-large-cnn}
SERVED_MODEL=${SERVED_MODEL:-bart}

CUDA_VISIBLE_DEVICES=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name "${SERVED_MODEL}" \
  --dtype float16 \
  --port 8005 \
  --scheduler-cls vllm_beam_search.scheduler.BeamSearchScheduler
```

Notes:

- `MODEL` can be a Hugging Face model ID or a local BART checkpoint path.
- The `tokenizers==0.22.1` pin avoids a current vLLM-nightly/BART tokenizer
  construction issue.

## Send Request

The BART plugin adapts simple OpenAI completion prompts into BART
encoder/decoder prompts. Beam search is enabled per request through
`vllm_xargs.beam_width`.

```bash
curl -s http://localhost:8005/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "bart",
    "prompt": "Summarize: Beam search keeps several likely continuations at each decoding step instead of committing to one token immediately.",
    "max_tokens": 64,
    "temperature": 0,
    "add_special_tokens": false,
    "vllm_xargs": {
      "beam_width": 4,
      "no_repeat_ngram_size": 3
    }
  }' | python -m json.tool
```

## Stress

After installing the beam plugin with the `stress` extra:

```bash
vllm-beam-stress \
  --base-url http://localhost:8005 \
  --model bart \
  --rounds 100 \
  --requests-per-round 32 \
  --concurrency 64 \
  --max-tokens 128 \
  --beam-width 4 \
  --no-repeat-ngram-size 3 \
  --abort-rounds 3
```

## Troubleshooting

- If the server does not recognize BART, verify the BART plugin is installed
  and loaded: `python -c "import vllm_bart_plugin"`.
- If beam search is not active, verify the request includes
  `vllm_xargs.beam_width` greater than 1.
- If plugins are restricted, `VLLM_PLUGINS` must include both `bart` and
  `beam_search`.
