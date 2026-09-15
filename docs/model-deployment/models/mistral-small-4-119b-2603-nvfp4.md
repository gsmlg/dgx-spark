# Mistral Small 4 119B 2603 NVFP4 on one DGX Spark

**Date:** 2026-09-15
**Profile:** `mistral-small-4-119b-2603-nvfp4`
**State:** Configured candidate; no target-host qualification has been performed for this profile.

## Artifact and runtime

The profile pins `mistralai/Mistral-Small-4-119B-2603-NVFP4` at commit
`45331841b631f4e281df8e959ea3cc9beb84298a`. The checkpoint is a compressed-tensors
NVFP4 export of the multimodal 119B-total, 6.5B-active mixture-of-experts model and the
pinned repository occupies about 70.8 GB. Its published serving target is 262,144 tokens.
The profile uses vLLM's `mistral` reasoning and tool parsers and `TRITON_MLA` attention.
[MS1, MS2](../sources.md)

The candidate image is `vllm/vllm-openai:v0.28.0`; `prepare` must resolve its immutable
Linux ARM64 digest, execute CUDA, and prove the required CLI options and parser/backend
names are present. The publisher requires current vLLM and `mistral_common >= 1.11.0`.
The help probe is only an early rejection gate: successful model loading and API tests
must still demonstrate actual compatibility. [MS1](../sources.md)

## Initial single-Spark policy

The publisher's generic example uses TP2, 128 sequences, a 16,384-token scheduler batch
and 80% GPU-memory utilization. A DGX Spark exposes one GB10 GPU, so this profile adapts
that to TP1, one scheduled sequence and a 4,096-token batch while retaining the 80%
memory target, chunked prefill and prefix caching. KV cache dtype remains automatic.
The 262K setting is a qualification target, not a measured capacity claim.

Text and image input are enabled. Reasoning defaults off. Clients enable it with the
top-level OpenAI-compatible request field `reasoning_effort: "high"` and temperature 0.7;
`reasoning_effort: "none"` selects direct response mode, for which the publisher allows
temperature 0.0 through 0.7. The server candidate defaults to 0.1. Tool calls execute in
the client, and returned reasoning and tool messages must be preserved across turns.
[MS1, MS3](../sources.md)

## Deployment and qualification

```sh
bin/spark-llm validate --profile mistral-small-4-119b-2603-nvfp4
bin/spark-llm download --profile mistral-small-4-119b-2603-nvfp4
bin/spark-llm prepare --profile mistral-small-4-119b-2603-nvfp4
# Pause clients and choose a switch window before replacing the active model.
bin/spark-llm start --profile mistral-small-4-119b-2603-nvfp4
bin/spark-llm test --mode api
```

Downloading and preparing do not stop the active model. Starting this profile does.
Validation and preparation are not startup, image, 262K, memory-headroom, concurrency,
throughput, soak or recovery acceptance. Qualify the exact prepared release and retain
OOM, swap, loader, MLA, image-processing, parser and long-context failures rather than
silently reducing the declared policy.
