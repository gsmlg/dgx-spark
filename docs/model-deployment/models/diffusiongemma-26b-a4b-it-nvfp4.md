# DiffusionGemma 26B-A4B IT NVFP4 on one DGX Spark

**Date:** 2026-09-15  
**Profile:** `diffusiongemma-26b-a4b-it-nvfp4`  
**State:** Configured candidate; no target-host qualification has been performed for this profile.

## Artifact and runtime

The profile pins `nvidia/diffusiongemma-26B-A4B-it-NVFP4` at commit
`ec4ff3df205028f4e81c954c2227f9312b3ec2ea`. It is NVIDIA's ModelOpt NVFP4 W4A4
export of Google's 26B-total, 4B-active block-diffusion Gemma 4 MoE. The pinned
configuration declares a 262,144-token language context, group-size-16 four-bit weights
and activations, and an FP8 KV-cache scheme. Quantization is detected from checkpoint
metadata. [D1, D2](../sources.md)

The recipe requires a diffusion-enabled vLLM build and selects
`vllm/vllm-openai:gemma`. Its current registry manifest includes Linux ARM64; `prepare`
must pin the mutable tag to an immutable digest, execute CUDA, and verify the diffusion,
parser, attention and loader features before creating a release. [D1](../sources.md)

## Diffusion and serving policy

Unlike autoregressive Gemma, this model denoises a fixed token canvas in parallel. The
profile follows the recipe's DGX Spark override: TP1, 80% GPU-memory utilization, eight
scheduled sequences, `TRITON_ATTN`, `fastsafetensors`, prefix caching and a 256-token
canvas. Chunked prefill is enabled with a conservative 4,096-token scheduler batch.
Configured 262K capacity and eight-way execution remain qualification targets.

The checkpoint's generation configuration caps `max_new_tokens` at 256. The profile
loads its denoising and entropy-sampler settings but explicitly clears only that cap with
`max_new_tokens: null`, preserving request-level output limits. The lifecycle validator
permits only this null override; it still rejects a server-wide numeric output cap.
[D1, D3](../sources.md)

This profile accepts text and image input. Thinking defaults on and remains request
controllable through `chat_template_kwargs.enable_thinking`; both reasoning and tools use
the `gemma4` parsers. Tool calling is reported upstream as more reliable with thinking
enabled. Audio is unsupported, and video is outside this profile's qualification scope.
The shared API suite exercises a generated local PNG. [D1, D4](../sources.md)

Block diffusion trades a slower first token for parallel generation throughput. Do not
apply upstream benchmark multipliers to this host. Specifically qualify streaming,
cancellation, request-level output limits, overlapping requests, tool history and the
large diffusion-state allocation before accepting a release.

## Deployment and qualification

```sh
bin/spark-llm validate --profile diffusiongemma-26b-a4b-it-nvfp4
bin/spark-llm download --profile diffusiongemma-26b-a4b-it-nvfp4
bin/spark-llm prepare --profile diffusiongemma-26b-a4b-it-nvfp4
# Pause clients and choose a switch window before replacing the active model.
bin/spark-llm start --profile diffusiongemma-26b-a4b-it-nvfp4
bin/spark-llm test --mode api
```

Downloading and preparing do not stop the active model. Starting this profile does.
Validation and preparation are not startup, image, 262K, concurrency, throughput, soak
or recovery acceptance. Retain OOM, diffusion-buffer, loader, parser, multimodal,
streaming and long-context failures rather than silently reducing the policy.
