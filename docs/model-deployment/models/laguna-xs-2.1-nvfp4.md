# Laguna XS 2.1 NVFP4 on one DGX Spark

**Date:** 2026-09-14  
**Profile:** `laguna-xs-2.1-nvfp4`  
**State:** Configured candidate; no target-host qualification has been performed for this profile.

## Artifact and runtime

The profile pins `poolside/Laguna-XS-2.1-NVFP4` at commit
`d32afde8b09af1539b49ff96ff5551c674485f8e`. The selected snapshot declares a
33B-parameter, 3B-active text model, NVFP4 weights, an FP8 KV cache and a 262,144-token
native context. vLLM detects the checkpoint quantization metadata; the profile does not
force a separate weight-quantization mode. [X1, X2](../sources.md)

The candidate uses the repository's `vllm/vllm-openai:v0.28.0` tag. `prepare` must
resolve it to a real Linux ARM64 digest and verify CUDA, the `poolside_v1` parsers and
all rendered CLI flags before creating an immutable release. The pinned runtime is new
enough for the model card's stated Laguna XS and FP8-KV minimum versions, but only an
actual prepare/start test establishes GB10 kernel compatibility. [X1](../sources.md)

## Initial policy

The configured combined context is 262,144 tokens with one scheduled request. This is
the checkpoint's native limit and a qualification target, not measured capacity. The
profile reserves 70% of unified GPU memory, enables chunked prefill and prefix caching,
uses a 4,096-token scheduler batch cap, and keeps speculative decoding disabled.

Reasoning is disabled by default for consistency with the service's existing client
behavior and can be enabled per request with
`chat_template_kwargs: {"enable_thinking": true}`. Both reasoning and tool calling use
`poolside_v1`; multi-turn tool tests must preserve interleaved reasoning. The checkpoint
generation defaults are loaded, while the explicit chat-template setting owns the
server's default reasoning mode. [X1, X3](../sources.md)

## Deployment and qualification

```sh
bin/spark-llm validate --profile laguna-xs-2.1-nvfp4
bin/spark-llm download --profile laguna-xs-2.1-nvfp4
bin/spark-llm prepare --profile laguna-xs-2.1-nvfp4
# Pause clients and choose a switch window before replacing the active model.
bin/spark-llm start --profile laguna-xs-2.1-nvfp4
bin/spark-llm test --mode api
```

Do not treat validation, preparation, health, or smoke generation as 262K acceptance.
With clients paused, run the disruptive context, benchmark and soak modes, then complete
the manual recovery review before accepting the exact release. Retain any memory-floor,
OOM, parser, kernel or long-context failure instead of silently reducing the configured
limit or changing precision.
