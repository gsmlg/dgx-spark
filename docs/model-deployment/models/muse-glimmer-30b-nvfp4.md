# Muse Glimmer 30B NVFP4 on one DGX Spark

**Date:** 2026-09-15  
**Profile:** `muse-glimmer-30b-nvfp4`  
**State:** Configured candidate; no target-host qualification has been performed for this profile.

## Artifact and runtime

The profile pins `Inferact/Muse-Glimmer-30B-NVFP4-W4A4` at commit
`d35cb79050f419c457611b1cee5c5d15b176f285`. This is the vLLM recipe's
Blackwell-oriented DGX Spark choice: a 25.42 GB ModelOpt NVFP4 checkpoint with W4A4
quantization and group size 16. vLLM detects the checkpoint quantization metadata, so
no manual quantization flag is configured. The native combined context is 131,072
tokens. [M1, M2](../sources.md)

Muse Glimmer is a dense 29.6B vision-language model. This is the repository's first
multimodal profile: `language-model-only` is disabled and the API qualification suite
sends a generated local PNG data URL. Text-only profiles continue to require media
rejection. Video is supported upstream but is not part of this profile's qualified
contract. [M1, M4](../sources.md)

The candidate uses `vllm/vllm-openai:v0.28.0`, the minimum version identified by the
recipe. `prepare` must resolve that mutable tag to an immutable Linux ARM64 digest and
verify that both the reasoning and tool parser registries expose `muse_glimmer`. Only an
actual start establishes GB10 ModelOpt kernel compatibility. [M1](../sources.md)

## Initial policy

The configured target is 131,072 combined tokens and one scheduled request. The profile
reserves 70% of unified GPU memory, enables chunked prefill and prefix caching, and uses
the runtime's automatic BF16 KV-cache dtype. It does not enable speculative decoding.
These are conservative starting settings, not measured capacity or throughput.

Both `--reasoning-parser muse_glimmer` and `--tool-call-parser muse_glimmer` are required
because the model emits channel-scoped reasoning and ATEM-formatted tool calls. The
reasoning parser requires special tokens to remain visible internally; clients should
consume the parsed `reasoning` or `reasoning_content` fields instead of stripping or
reconstructing channel tokens. Preserve reasoning and tool messages in multi-turn
history. [M1](../sources.md)

Reasoning is intrinsic. Select `reasoning_strength` as `low`, `medium`, `high`, or
`xhigh` through `chat_template_kwargs`; the template defaults to `high`. The recipe's
recommended sampling is temperature 1.0, top-p 0.95 and top-k 64. Do not force greedy
sampling, and allow enough output tokens for the model to reach its final channel: a
tight output budget can produce reasoning with empty final content. [M1](../sources.md)

## Deployment and qualification

```sh
bin/spark-llm validate --profile muse-glimmer-30b-nvfp4
bin/spark-llm download --profile muse-glimmer-30b-nvfp4
bin/spark-llm prepare --profile muse-glimmer-30b-nvfp4
# Pause clients and choose a switch window before replacing the active model.
bin/spark-llm start --profile muse-glimmer-30b-nvfp4
bin/spark-llm test --mode api
```

Downloading and preparing do not stop the active model. Starting this profile does.
Validation, preparation, health and smoke generation are not 131K or multimodal release
acceptance. After the API suite passes, run the disruptive context, benchmark and soak
modes and complete the manual recovery review before accepting the exact release. Record
memory-floor, OOM, image-processing, parser, kernel and truncation failures instead of
silently changing the context or precision.
