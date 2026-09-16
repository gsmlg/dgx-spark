# Muse Glimmer 30B NVFP4 on one DGX Spark

**Date:** 2026-09-16
**Profile:** `muse-glimmer-30b-nvfp4`  
**State:** DFlash candidate deployed on the target host; health, generation, streaming and tool-call smoke passed. Context, benchmark, soak and recovery qualification remain pending.

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

The configured target is 131,072 combined tokens and up to 32 scheduled requests. The
profile reserves 70% of unified GPU memory, enables chunked prefill and prefix caching,
uses the V2 model runner, and sets KV-cache dtype to FP8. The
70% memory reservation is the Spark-specific guardrail retained from the working
baseline; it intentionally does not copy the generic recipe's more aggressive memory
fraction. These are configured settings, not measured capacity or throughput.

The vLLM 0.28.0 Rust frontend cannot be used for this profile: it does not register the
`muse_glimmer` tool parser and reports automatic tool choice as unimplemented. This
profile therefore retains the Python frontend so Muse reasoning and Codex tool calls
remain functional. Re-evaluate Rust only after the pinned runtime exposes the same Muse
parser contract.

DFlash is enabled with 15 speculative tokens and the separately pinned
`meta-models/Muse-Glimmer-30B-assistant` draft at commit
`e8192f3a8f617f74be2ce220360c89ef4789f39f`. The lifecycle downloads and hashes both
snapshots, rewrites the draft alias to its local immutable snapshot, and verifies both
again before startup. The draft is a five-layer assistant head, not a standalone model,
and must remain paired with this target checkpoint. [M1, M5](../sources.md)

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
Validation, preparation, health and smoke generation are not 131K, 32-way concurrency,
DFlash performance, or multimodal release acceptance. After the API suite passes, run
the disruptive context, benchmark and soak modes and complete the manual recovery review
before accepting the exact release. Record memory-floor, OOM, DFlash acceptance rate,
image-processing, parser, kernel and truncation failures instead of silently changing
the context, speculation, or precision.
