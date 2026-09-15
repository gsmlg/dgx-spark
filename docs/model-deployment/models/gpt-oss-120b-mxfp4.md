# GPT-OSS 120B MXFP4 on one DGX Spark

**Date:** 2026-09-15  
**Profile:** `gpt-oss-120b-mxfp4`  
**State:** Configured candidate; no target-host qualification has been performed for this profile.

## Artifact and runtime

The profile pins `openai/gpt-oss-120b` at commit
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`. The checkpoint configuration declares
the GPT-OSS architecture, native MXFP4 mixture-of-experts weights and a 131,072-token
context limit. The model is text-only and is designed for configurable reasoning and
tool use. [G1, G2](../sources.md)

The candidate reuses the repository's `vllm/vllm-openai:v0.28.0` tag. Its installed
runtime exposes GPT-OSS's `openai_gptoss` reasoning parser and the `openai` tool-call
parser. `prepare` still must resolve the tag to an immutable Linux ARM64 digest, run
the CUDA and CLI probes, and verify the pinned model snapshot before creating a release.
Only an actual start and qualification run establish GB10 kernel compatibility. [G3, G4](../sources.md)

## Initial policy

The configured combined context is the native 131,072-token limit with one scheduled
request. This is a qualification target, not measured capacity. The profile reserves
70% of unified GPU memory, enables chunked prefill, uses an FP8 E4M3 KV cache and a
4,096-token scheduler batch cap. Prefix caching starts disabled, matching the vLLM
GPT-OSS measurement recipe's consistency guidance. [G3](../sources.md)

Reasoning is intrinsic to GPT-OSS rather than controlled by the other profiles'
`enable_thinking` template switch. vLLM extracts reasoning with `openai_gptoss`;
clients may select the model's supported reasoning effort in requests. Tool calling
uses the `openai` parser and automatic tool selection. Preserve reasoning and tool
messages in multi-turn conversations and qualify the behavior against the exact release.

## Deployment and qualification

```sh
bin/spark-llm validate --profile gpt-oss-120b-mxfp4
bin/spark-llm download --profile gpt-oss-120b-mxfp4
bin/spark-llm prepare --profile gpt-oss-120b-mxfp4
# Pause clients and choose a switch window before replacing the active model.
bin/spark-llm start --profile gpt-oss-120b-mxfp4
bin/spark-llm test --mode api
```

Downloading and preparing do not stop the active model. Starting this profile does.
Do not treat validation, preparation, health, or smoke generation as 131K acceptance.
With clients paused, run the disruptive context, benchmark and soak modes, then complete
the manual recovery review before accepting the exact release. Retain any memory-floor,
OOM, parser, kernel or long-context failure instead of silently changing context or
precision.
