# Ornith 1.5 35B-A3B NVFP4 on one DGX Spark

**Date:** 2026-09-20

**Profile:** `ornith-1.5-35b-a3b-nvfp4`

**State:** Prepared release `af366d840c1b6ae81a5ecd3b` is running and passed startup smoke; full target-host qualification is pending.

## Artifact and runtime

The profile pins the official `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` checkpoint at
commit `94e431d9cc47fa1986a7a1a4e9a80f7f118b03aa`. It is the ModelOpt NVFP4 export
of the 35B-total, approximately 3B-active Qwen3.5-MoE-based model. The pinned weight
index reports 23,407,457,492 bytes. Its language configuration declares a native
262,144-token window; the repository also contains the Qwen3.5 vision encoder. The
quantization metadata uses group-size-16 W4A16 NVFP4 for the MoE/MLP weights and FP8
for other linear layers. [O1, O2](../sources.md)

The candidate runtime is `vllm/vllm-openai:v0.28.0`. The locally cached ARM64 image
reports vLLM 0.28.0, registers `Qwen3_5MoeForConditionalGeneration`, and exposes the
`qwen3_xml` tool parser. Its GPU-backed full-help probe also exposes `qwen3`,
`qwen3_xml`, prompt-token details, multimodal language-model control and remote-code
flags. `prepare` must repeat the CUDA/help probes and pin the current image digest.
The publisher requires vLLM 0.19.1 or newer. [O1, O4](../sources.md)

On the target GB10, vLLM selected `MarlinNvFp4LinearKernel` and its MARLIN NVFP4 MoE
backend, plus FlashInfer for the FP8 linear and attention paths. vLLM also warned that
this W4A16 checkpoint uses weight-only FP4 compression rather than native FP4 compute
on the detected device. Preserve that runtime distinction when describing acceleration.

The upstream command includes `--trust-remote-code`, but this pinned repository has no
Python implementation files and the selected vLLM image has native architecture
support. The profile deliberately does not authorize remote checkpoint code execution.
If a later checkpoint actually requires custom code, treat that as a reviewed profile
change rather than silently enabling it.

## Initial single-Spark policy

The upstream BF16 example targets two 80 GB GPUs and 90% memory utilization. This
single-GB10 profile instead uses the official NVFP4 checkpoint with TP1, one scheduled
request, a 4,096-token scheduler batch and 70% GPU-memory utilization. The initial 80%
setting exposed ample KV capacity but missed the host's 16 GiB available-memory floor
during smoke validation, so it was rejected rather than weakening the guardrail. Chunked prefill,
prefix caching, per-request cache reporting and FP8 KV cache are enabled. These are
initial qualification settings, not measured capacity or performance claims. [O1](../sources.md)

Text and image input are in scope. The template also renders video markers, but video
and audio are outside this profile's qualification contract. Reasoning defaults on and
is parsed with `qwen3`; clients can disable it per request with
`chat_template_kwargs: {"enable_thinking": false}`. Automatic tool calls use the
publisher's `qwen3_xml` parser. Preserve parsed reasoning and tool messages across
turns. General-task generation defaults are temperature 0.6, top-p 0.95 and top-k 20;
the upstream benchmark setting of temperature 1.0 is not the serving default. [O1, O3](../sources.md)

The publisher documents YaRN extension beyond the native window. This profile does not
enable it: static scaling can reduce ordinary-input quality, and a larger configured
window would require a separate memory, retrieval and recovery qualification.

## Deployment and qualification

```sh
bin/spark-llm validate --profile ornith-1.5-35b-a3b-nvfp4
bin/spark-llm download --profile ornith-1.5-35b-a3b-nvfp4
bin/spark-llm prepare --profile ornith-1.5-35b-a3b-nvfp4
# Pause clients and choose a switch window before replacing the active model.
bin/spark-llm start --profile ornith-1.5-35b-a3b-nvfp4
bin/spark-llm test --mode api
```

Downloading and preparing do not stop the active model. Starting this profile does.
Release `af366d840c1b6ae81a5ecd3b` loaded the checkpoint, passed health plus English,
streaming and tool-call smoke, and retained about 27 GiB host-available memory afterward.
This is not proof of the full API suite, multimodal correctness, 262K execution,
throughput, stability or recovery. Qualify the exact release and retain loader, OOM,
parser, image, long-context and memory-floor failures rather than silently reducing the
declared policy.
