# Gemma 4 26B-A4B NVFP4 on one DGX Spark

**Date:** 2026-09-15  
**Profile:** `gemma-4-26b-a4b-nvfp4`  
**State:** Configured candidate; no target-host qualification has been performed for this profile.

## Artifact and runtime

The profile pins `nvidia/Gemma-4-26B-A4B-NVFP4` at commit
`a19cfe00be84568a6867111c9a68c9c44fdcffe6`. It is NVIDIA's ModelOpt NVFP4 W4A4
export of Google's 26B-total, 4B-active Gemma 4 mixture-of-experts model. The pinned
configuration declares a 262,144-token context and group-size-16 four-bit weights and
activations. vLLM detects this metadata; no manual quantization flag is used. [E1, E2](../sources.md)

The DGX Spark recipe prescribes `eugr/spark-vllm:nightly-20260704`, because the generic
runtime candidates tested by the recipe did not serve this checkpoint. The tag currently
resolves to a Linux ARM64 manifest; `prepare` must pull it, pin its immutable digest, run
a CUDA operation, and prove the `gemma4` parsers and `fastsafetensors` loader exist before
creating a release. The adapter invokes `vllm serve` explicitly so the image's NVIDIA
wrapper entrypoint cannot reinterpret the config arguments. [E1](../sources.md)

## Initial policy

This is a multimodal text-and-image profile. It uses the recipe's DGX Spark settings:
TP1, 80% GPU-memory utilization, eight scheduled sequences, an 8,192-token scheduler
batch, prefix caching, the Rust API frontend, the v2 model runner and `fastsafetensors`.
Chunked prefill is explicitly enabled. The KV cache stays at the checkpoint/runtime automatic dtype.
Configured 262K capacity is a target, not a measured claim.

Reasoning defaults off and can be enabled per request with
`chat_template_kwargs: {"enable_thinking": true}`. Both reasoning and automatic tool
calling use the `gemma4` parser. The pinned generation configuration supplies temperature
1.0, top-p 0.95 and top-k 64. Preserve parsed reasoning and tool messages across turns.
The shared API suite exercises a generated local PNG rather than fetching remote media.
Audio and video are outside this profile's qualification contract. [E1, E3, E4](../sources.md)

The recipe also describes four-token MTP drafting with
`google/gemma-4-26B-A4B-it-assistant` and a Triton MoE backend. This profile intentionally
omits it: the current lifecycle pins and verifies one model repository per release. Add
multi-snapshot integrity support and qualify the non-speculative baseline before enabling
a draft checkpoint.

## Deployment and qualification

```sh
bin/spark-llm validate --profile gemma-4-26b-a4b-nvfp4
bin/spark-llm download --profile gemma-4-26b-a4b-nvfp4
bin/spark-llm prepare --profile gemma-4-26b-a4b-nvfp4
# Pause clients and choose a switch window before replacing the active model.
bin/spark-llm start --profile gemma-4-26b-a4b-nvfp4
bin/spark-llm test --mode api
```

Downloading and preparing do not stop the active model. Starting this profile does.
Validation and preparation are not startup, image, 262K, concurrency, throughput, soak
or recovery acceptance. Qualify the exact prepared release and retain memory-floor, OOM,
loader, image-processing, parser and long-context failures rather than silently reducing
the policy.
