# Laguna S 2.1 NVFP4 on one DGX Spark

**Date:** 2026-09-10  
**Proposed profile:** `laguna-s-2.1-nvfp4`  
**State:** Candidate design; no target-host qualification performed here.

## 1. Artifact and engine choice

Use `poolside/Laguna-S-2.1-NVFP4` with vLLM. The publisher identifies this as a text model, documents native NVFP4 serving, and supplies a Spark-specific route. Its card also warns that an August update replaced the weights, not just configuration. Pin the selected complete revision rather than `main`. [L1](../sources.md)

The publisher's roughly 71 GB claim conflicts with the 99.7 GB total shown in the reviewed repository file listing. These measure different things and may reflect different revisions; do not treat the directory total as measured GPU residency or assume the smaller number is authoritative. Sum the selected snapshot's required files and inspect the safetensors weight index before estimating disk or memory. [L1, L2](../sources.md)

The reviewed card declares a 1,048,576-token context. That is a checkpoint property, not proof of million-token execution on this Spark. Use a smaller server cap for first qualification; do not rewrite pinned RoPE configuration or silently download older weights to make the new checkpoint fit. [L1](../sources.md)

## 2. Image selection and admission

First inspect the repo's existing `vllm/vllm-openai:v0.28.0` image candidate. Reuse it only if it passes all Laguna probes. The upstream minimum version is not proof that any newer ARM64 image has the needed GB10 kernels. Do not assume the unpublished `v0.28.0-cu130` alias exists; the repository already records that failure. [R1, L1](../sources.md)

Resolve a real `linux/arm64` image digest and full model/tokenizer commits during implementation. No digest or new model commit has been resolved by this documentation task. Preparation must reject unresolved placeholders, incompatible drivers or missing parsers before replacing the current service.

Check Python development headers, CUDA compiler availability where JIT requires it, compatible FlashInfer components, the actual FP4 kernel path, and offline model/template loading inside the container. Do not install Python ML packages or development headers on the host just because the publisher's bare-metal recipe does so.

The Spark recipe uses `MAX_JOBS=4` to bound compilation and `CUTE_DSL_ARCH=sm_121a` for its kernel stack. Carry supported controls into the profile's frozen runtime environment; verify them for the selected image rather than assume an old wheel recipe is universally valid. Probe and server environments must agree. [L1](../sources.md)

## 3. Proposed first-release configuration

These are integration defaults to test, not a copy of the publisher's throughput benchmark.

| Concern | Initial policy |
|---|---|
| Served alias | `local-assistant` |
| Tensor parallelism | 1 |
| Combined context | 5,120 tokens |
| Scheduled requests | 1 |
| GPU allocation candidate | `gpu-memory-utilization: 0.776`; eager execution avoids the measured CUDA-graph tuning spike |
| Prefill work | 64-token batch cap to bound measured operator-tuning workspace; chunked prefill remains enabled |
| Quantization | Detect checkpoint metadata; do not force an unrelated quantization mode |
| Reasoning/tool parsers | `poolside_v1` for both; validate through runtime and API tests |
| Sampling | Pinned checkpoint generation defaults, not the existing Qwen override mapping |
| KV precision | `fp8_e4m3`; verify the effective runtime dtype and 32K capacity |
| Speculation | Disabled |
| Input modalities | Text only |

Host validation on 2026-09-14 found that the 32,768-token FP8-KV candidate needed 1.35 GiB of KV cache while 0.80 utilization supplied 0.77 GiB. Raising utilization to 0.81 caused a sustained sub-16-GiB memory interval during kernel tuning. Warm-cache retries at 0.776 varied: one estimated 7,888 tokens, while a later restart estimated 5,632 tokens. The initial restart-tolerant baseline is therefore 5,120 tokens at 0.776; 32K remains a separate failed capacity result, not a qualified claim.

The Poolside parser choice and checkpoint-owned sampling follow the native serving recipe. Its example is not an instruction to inherit public binding or a high scheduled-request limit. [L3](../sources.md)

Keep normal runtime execution initially. Do not add an unverified linear-backend override or silently fall back to another quantization. If the selected FP8 cache/kernel combination fails, record the failure and create a new deliberate configuration for any alternative.

## 4. Reasoning and tool behavior

Test explicit thinking-on and thinking-off separately from the selected default. For coding evaluation, preserve the publisher's reasoning behavior; clients needing direct responses should use the model's documented `enable_thinking` control. Preserve returned reasoning across tool turns in the field accepted by the pinned runtime. Laguna supports interleaved reasoning, so the current harness's unconditional removal of reasoning history is inappropriate. [L4](../sources.md)

Freeze the behavior expectation in profile metadata. Count requests using exactly the same template options, tool definitions and retained history as generation. A short arithmetic request that exhausts its completion budget before the final answer is not evidence of a parser defect; record length termination and retry only under a deliberate test budget.

Do not carry Qwen's `generation-config: vllm` plus its overrides into this profile. Read and validate the pinned `generation_config.json`, including any output-limit field, and record the effective defaults. Avoid a hidden global completion cap.

## 5. Deployment sequence after implementation

The profile commands below are implemented. Preparation still has to prove the selected
image and pinned checkpoint on the target Spark.

```sh
bin/spark-llm validate --profile laguna-s-2.1-nvfp4
bin/spark-llm doctor
bin/spark-llm prepare --profile laguna-s-2.1-nvfp4
# Pause clients and select a test window before replacing the active model.
bin/spark-llm start --profile laguna-s-2.1-nvfp4
bin/spark-llm status
bin/spark-llm test --mode api
```

Review the prepared manifest, actual artifact size, parser/kernel probes and available accepted rollback target before `start`. That command may replace the owned active model; it must not terminate unrelated workloads.

After API success, and only with clients paused:

```sh
bin/spark-llm test --mode context --disruptive
bin/spark-llm test --mode benchmark --disruptive
bin/spark-llm test --mode soak --disruptive
```

Complete the manual recovery checks and operator evidence before the existing `accept --evidence` flow. A 32K acceptance is specifically a 32K acceptance. Keep startup restart policy disabled until then.

## 6. Larger contexts and DFlash

Try 65,536, 131,072 and 262,144 combined tokens in separate immutable releases after the initial baseline. For each, remeasure memory, token-count agreement, retrieval, forced boundary behavior and recovery. Preserve a failed larger-context result; never relabel the previous passing result.

DFlash is optional. The documented companion is `poolside/Laguna-S-2.1-DFlash-NVFP4`; add it only with its own full revision, hashes, local snapshot path and disk budget. Validate the selected runtime's `method: dflash` schema and supported speculative-token count. The publisher warns about unsupported sampling combinations under speculation; revalidate defaults and API errors instead of copying Qwen's `min_p` setting. [L1](../sources.md)

Use the non-speculative release as the quality and performance reference. Measure acceptance length, output quality, TTFT, decode speed and memory under identical prompts. Do not import an upstream tokens-per-second result into this repo as a local benchmark.

## 7. Stop conditions

Block promotion on unavailable ARM64 kernels/parsers, revision/manifest inconsistencies, JIT memory spikes, missing offline code dependencies, reasoning history corruption, memory-floor violations or failed recovery. A different GGUF/Ollama deployment is not a successful implementation of this requested NVFP4 profile.
