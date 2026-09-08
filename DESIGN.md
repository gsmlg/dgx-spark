# Spark LLM — Deployment Design

**Version:** 1.0  
**Date:** 2026-09-08  
**Status:** Implementation specification; not yet validated on the target host  
**Initial profile:** `qwen38-27b-nvfp4`  
**Companion document:** [PRD.md](PRD.md)

## 1. Objective and scope

Deploy `Inferact/Qwen3.8-27B-NVFP4` on Jonathan’s single DGX Spark as a persistent, private, daily-use inference service. Use a prebuilt Docker image and Docker Compose, expose the native vLLM API, and retain a configuration boundary that allows other models to be introduced later.

Version 1 ships **one model profile only**. It does not deploy Gemma, a BF16 reference model, another inference engine, or a second GPU workload. It adds no gateway, database, web UI, Kubernetes deployment, model registry service, or custom Docker build.

The key requirement is **262,144 tokens of combined input and output**, not merely a container that starts with a large context argument.

### Fixed product decisions

| Area | Decision |
|---|---|
| Host | One DGX Spark; use the existing Docker installation and model-cache location |
| Runtime | vLLM in a native `linux/arm64` container |
| Model | `Inferact/Qwen3.8-27B-NVFP4`; pin a full repository commit |
| Context | Explicit `max-model-len: 262144`; no automatic reduction |
| Workload | Text chat, streamed responses, coding assistance, and function/tool calling |
| Scheduling | Start with `max-num-seqs: 1` |
| Identity | Compose project `spark-llm`, service `llm`, API alias `local-assistant` |
| Availability | Persistent service; planned replacement causes downtime |
| Configuration | Shared container infrastructure plus a model-specific profile |
| Deployment | Prepare artifacts, validate, start, test, then record an accepted release |

## 2. Evidence and validation boundary

The checkpoint configuration declares `max_position_embeddings: 262144`. This supports a native 256K target without extending RoPE. It does **not** establish memory consumption or latency for a particular engine build. [^S1]

NVIDIA documents Spark as an Arm system with 128 GB of unified CPU/GPU memory. This design therefore budgets against observed host memory pressure as well as engine allocations. [^S2]

The initial image candidate is `vllm/vllm-openai:v0.28.0-cu130`, a published CUDA 13.0 release artifact. Its publication is verified; the selected image manifest, GB10 kernels, installed driver, and full-length execution still require host validation. The deployed image must be pinned by digest. [^S3]

The vLLM Qwen recipe supplies the initial `qwen3` reasoning parser and `qwen3_coder` tool parser. Hardware-specific results in that recipe are not Spark acceptance results. [^S4]

**No design setting below is a measured performance claim.** A failed 256K test must remain a failed requirement, not be hidden by changing the configured context to 64K or 128K.

## 3. Architecture and responsibility boundaries

Clients connect directly to the native vLLM server. The model engine and its worker processes live inside one long-running Compose service. A small host-side lifecycle command operates Docker; it is not another resident service.

| Component | Owns | Must not own |
|---|---|---|
| `compose.yaml` | GPU access, mounts, networking, restart policy, logs, healthcheck | Model-family parsers or hard-coded model identity |
| Model profile | Image selection, checkpoint, serving parameters, generation defaults | Host-specific storage or credentials |
| Host configuration | Cache paths, bind address, port, operational timeouts | Model architecture settings |
| Lifecycle wrapper | Preflight, artifact preparation, validated launch, status, replacement, rollback | HTTP proxying, inference scheduling, conversation storage |
| vLLM | Tokenization, inference, scheduling, API response serialization | Executing the client’s tools or persisting chat history |
| Test harness | API checks, long-context fixtures, lifecycle tests, evidence reports | A second production inference service |

Keep configuration parsing and validation separate from side effects. Prefer a thin shell wrapper around Docker Compose, with small test utilities as needed. Do not create an application framework for this deployment.

### Repository and local data layout

| Path | Purpose |
|---|---|
| `compose.yaml` | The single common service definition |
| `profiles/qwen38-27b-nvfp4/image.env` | Candidate image reference and runtime-specific environment settings |
| `profiles/qwen38-27b-nvfp4/vllm.yaml` | Native vLLM model and serving configuration |
| `host.env.example` | Documented machine-specific options |
| `host.env` | Local paths, bind address and operational settings; untracked |
| `secrets.env` | Optional download credential and API credential; untracked and permission-restricted |
| `bin/spark-llm` | Lifecycle interface |
| `tests/` | Smoke, context, performance and recovery tests |
| `state/releases/<release-id>/` | Immutable resolved configuration, artifact identities and validation metadata |
| `state/active.json` | Active and previous accepted release references; atomically replaced |
| `reports/` | Sanitized test results and environment reports |
| Existing `$HF_CACHE` | Persistent checkpoint/tokenizer artifacts, outside Git |
| Configured runtime-cache root | Persistent vLLM compilation artifacts, outside Git |

Release state and reports are local generated data, not a database. The source profile expresses intent; a resolved release records exactly what is launched. Do not modify an already accepted release in place.

## 4. Configuration contract

The model profile owns the checkpoint, tokenizer revision, context limit, parsers, and generation behavior. Host configuration owns networking, storage and lifecycle timeouts. Reject duplicate settings across owners instead of silently applying an arbitrary precedence rule.

Model and tokenizer must resolve to the same pinned snapshot unless a deliberately documented exception exists. Version 1 has no such exception. Store the image digest, full model revision, effective configuration hash, runtime versions and selected GPU/backend information in the release record.

A mounted YAML file must contain literal resolved values. Do not assume Docker Compose substitutes environment variables inside that file. Any generated configuration must use explicit, validated substitutions and be written atomically.

Preserve the chosen image’s entrypoint contract. For an image that already launches vLLM, pass the configuration/server arguments rather than prepending a duplicate `vllm serve`. Inspect the actual image entrypoint during preparation. Persist the runtime compile cache independently of the weights cache. [^S5]

### Initial Qwen serving policy

These values are the initial configuration to validate, not parameters to force onto future models.

| Setting | Initial value or policy |
|---|---|
| Model | `Inferact/Qwen3.8-27B-NVFP4` |
| Revision / tokenizer revision | Full pinned commit, resolved during preparation |
| Served model name | `local-assistant` |
| Tensor parallelism | `1` |
| Maximum model length | `262144` |
| Maximum scheduled sequences | `1` |
| GPU memory utilization | `0.70` |
| Maximum batched tokens | `4096` |
| Chunked prefill | Enabled |
| Prefix caching | Enabled; validate the hybrid-attention path |
| Weight quantization | Read the checkpoint’s quantization metadata; do not requantize |
| Weight dtype / KV-cache dtype | `auto` / `auto`; record resolved values |
| Recurrent-state cache precision | Runtime/model defaults; record effective configuration |
| Attention/GDN backend | Automatic selection initially; record selected backend |
| CUDA graphs | Normal runtime behavior; no blanket eager-mode override |
| Speculative decoding | Disabled |
| CPU or external KV offloading | Disabled |
| Reasoning parser | `qwen3` |
| Tool parser / automatic choice | `qwen3_coder` / enabled |
| Input modalities | Text only; use the selected runtime’s language-model-only setting |
| Thinking default | Disabled; permitted as an explicit per-request option |

In vLLM, KV precision is separate from weight quantization, batching limits are not context limits, and `max-num-seqs` controls scheduled sequences rather than HTTP admission. Additional requests can be accepted and wait. This is not a bounded-queue or multi-user capacity guarantee. [^S6]

The intended operating pattern is one active generation from trusted clients. Test two overlapping requests for sane waiting and cancellation behavior, but do not advertise two simultaneous 256K generations.

### Generation behavior

Apply the non-thinking sampling preset explicitly rather than inheriting an unrelated checkpoint default. Initial values follow Qwen’s recommendations: temperature `0.7`, top-p `0.80`, top-k `20`, presence penalty `1.5`, minimum-p `0`, and repetition penalty `1.0`. The thinking preset uses temperature `1.0`, top-p `0.95`, top-k `20`, and presence penalty `0`. [^S7]

Use the native server chat-template default for non-thinking. Clients enabling thinking must also select the thinking sampling preset; do not assume the server automatically changes sampling when a template option changes.

The documented client default completion budget is 4,096 tokens. Larger explicit budgets, including 16,384, are supported when the combined context fits. This client default is **not** a global server cap. Avoid inadvertently imposing it through `max_new_tokens` in a generation configuration, which vLLM treats as a server-wide output limit. [^S8]

## 5. Context and memory policy

The token budget is:

**Formatted input tokens + requested completion budget ≤ 262,144.**

Formatted input includes the system message, conversation history, tool definitions/results, chat-template markers and any preserved reasoning included in the request. Generated reasoning also consumes completion capacity. For a 16,384-token completion budget, the formatted input budget is 245,760 tokens. The harness must count using the deployed tokenizer and template, not character counts or whitespace estimates. [^S7]

No silent input truncation, automatic context reduction, RoPE extension, or extra model download is allowed to make a failed test pass. Any cache-precision or backend adjustment creates a new resolved configuration and requires revalidation.

`gpu-memory-utilization: 0.70` is a starting allocation policy, not a whole-machine memory limit. Do not interpret the remaining fraction as guaranteed free operating-system memory. Similarly, a raw attention-KV calculation is not a complete allocation estimate: recurrent states/checkpoints, graph capture, runtime workspaces and host allocations must be measured. [^S6]

Adopt an initial operational guardrail of at least **16 GiB of observed host `MemAvailable`** during qualification, with no host OOM event or sustained swap-in/swap-out under inference. This is a proposed safety margin, not a hardware specification. Adjust it only through a documented qualification decision.

Keep other substantial GPU inference jobs stopped during qualification and normal use of this profile. The wrapper must identify conflicts and refuse to proceed; it must not kill unrelated processes or silently change other containers’ restart policies.

## 6. Host preflight and artifact preparation

Preflight records the actual OS, CPU architecture, Docker/Compose versions, NVIDIA driver, Container Toolkit, available disk space and current GPU workloads. Do not use historical driver information as the current host state.

Validate native ARM64 image availability, GPU visibility and a real CUDA operation inside the selected image. The first model-generation test must also validate the selected NVFP4 execution path; a successful `nvidia-smi` check alone is insufficient.

NVIDIA documents compatibility restrictions involving driver features and PTX even where a CUDA minor-compatibility baseline is met. Do not bypass image compatibility checks or automatically upgrade the host driver. Report an actionable incompatibility instead. [^S9]

Preparation downloads a complete pinned model snapshot, including tokenizer/template/configuration files, and verifies its integrity and completeness. Pull and retain the image digest. Check disk space for the download, image extraction, runtime cache and a configurable reserve before writing large artifacts.

Normal startup uses local prepared artifacts and must succeed without Hugging Face availability. Configure the relevant libraries for local/offline resolution. This is an artifact-availability guarantee, not a claim that host networking is an air gap. Do not provide a Hugging Face download token to the long-running container when it is unnecessary.

## 7. Container operation and API contract

Use host networking and host IPC. Bind to `127.0.0.1:8000` by default; expose a selected private interface only through explicit host configuration. No Docker port publishing is used with host networking. GPU access must use the documented Compose device reservation mechanism. [^S10]

Mount the resolved configuration read-only. Reuse the existing Hugging Face cache without copying weights into the image. Keep compilation caches in a namespace derived from image/runtime identity and relevant profile configuration. No privileged container or Docker-socket mount is required.

The supported client contract is `GET /v1/models` and `POST /v1/chat/completions`, including streamed text and function/tool-call messages. Expose native health/metrics for operations on the same private trust boundary. Record the actual model repository and release identity in lifecycle status; an alias must not obscure them operationally.

This is not a promise of complete compatibility with every provider API or agent CLI. Responses API, embeddings, audio and image/video requests are outside the v1 contract. Tool execution remains the client’s responsibility. Validate the exact response and reasoning fields produced by the pinned runtime rather than patching arbitrary responses in a new proxy. [^S11]

Credentials must be excluded from Git, command output, release reports and diagnostics. When exposing the private endpoint, configure native API authentication where supported, but do not assume it protects every operational route. Never expose the service directly to the public internet. [^S14]

## 8. Lifecycle and recovery

The lifecycle interface provides `doctor`, `prepare`, `start`, `stop`, `restart`, `status`, `logs`, `test`, `upgrade` and `rollback`. Repeating a successful operation should converge on the same state rather than create duplicate containers.

**Preparation** is non-disruptive: validate and download the candidate without changing the running service. **Start** resolves only prepared, pinned artifacts, creates the single service, checks health and model identity, and performs a small generation/tool-call smoke test. Mark the release running only after these checks succeed.

Distinguish runtime readiness from qualification. A release can be running and healthy while its 256K qualification is still pending. Status must report both, and promotion to accepted daily use requires the PRD acceptance suite.

Serialize all mutating operations with a host lock. For an upgrade, retain the previous accepted release and artifacts, stop the old engine, verify resource release, then start and test the candidate. If it fails, stop the candidate and restore the previous accepted release. Report the upgrade as failed even when rollback succeeds. On first installation there may be no rollback target.

For qualification of a new daily release, keep clients paused until acceptance completes. No zero-downtime cutover or uninterrupted request guarantee is claimed.

Configure a positive native shutdown wait and a longer Docker stop grace period. Initial values are 300 seconds and 330 seconds respectively. Long requests may still exceed this budget; record forced termination rather than claiming they drained. Separate startup, health-probe, request and shutdown timeouts. [^S8]

Use Docker `unless-stopped` after initial successful startup: running services return after a daemon/host restart, while an intentional stop remains stopped. During candidate qualification, use an explicit launch without unattended restart loops; promote the restart policy only after validation. Do not add a competing host process supervisor. [^S12]

A failed Docker healthcheck marks a container unhealthy; it is not, by itself, a process-exit restart policy. Version 1 reports sustained unhealthy state and provides an explicit restart operation, rather than adding an auto-healing daemon. [^S12][^S13]

## 9. Observability and validation evidence

Keep bounded Docker logs, proposed at three files of 20 MiB each. Disable prompt/output-body logging and keep engine statistics available. Logs and reports must identify configuration/backend errors without copying credentials or personal conversation contents. [^S8]

Status reports container state, health, restart count, active/previous release, checkpoint revision, image digest, API alias, effective context, scheduler limit and qualification status. Treat the 256K test outcome as independent evidence, not a value inferred from configuration.

Qualification reports include actual prompt/generated token counts, time to first token, generation rate, total latency, minimum host available memory, memory-pressure observations, and runtime failures. Record cold and warm runs separately. Do not claim prefix-cache acceleration from a weight-cache hit; verify engine cache behavior separately.

Short-request measurements establish the daily-use baseline. Long-request measurements establish capacity and its latency cost. No tokens-per-second or 256K first-token-time promise is made before measurements exist.

## 10. Future model extension

Later models add another directory under `profiles/`, potentially with another image digest, parsers, tokenizer/template behavior and memory settings. Reuse the same lifecycle, cache conventions, API endpoint and service identity.

Version 1 does not implement or validate a Gemma profile. It only avoids embedding Qwen assumptions in common infrastructure. Future model selection remains an explicit stop-and-replace operation with model-specific tests.

## References

Sources checked on 2026-09-08. Live documentation may advance; implementation must validate against the selected image’s actual version.

[^S1]: Inferact checkpoint configuration — https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4/blob/main/config.json
[^S2]: NVIDIA DGX Spark hardware — https://docs.nvidia.com/dgx/dgx-spark/hardware.html
[^S3]: vLLM 0.28.0 release artifacts — https://github.com/vllm-project/vllm/releases/tag/v0.28.0
[^S4]: vLLM Qwen3.8-27B recipe — https://recipes.vllm.ai/Qwen/Qwen3.8-27B
[^S5]: vLLM Docker deployment and persistent compile cache — https://docs.vllm.ai/en/stable/deployment/docker/
[^S6]: vLLM engine arguments — https://docs.vllm.ai/en/stable/configuration/engine_args/
[^S7]: Qwen3.8-27B model card — https://huggingface.co/Qwen/Qwen3.8-27B
[^S8]: vLLM serve configuration — https://docs.vllm.ai/en/stable/cli/serve/
[^S9]: NVIDIA CUDA minor-version compatibility — https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
[^S10]: Docker Compose GPU support — https://docs.docker.com/compose/how-tos/gpu-support/
[^S11]: vLLM online APIs and OpenAI-compatible server — https://docs.vllm.ai/en/stable/serving/online_serving/ and https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/
[^S12]: Docker restart-policy semantics — https://docs.docker.com/engine/containers/start-containers-automatically/
[^S13]: Docker Compose service/healthcheck reference — https://docs.docker.com/reference/compose-file/services/
[^S14]: vLLM API authentication limitations — https://docs.vllm.ai/en/stable/usage/security/
