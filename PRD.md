# Spark LLM — Product Requirements

**Version:** 1.0  
**Date:** 2026-09-08  
**Status:** Ready for implementation; hardware qualification pending  
**Owner / primary user:** Jonathan  
**Technical specification:** [DESIGN.md](DESIGN.md)

## 1. Product outcome

Provide a dependable local model endpoint for daily chat, coding assistance and agent tool-calling workflows on one DGX Spark. The operator starts a prepared Docker deployment rather than rebuilding an inference environment or assembling a new command each time.

The first release serves only **`Inferact/Qwen3.8-27B-NVFP4`**, under profile **`qwen38-27b-nvfp4`**, with a **262,144-token combined context ceiling**. The checkpoint declares this native context; successful operation at that length on the target machine is a release acceptance requirement, not an assumed result. [^R1]

The deployment must remain reusable: adding a different model later should change the model profile, not duplicate the service infrastructure. No alternative model is part of this release.

## 2. Users and operating assumptions

The operator is a technically experienced developer using a single private workstation/server. API clients may be local applications or trusted applications on the private network. Conversation storage, retrieval and tool execution belong to those clients.

The initial operating target is one active generation. A 256K request may occupy the engine long enough to delay ordinary chat. Version 1 favors predictable memory use and reliable recovery over concurrent long-context throughput.

Installed driver, available memory, free disk space and runtime compatibility must be discovered on the actual host. Existing Docker and model-cache conventions should be preserved rather than replaced.

## 3. Scope

**Included:** one Docker Compose service; one Qwen NVFP4 profile; a stable native API; text and streaming; tool-call round trips; optional request-level thinking; long-context qualification; persistent caches; restart behavior; explicit upgrades and rollback; status, bounded logs and test reports.

**Excluded:** Gemma or BF16 profiles, simultaneous model serving, custom inference-image development, fine-tuning, speculative decoding, distributed inference, external KV-cache systems, model hot-swapping, a management web UI, a new API gateway, a database, cloud fallback and a new MCP server.

Images, video, audio, embeddings, full provider-API parity, and agent-client-specific protocol adapters are outside the v1 API contract. The model may have capabilities beyond those qualified by this deployment.

## 4. Primary user journeys

| Journey | Expected result |
|---|---|
| First setup | Validate host and image, download one pinned snapshot, start the service, and receive a useful response |
| Daily use | Reuse the same base endpoint and `local-assistant` model name for chat, code and tools |
| Long-context work | Submit a correctly budgeted near-256K request without silent truncation or engine failure |
| Restart / reboot | Reuse prepared artifacts and return to service without depending on the model host |
| Upgrade | Prepare a new image/configuration revision, validate it, and restore the previous accepted release on failure |
| Future extension | Add a separate model profile while retaining the common deployment machinery |

## 5. Functional requirements

All requirements marked P0 are necessary for v1 acceptance. P1 is an extension point, not permission to expand the initial model scope.

| ID | Priority | Requirement |
|---|---|---|
| FR-01 | P0 | Ship exactly one model profile: `qwen38-27b-nvfp4` |
| FR-02 | P0 | Use one long-running native ARM64 Docker Compose service with NVIDIA GPU access |
| FR-03 | P0 | Record and deploy an immutable image digest and full checkpoint/tokenizer revision; never update them implicitly |
| FR-04 | P0 | Provide idempotent preflight, prepare, start, stop, restart, status, logs, test, upgrade and rollback operations |
| FR-05 | P0 | Keep downloaded weights and runtime compilation artifacts across container recreation |
| FR-06 | P0 | Start from complete local artifacts without contacting Hugging Face for required files |
| FR-07 | P0 | Support model discovery and chat completions using the stable `local-assistant` alias |
| FR-08 | P0 | Support streamed responses, correct stream termination and client cancellation |
| FR-09 | P0 | Support a function call, valid serialized arguments, correlated client tool result and final assistant answer |
| FR-10 | P0 | Default to non-thinking; allow explicit request-level thinking with documented sampling settings |
| FR-11 | P0 | Serve a combined context of 262,144 tokens and accurately document input/output budgeting |
| FR-12 | P0 | Reject oversized inputs explicitly; never silently trim history, tools or the prompt to appear successful |
| FR-13 | P0 | Separate HTTP readiness, successful generation and full-length qualification in status |
| FR-14 | P0 | Serialize deployment mutations and prevent overlapping model containers controlled by this project |
| FR-15 | P0 | Restore the previous accepted release after a failed candidate replacement; preserve failure diagnostics |
| FR-16 | P0 | Resume an intentionally running service after a host restart, but preserve an intentional operator stop |
| FR-17 | P0 | Keep credentials and prompt/output bodies out of committed files and ordinary diagnostics |
| FR-18 | P1 | Preserve a profile boundary for future models; do not implement a second model during v1 |

The API contract uses vLLM’s native interface, not a new compatibility layer. Parser names and baseline serving behavior must be checked against the selected runtime. The published Qwen recipe is a starting reference, not a guarantee for every image/hardware combination. [^R2]

## 6. Configuration requirements

The implementation must provide documented configuration for host storage paths, bind address, port, credentials, operational timeouts, image selection and the model-specific serving parameters listed in DESIGN.md.

Initial serving targets are 262,144 context tokens, one scheduled sequence, tensor parallelism one, memory utilization 0.70, 4,096 batched tokens, automatic KV dtype, chunked prefill and prefix caching. These are qualification inputs; changes must be visible in a new configuration record.

The initial image candidate is `vllm/vllm-openai:v0.28.0-cu130`. Resolve a real digest during preparation and validate native ARM64 execution, CUDA operations and the NVFP4 model path. Do not replace the digest with a fabricated placeholder or label an untested tag “Spark-validated.” [^R3]

Each request uses one output sequence. Document that a scheduler limit is not an HTTP queue-size limit. Completion budgets must be explicit in the test/client guidance; the 4,096-token client default must not prevent a valid 16,384-token completion budget. [^R4]

Unknown configuration keys, missing pins, unresolved variables and incompatible image arguments must fail with an actionable error. Do not mutate the host driver, silently install packages inside a running container, or fall back to another checkpoint.

## 7. Acceptance tests

### A. Deployment and API

| Test | Passing condition |
|---|---|
| Host preflight | Records environment and verifies the actual candidate image can execute CUDA work |
| Startup | Exactly one project-managed model service; health, alias lookup and short generation pass |
| Model identity | Status exposes the actual repository, full revision, image digest and configuration hash |
| Ordinary text | English and Chinese prompts produce nonempty, relevant answers without parser artifacts |
| Streaming | Received events assemble into a complete answer and terminate normally |
| Tool use | At least three different tool fixtures complete call → client result → final response with valid argument JSON and matching identifiers |
| Thinking | Both modes work; actual reasoning/final-answer fields are documented and the non-thinking default is verified |
| Invalid requests | Oversized input, malformed payload and unknown model alias return explicit errors without crashing the engine |
| Overlapping requests | Two submitted requests complete or cancel cleanly under the configured scheduler; no promise of parallel 256K execution |
| Cancellation | Cancelling a streamed request releases its work; a subsequent short request succeeds |
| Text-only contract | Unsupported media requests fail clearly instead of being silently ignored |

Tool tests validate transport and selected fixtures, not universal correctness of every tool decision. Use synthetic inputs and harmless mock tool results, not live side-effecting tools.

### B. Long-context qualification

Generate deterministic, nontrivial text fixtures and count the **complete formatted prompt** using the pinned tokenizer and matching chat template, including tools and generation markers. Verify the server’s reported input usage agrees. Do not approximate tokens from bytes or repeat a single padding token as the only capacity test.

| Case | Required test |
|---|---|
| 32K input | Generation succeeds; establish the first long-input timing result |
| 128K input | Generation succeeds without engine or host memory failure |
| Large completion reservation | 245,760 formatted input tokens plus a requested 16,384-token completion budget is accepted and produces a response |
| Near-limit input | 261,120 formatted input tokens plus a requested 1,024-token completion budget is accepted and produces meaningful output |
| Oversized prompt | An input exceeding 262,144 tokens produces an explicit client error without truncation |
| Invalid combined budget | An explicit input-plus-completion budget over the configured limit produces a documented validation error, not silent prompt removal |
| Post-load recovery | A short request succeeds after each long request and after a cancelled long request |

An early end-of-sequence in the reservation test does not prove that the engine generated all 16,384 tokens. Record requested and actual output lengths separately. Also require at least one clearly labeled engine stress run that reaches the full 262,144-token input-plus-generated-output boundary, using test-only generation controls when necessary. Never enable those controls in daily defaults or use forced-length output as an answer-quality benchmark.

For the near-limit fixture, place known independent key/value facts near the beginning, middle and end. Require correct retrieval of all three in the acceptance fixture. Record quality failures separately from allocation or transport failures; one retrieval fixture does not establish general 256K reasoning quality.

Run at least three near-limit trials with distinct fixture seeds. At least one must have no reusable inference prefix cache. Record each trial’s latency and resource use individually. A configured maximum, tokenizer-only test or successful startup does not satisfy this section.

### C. Reliability and recovery

| Test | Passing condition |
|---|---|
| Offline artifact restart | Recreated service starts using cached files while model-host access is unavailable |
| Warm recreation | Weights remain present and runtime compilation-cache use is recorded separately |
| Process failure | After qualification, a controlled process exit is recovered according to Docker’s configured restart policy |
| Host reboot | A previously running service returns; an intentionally stopped service remains stopped |
| Failed upgrade | A deliberately invalid candidate cannot replace the accepted release permanently; rollback restores working inference |
| Partial preparation | Interrupted downloads do not corrupt the accepted release or change active configuration |
| Unhealthy process | Status reports unhealthy state accurately; a healthcheck alone is not represented as automatic repair |
| Mutation race | Two lifecycle mutations cannot create overlapping replacements or corrupt release state |
| Memory guardrail | No host OOM or sustained swapping; the documented available-memory safety margin is met |
| Mixed-use soak | An eight-hour test window with at least 100 short requests, tool rounds, and three long-input requests finishes without an unexplained engine restart |

Reboot, fault injection and long stress tests must require an explicit test mode; routine smoke tests must not reboot the host, terminate unrelated workloads or interrupt an accepted service. Docker restart behavior must match its actual policy semantics. [^R5]

## 8. Performance and daily-use acceptance

No unmeasured throughput or first-token target is promised. Qualification must produce measurements on this Spark for warm 4K-input requests with a 512-token output budget, plus 32K, 128K and near-256K requests.

For at least 20 warm short requests, record actual input/output lengths, first-token latency, inter-token/decode performance and end-to-end latency; summarize the distribution. For the small long-context sample, publish individual results rather than presenting statistically strong percentile claims.

The operator must explicitly accept the measured short-request responsiveness for daily use. Until then, report “capacity validated” separately from “daily-use accepted.” Save the accepted baseline with its exact image, model revision, settings and host state.

For subsequent upgrades, a greater than 20% regression in median short-request first-token latency or decode performance under equivalent conditions blocks automatic acceptance pending review. This is a proposed regression policy, not a claim about the model’s absolute speed.

## 9. Deliverables and implementation sequence

| Stage | Deliverable | Exit condition |
|---|---|---|
| 1. Configuration | Common Compose file, one profile, host environment template and validation | No duplicate model settings or unresolved runtime arguments |
| 2. Host/artifacts | Preflight, preparation, image/model pinning and persistent storage | Candidate is locally available and its CUDA path passes |
| 3. Basic serving | Native endpoint, streaming, tool calling and thinking behavior | API acceptance suite passes |
| 4. Context | Token-accurate fixtures and memory/performance reports | Long-context acceptance suite passes at 262,144 total tokens |
| 5. Operations | Restart, status, bounded logs, locking, upgrade and rollback | Recovery tests and mixed-use soak pass |
| 6. Handoff | README with lifecycle usage, client settings and troubleshooting; accepted release report | Daily-use baseline is reviewed and acceptance recorded |

The implementation must include test utilities and a concise operations README. Do not expand this stage into building a dashboard, generic multi-engine framework or additional model profiles.

## 10. Definition of done

Version 1 is complete only when the single pinned NVFP4 deployment passes API, long-context and recovery requirements on the target Spark, its actual daily-use performance has been accepted, and another invocation of the documented lifecycle operation reproduces the same working release from prepared artifacts.

All known limitations must be recorded. A smaller context, silently changed quantization, unpinned nightly image, successful `/health` response, or a benchmark from another GPU is not a substitute for the required evidence.

## References

Requirements and test thresholds above are proposed product decisions. Published capabilities and runtime semantics are supported by these references, checked on 2026-09-08:

[^R1]: Checkpoint context configuration — https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4/blob/main/config.json
[^R2]: Qwen vLLM serving recipe — https://recipes.vllm.ai/Qwen/Qwen3.8-27B
[^R3]: Published vLLM image candidates — https://github.com/vllm-project/vllm/releases/tag/v0.28.0
[^R4]: vLLM engine and serving semantics — https://docs.vllm.ai/en/stable/configuration/engine_args/ and https://docs.vllm.ai/en/stable/cli/serve/
[^R5]: Docker restart behavior — https://docs.docker.com/engine/containers/start-containers-automatically/
