# Qwen3.8-Flash-Next on one DGX Spark

**Date:** 2026-09-10  
**Proposed profile:** `qwen38-flash-next-nvfp4`  
**State:** Experimental integration; host qualification pending.

## 1. This is not the existing Qwen 27B profile

The official open-weight model is `Qwen/Qwen3.8-Flash-Next`. Its model card describes 125B language-model parameters with 6B activated, an additional 51B n-gram embedding table and a 4B MTP component. It uses Gated DeltaNet and Qwen Sparse Attention and advertises 262,144 native context. Activated parameter count does not determine weight residency. [Q1](../sources.md)

For this single-Spark integration, select the published `nvidia/Qwen3.8-Flash-Next-NVFP4` export, not the full-precision upstream checkpoint. The NVIDIA export uses ModelOpt mixed precision; the alternative `RadixArk/Qwen3.8-Flash-Next-NVFP4` is a different artifact/loader path. Never switch between them automatically. [Q3, Q4, Q6](../sources.md)

## 2. Why SGLang and file-backed PLE

Recommend SGLang because its current cookbook explicitly documents a single-Spark route using an NVMe-backed PLE n-gram table. The documented NVFP4 shape is around 126 GiB, including a 47.7 GiB FP8 table; ordinary pinned-host PLE offload does not make that fit unified memory. [Q2](../sources.md)

The proposed runtime keeps the main weights resident and gathers table rows from a memory-mapped local NVMe file. File-backed pages still consume page cache and incur I/O; this is not cost-free memory expansion. The underlying implementation uses host page-table access and checks the device capability. Do not disable that check. [Q5](../sources.md)

This is a chosen deployment route, not a claim that vLLM can never serve this architecture. A vLLM route would need separate evidence for this exact quantization, file-backed table and single-GB10 memory behavior.

## 3. Runtime candidate and feature gates

The official cookbook names `lmsysorg/sglang:dev-qwen38-next-local` for the Spark-oriented build; it distinguishes that build from the older day-zero image. Treat the tag as a candidate only. Resolve its ARM64 manifest and immutable digest, and record its actual engine/source/kernel versions. [Q2](../sources.md)

Required features include native model support, file-backed PLE, the selected ModelOpt loader and GB10-appropriate expert kernels. For later MTP, require the corresponding router/kernel fixes too. A PR merged to the model-support branch is not proof of inclusion in any arbitrary release image. [Q2, Q5, Q6](../sources.md)

Verify the selected engine exposes the required native arguments. The core file-backend settings are `--ple-offload-embedding`, `--ple-offload-backend file`, and `--ple-offload-dir` pointing at the owned NVMe mount. [Q5](../sources.md)

For the NVIDIA export, leave quantization detection to checkpoint metadata and explicitly validate `flashinfer_cutlass` as the MoE runner rather than accepting an inappropriate automatic choice. Do not transplant RTX PRO 6000 pinned-memory results onto a Spark. [Q6](../sources.md)

If no compatible prebuilt ARM64 image passes admission, mark the profile blocked. Do not install packages into a running container or overlay a mutable Git branch at startup. A custom image requires a separately reviewed reproducible Dockerfile and exact dependency/source pins.

## 4. Proposed first-release policy

| Concern | Initial decision |
|---|---|
| Primary artifact | NVIDIA NVFP4 export with a real full commit |
| Engine | Pinned SGLang candidate with required features |
| Alias | `local-assistant` |
| Tensor parallelism | 1; one Spark only |
| Combined context | 32,768 tokens |
| Running requests | 1; inspect effective scheduler limit |
| Shared KV pool | `max-total-tokens: 32768`; bound allocation to the single configured-context request |
| Static allocation starting point | `mem-fraction-static: 0.85`, subject to host guardrails |
| PLE | NVMe file backend; separate writable derived-data mount |
| Expert backend | Explicit compatible CUTLASS path; validate resolved logs |
| Reasoning parser | Start from the cookbook's `auto`; record and validate the resolved parser |
| Tool parser | Enable cookbook `auto`; record the resolved detector and validate tool round trips |
| Prefill | Conservative 4,096-token chunks if supported |
| MTP/speculation | Disabled |
| Modality | Text-only serving and media rejection |

Host validation on 2026-09-14 found that leaving `max-total-tokens` unset allocated a
750,016-token BF16 KV pool (17.16 GiB) despite the single 32,768-token request policy.
The server reached readiness, but CUDA-graph capture reported 14.92 GiB available and
the lifecycle stopped it after 15 continuous seconds below the 16 GiB host floor. The
profile now caps the shared KV pool at 32,768 tokens; this revised release still requires
startup and API verification before it is considered runnable.

The first bounded-KV start reached health with 32.88 GiB available during CUDA-graph
capture, but PLE-related swap activity did not settle within the original 60-second
post-readiness window. The lifecycle now permits a bounded ten-minute settling window
while still requiring ten consecutive quiet seconds before smoke generation.

The remaining numbers in this table are proposed local qualification settings, not measured capacity. Do not copy the cookbook's datacenter TP, request count, cache-slot sizes or model-card million-token extension into the first release.

Recurrent-state precision, page size, cache strategy and slot counts must come from a compatible verified engine configuration. Record the effective pools and admitted concurrency. Native `max-running-requests` is not a guarantee of bounded HTTP admission or simultaneous long-context requests.

## 5. Storage and restart behavior

The disk plan must include the full pinned checkpoint, image extraction, compile cache, the fully materialized derived table, free reserve and retained rollback artifacts. A sparse file's initially small allocated size is not its eventual budget. Require local NVMe; do not assume a network filesystem has the required latency or mapping behavior.

The reviewed cookbook reports a substantial existing-file rewrite penalty on its current build, recommending a fresh derived table before boot. Therefore fresh-file and existing-file startup are separate mandatory tests. Do not promise fast warm restart merely because the model download is cached. [Q2](../sources.md)

Use the ownership and `fresh-file-before-start` mechanism in [design.md](../design.md), not a handwritten `rm` glob. Only after the engine is stopped and mappings are released may the lifecycle reset its recorded derived file. Never delete checkpoint shards or a currently mapped file. Include regeneration in the configured readiness timeout and rollback duration.

Monitor mapping RSS, `MemAvailable`, file-backed page cache, faults and disk throughput. `SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB` is an upstream mapping-management control, not a hard page-cache or host-memory cap. The source describes dropping mappings while pages may remain cached. Do not use its numerical setting as proof that the host has spare RAM. [Q5](../sources.md)

If the upstream image later fixes existing-file startup, pin that new image and requalify; do not silently change cache policy on an accepted release.

## 6. Reasoning, tools and text-only mode

The official model card describes thinking controls including `enable_thinking`, `preserve_thinking` and `reasoning_effort`, while the reviewed SGLang cookbook says thinking cannot be disabled. These sources conflict. Inspect the pinned checkpoint template and test the selected engine before advertising a toggle. Preserve supported reasoning history on multi-turn requests; qualify native field names instead of hard-coding one. [Q1, Q2](../sources.md)

For initial daily chat, choose and document an explicit non-thinking default only if the selected SGLang build supports enforcing it correctly. Otherwise retain the documented native default and require clients/tests to set their intended mode. Never label the server non-thinking based only on a request example.

Freeze the sampling policy for each supported mode from the selected checkpoint/template and verify engine support. The cookbook also conflicts with the model card about whether sampling recommendations exist; record the chosen pinned source and do not silently inherit the 27B profile. Count the same thinking/template options in tokenization and generation. Tool tests must include successive assistant/tool messages, valid JSON arguments and streamed argument reconstruction.

The upstream model is multimodal, but this repo extension is text-only. Verify that the engine can avoid unnecessary vision initialization and reject image/video requests. Use supported native controls only. Failure to enforce the promised input contract is a release blocker, not an undocumented exception.

## 7. Deployment sequence after implementation

The profile commands below are implemented. Preparation still has to prove the selected
SGLang image, file-backed PLE path, and pinned checkpoint on the target Spark.

```sh
bin/spark-llm validate --profile qwen38-flash-next-nvfp4
bin/spark-llm doctor
bin/spark-llm prepare --profile qwen38-flash-next-nvfp4
# Inspect the frozen artifact/runtime plan; pause clients before replacement.
bin/spark-llm start --profile qwen38-flash-next-nvfp4
bin/spark-llm status
bin/spark-llm test --mode api
```

Before `start`, confirm the NVIDIA loader, file backend, exact NVMe directory, host memory floor, readiness timeout and real accepted fallback. Check startup logs for effective quantization, expert kernels, recurrent pools and file-backed table selection.

After smoke/API success, pause clients for configured-context, benchmark and soak modes. Complete fresh/existing-file restart, manual stop, offline start and failed-replacement recovery. No automatic host reboot or cache dropping is permitted by the test harness.

Promote with reviewed evidence only after all required cases pass. Retain failure reports if the memory floor or restart budget cannot be met; the profile remains experimental.

## 8. Later experiments

Increase context through explicit 64K, 128K and 256K releases. QSA and recurrent state make naive dense-KV estimates incomplete; verify actual formatted token counts, allocated pools and retrieval behavior. Do not infer qualified context from `/get_server_info` or a command-line setting alone.

MTP is a separate quality/performance experiment after the no-MTP baseline. Confirm the selected NVIDIA export's draft-loader path, pin any separately referenced draft and require the relevant router fixes. Test for repetitive/output-collapse failures, compare tool correctness and keep a no-MTP accepted rollback release. Do not borrow two-node draft workarounds for this TP=1 configuration without evidence.

## 9. Expected report

Record checkpoint revision, image digest/source build, context and concurrency, host driver/kernel, NVMe filesystem, fresh/existing-file startup times, minimum `MemAvailable`, swap/OOM, page faults, disk I/O, tokenizer agreement, reasoning fields, tool results, TTFT and decode rate. Separate published evidence from measurements on this specific Spark.
