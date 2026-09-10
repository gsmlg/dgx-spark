# Product requirements: two additional Spark model deployments

**Date:** 2026-09-10  
**Status:** Proposed; companion [design](design.md) and [implementation plan](implement_plan.md).

## Objective

Allow Jonathan to prepare, switch, test and operate Laguna S 2.1 NVFP4 and Qwen3.8-Flash-Next through the existing `spark-llm` lifecycle, without discarding the current Qwen 27B setup or its safety properties.

## Scope

Ship profile-aware configuration, a vLLM Laguna profile, a narrowly scoped SGLang Flash-Next profile, profile-aware tests, immutable evidence and operational documentation. Run one model service on one Spark at a time. Preserve the native private Chat Completions interface and `local-assistant` alias.

Exclude simultaneous model residency, two-Spark deployment, Kubernetes, a gateway, a UI, embeddings, Responses API, multimodal serving, fine-tuning, host driver replacement and runtime package installation on the host. Speculation and larger context are follow-on qualification tracks, not first-boot prerequisites.

The initial proposed target for each new profile is 32,768 combined input/output tokens and one scheduled request. A later 64K/128K/256K release requires independent evidence. Existing Qwen 27B v1 acceptance remains unchanged at 262,144 tokens.

## Functional requirements

| ID | Requirement | Acceptance evidence |
|---|---|---|
| F01 | Profile selection is explicit and safe | Default-Qwen compatibility plus unknown-ID/path-traversal tests |
| F02 | Preparation does not replace the running service | Active state/container unchanged during another profile's preparation |
| F03 | Full immutable artifact identity | Image digest, full checkpoint/tokenizer revisions, manifest hashes and auxiliary artifacts recorded |
| F04 | Offline serving after preparation | Restart/generation with Hub access unavailable; no artifact downloads during start |
| F05 | Engine-specific admission | ARM64, CUDA, parsers, architecture/quantization/kernel support and actual generation checked |
| F06 | One service, safe replacement | Owned previous service stops; unrelated GPU workloads cause refusal, not termination |
| F07 | Cross-engine rollback | Failed vLLM-to-SGLang and SGLang-to-vLLM replacement restores an accepted target where available |
| F08 | Recovery independent of source edits | Malformed/deleted candidate YAML cannot break status, stop or frozen-release rollback |
| F09 | Correct native API | Model alias, text, streaming, usage, tools, errors, cancellation and bounded overlap pass |
| F10 | Reasoning-aware behavior | Default/on/off tests; preserved reasoning survives at least three synthetic tool round trips |
| F11 | Honest context qualification | Token counts match inference, combined-budget boundaries pass, qualified capacity is reported separately |
| F12 | Safe Flash-Next PLE handling | Local NVMe file backend; startup/restart policy tested; owned derived files cannot affect immutable weights |
| F13 | Release-bound acceptance | Wrong-release/policy/harness, missing-case and failed reports cannot promote a release |
| F14 | Evidence-preserving failures | Failed attempts remain recorded; no silent context, precision, image or model fallback |

## Operational requirements

No secrets or prompt/response bodies in reports or Git. Use explicit private binding; LAN exposure requires native authentication and a trusted network boundary. Preserve the existing nonblocking host lock and bounded logs. Candidate containers remain `restart: no`; enable `unless-stopped` only after acceptance.

Reject insufficient disk capacity before a large download. Count both checkpoint artifacts and runtime-generated data on their actual filesystems. Do not rely on a checkpoint's marketing size or on sparse-file apparent allocation alone.

Measure the configured host memory floor continuously during startup and qualification. No host OOM, sustained swapping, unexplained container restart or unbounded initialization is acceptable. Any reduction in the existing safety margin requires an explicit policy revision and new release.

The operator must set and review finite first-start, restart, cancellation and responsiveness budgets before daily-use qualification. A successful first generation does not satisfy recovery usability. Keep cold compilation and cold/warm PLE-file results separate from warm inference benchmarks.

## Acceptance levels

| Level | Meaning |
|---|---|
| `blocked` | Missing artifact, compatible image/feature, storage, driver or safety prerequisite |
| `prepared` | Pinned artifacts and runtime admission checks completed; generation not established |
| `smoke-passed` | Healthy service with generation and tool smoke; no daily-use/capacity promise |
| `qualified` | API, configured context, benchmark and eight-hour soak evidence passed |
| `accepted` | Qualified plus operator-reviewed responsiveness, recovery and regression evidence |

Status describes a release, not a model family. It must never inherit a larger context claim or an acceptance result from another profile/revision.

## Release gates

**G0 — Regression safety.** Existing host configuration, active-state reading, immutable release verification, default CLI behavior and Qwen 27B validation remain compatible.

**G1 — Laguna baseline.** A pinned, non-speculative release passes native NVFP4 generation, both reasoning modes, tools and API tests without using Qwen-only parser assumptions. Artifact size discrepancies are resolved against the selected manifest.

**G2 — SGLang/PLE admission.** The selected ARM64 image contains the required model, mixed-precision loader and file-backed PLE support. The device check passes without bypasses. Fresh and subsequent startup have bounded behavior; rollback still works.

**G3 — Flash-Next baseline.** A pinned text-only, non-MTP release passes the shared API and profile-specific suites with the PLE table file-backed, within the host's actual memory and disk policy.

**G4 — Daily use, independently per profile.** Pass configured-context tests, a warm 20-request benchmark, eight-hour soak and manually controlled recovery tests; then accept that exact release. Record TTFT, decode rate, cancellation recovery, memory floor, swap/OOM, disk/page-fault behavior and cold/warm startup times. Compare identical workloads; a cross-model benchmark is a selection tradeoff, not necessarily a runtime regression.

## Definition of done

Both profiles and adapters have automated non-GPU tests, clear blocked-state diagnostics, deployment guides and complete pinned configuration. A profile may ship as experimental when hardware evidence is missing, but must not be labeled accepted. Report hardware gates as pending until they have actually run on the target Spark.

References and upstream limitations: [sources.md](sources.md). Runtime choices and operational defaults in this PRD are design decisions; they are not publisher performance claims.
