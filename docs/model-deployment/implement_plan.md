# Implementation plan

**Date:** 2026-09-10  
**Target repository:** `gsmlg/dgx-spark`  
**Execution order:** Safety/refactor → Laguna → SGLang/PLE → Flash-Next → independent acceptance.

## Milestone 0 — Freeze the starting point

Read the current branch again before implementation; this review may not match a later checkout. Record the actual Git commit and relevant source hashes. Back up local mutable `state/` metadata without copying credentials into Git. Inventory active/prepared/accepted release IDs and retained artifacts.

Run the existing host-independent tests and capture failures before changing code. The README says Qwen 27B has basic API evidence but outstanding full qualification; inspect actual host state rather than assuming an accepted fallback exists. Do not run destructive recovery tests or change the host driver.

**Gate:** Baseline and migration constraints documented; no change to the running service.

## Milestone 1 — Separate profile data from lifecycle execution

Modify `lib/lifecycle.py` and add `lib/profiles.py`. Replace the global profile path with a safe resolver. Split host/secrets parsing, source-profile validation and immutable-release loading. Add the metadata schema described in `design.md`; preserve the existing Qwen YAML behavior through compatibility defaults.

Implement `--profile` for validate/prepare/start/upgrade, profile-scoped prepared pointers and explicit conflict checking with `--release`. Recovery/read operations use frozen state. Make state migration idempotent and maintain old release identity verification without rewriting old manifests.

Add `lib/runtime_vllm.py` containing the existing runtime behavior expressed as functions. Move CLI/parser/entrypoint probing there, initially preserving the default Qwen behavior. Introduce a common Compose renderer; keep networking, GPU reservation, logging, mounts and restart policy unchanged.

**Tests:** Unknown profiles; traversal and symlink escape; duplicate/missing/unknown values; engine-specific option rejection; wrong-profile release selection; concurrent preparation; bad source YAML during stop/rollback; legacy release/prepared-state loading; secret redaction; changed policy/environment changing identity.

**Gate:** Existing Qwen remains functionally unchanged; new metadata does not relax its 256K requirement.

## Milestone 2 — Fix artifact planning and qualification

Extend `lib/prepare_model.py` with a metadata-only planning stage. Estimate actual new blob bytes and runtime/derived-file requirements per filesystem before pulling/downloading large data. Keep the existing integrity checks; add named auxiliary artifact manifests without enabling drafts by default.

Freeze profile behavior, test policy, native invocation and runtime environment into each new release. Update status with engine/profile, configured versus qualified context, accepted state and evidence identity.

Refactor `tests/acceptance.py`: dynamic boundary arithmetic, engine-aware token counting, profile defaults, preserved reasoning, streamed tools, and bounded request output. Derive soak fixture size from available combined context. Do not silently skip required tests on an engine lacking a needed endpoint.

Strengthen `accept`: verify each report's embedded release ID, policy hash, harness identity, required cases and pass status. Continue requiring operator responsiveness/recovery/regression review. A failed or partial last run must not be overwritten into apparent success.

**Tests:** 32K/64K/256K budget boundaries; exact tokenizer/inference agreement; 32K soak not exceeding context; wrong-release reports; missing cases; different reasoning fields; streaming argument fragments; unsupported forced-generation/tokenization controls; preserved legacy evidence behavior.

**Gate:** Host-independent fixtures prove new profiles cannot inherit old capacity or acceptance.

## Milestone 3 — Add Laguna without speculation

Create `profiles/laguna-s-2.1-nvfp4/{profile.yaml,image.env,vllm.yaml}` using the [Laguna guide](models/laguna-s-2.1-nvfp4.md).

Resolve and commit a real full checkpoint revision. Inspect its full manifest, quantization metadata, context configuration, tokenizer/template and generation defaults. Do not put placeholder hashes in a launchable profile. The existing vLLM image is only a candidate: verify parsers, kernels, offline custom-code requirements and bounded JIT inside it.

Implement profile-owned runtime environment so JIT controls are present during probes and serving and included in release identity. Download only the target checkpoint. No DFlash artifact or speculative options in this baseline.

On the Spark, prepare without replacing the current service. During an explicitly chosen test window, start the candidate, run smoke/API, record memory and startup behavior, then run capacity/benchmark/soak only with clients paused. Restore the accepted baseline after a failed attempt where available.

**Gate:** Either exact-release evidence passes or a specific blocker is recorded. No source-level "supported" flag substitutes for hardware evidence.

## Milestone 4 — Add the SGLang adapter and PLE ownership

Add `lib/runtime_sglang.py` and adapter contract tests. Render a native argument vector, not shell text and not vLLM YAML. Map API authentication, health/tokenizer behavior, offline cache settings, shutdown, logs and engine metadata explicitly.

Verify the candidate image named by the current upstream Spark recipe, including ARM64 manifest, digest, source provenance and required fixes. Do not assume a merged PR appears in a tagged image. Require file-backed PLE/device capability and the NVIDIA mixed-precision loader before selecting that checkpoint.

Add the release-owned local-NVMe mount and derived-file manifest. Implement the documented reusable/fresh-file policy with process/mapping checks, path containment, locking and narrowly scoped cleanup. Preserve immutable weights. The runtime file must be recreated correctly during restart and cross-engine rollback, not merely during first installation.

Keep host safety monitoring independent of the inference process. Only the owned candidate may be stopped on a configured low-memory threshold. Measure mapping RSS and page-cache behavior separately; do not treat an 8 GiB RSS setting as a RAM quota.

**Tests:** Wrong architecture; image missing flags; no NVMe space; read-only/unsafe/symlink PLE path; wrong derived-file identity; cleanup while running; interrupted initialization; unchanged Hub weights; warm restart; current-profile corruption; vLLM↔SGLang failure rollback.

**Gate:** Adapter and PLE behavior are reproducible and independently tested. Missing runtime features block the profile.

## Milestone 5 — Add Flash-Next without MTP

Create `profiles/qwen38-flash-next-nvfp4/{profile.yaml,image.env,sglang.yaml}` using the [Flash-Next guide](models/qwen38-flash-next.md).

Pin the NVIDIA export and compatible image only after successful inspection. Keep RadixArk as an explicitly selected alternative revision/profile, never an automatic fallback. Disable speculation, keep scheduled concurrency at one and start at 32K combined context. Do not copy datacenter tensor-parallel or high-concurrency settings.

Run first-start and subsequent-start tests separately. Confirm the logs show the file PLE backend and correct mixed-precision/kernel choices. Resolve the upstream reasoning-mode documentation conflict against the pinned template/runtime; test every supported mode and preserve multi-turn tool history. Reject media in this text-only deployment.

**Gate:** No OOM, sustained swap or memory-floor violation; full required API cases pass. Cold-start and restart timings remain visible even when inference is successful.

## Milestone 6 — Daily-use acceptance and documentation

For each profile independently, run exact-context retrieval/boundary tests, 20 warm benchmark requests and the eight-hour soak. Complete the operator-controlled recovery suite, including fresh/warm startup, failed replacement, manual stop, container restart, host reboot and offline artifact availability. Do not trigger a host reboot automatically.

Only then promote the exact release with reviewed evidence. Update README with measured context, concurrency, runtime/artifact pins and remaining limitations. Preserve previous reports and accepted artifacts. Keep new experimental capabilities out of the default client recommendation.

**Gate:** Both profiles are either independently accepted with evidence or explicitly experimental/blocked with concrete remaining requirements. No invented measurements.

## Follow-on work, after baseline acceptance

Evaluate Laguna DFlash as a separate release with a separately pinned draft artifact. Evaluate Flash-Next MTP as another release with verified router/kernel fixes. Increase context in explicit 64K/128K/256K steps, and concurrency only after single-request stability. Do not expand context and enable speculation in the same first experiment.

## Agent handoff

Implement these milestones in order. Prefer small, reviewable commits and pure data transformations. Do not redesign the repository into a service platform. Report changed files, automated tests, target-host tests actually executed, blockers and pending acceptance separately. Never claim deployment success from mocked tests or a model card. Do not push, upgrade drivers, delete accepted artifacts or accept a candidate without the operator's intended authorization.
