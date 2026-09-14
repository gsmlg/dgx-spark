# Multi-model deployment design

**Date:** 2026-09-10  
**Status:** Proposed extension to Spark LLM v1.  
**Scope:** One DGX Spark, one active inference container, two new model profiles.

## 1. Preserve the useful architecture

The repository already separates model configuration, host settings, credentials, artifact preparation, immutable releases, and acceptance. Its README records a successful basic Qwen 27B API validation, but still lists long-context, performance, recovery and soak qualification as pending. Do not reinterpret that state as an accepted rollback baseline. See repository sources R1–R7 in [sources.md](sources.md).

Keep the existing Python lifecycle and thin shell wrapper. Use pure functions for profile parsing, validation, command rendering and release identity; leave Docker, filesystem and HTTP operations at explicit side-effect boundaries. This does not need a framework, an inheritance hierarchy, a database, or a resident model-management service.

The public contract remains the native Chat Completions API. The lifecycle is an operator tool, not an API proxy. Model tools are executed by the client, never by the inference server or acceptance harness.

## 2. Repository-specific blockers

These findings come from the reviewed source, not from generic deployment advice.

| Current location | Assumption | Required change |
|---|---|---|
| `lib/lifecycle.py: PROFILE, config()` | Only `profiles/qwen38-27b-nvfp4`; exact vLLM key set; 262,144 tokens and one sequence are mandatory | Resolve an explicit profile and validate common versus engine-specific settings separately |
| `prepare()` | Requires entrypoint `['vllm', 'serve']`; imports vLLM; probes `qwen3` and `qwen3_coder` | Delegate engine probes and validate the selected profile's parsers |
| `prepare()` disk check | Adds a fixed 30 GiB model allowance | Precompute actual pinned artifact bytes plus runtime/derived-file headroom |
| `compose.yaml` | Native vLLM config, cache variables and authentication | Render a frozen engine-specific invocation over the same common container skeleton |
| `state/prepared.json` | One global last-prepared pointer | Use profile-scoped prepared references; preserve legacy compatibility |
| `main()` | Calls complete mutable configuration validation before every command | Recovery/status commands must work without a valid current source profile |
| `tests/acceptance.py` | Thinking disabled by default; Qwen template options; reasoning removed from tool history | Use profile-owned behavior expectations and retain supported reasoning fields |
| Context/soak tests | Fixed 256K fixtures; even a 32K input plus output exceeds a 32K combined budget | Derive every test budget from the frozen qualification policy |
| `accept` | Checks report pass flags but not every report's embedded release/policy identity | Bind acceptance to release, artifact, policy and harness identities |

A second YAML directory alone cannot satisfy these requirements. Source references: R2–R6.

## 3. Profile contract

Retain native engine configuration rather than translate all engine flags into a new universal language. Introduce `profile.yaml` for lifecycle metadata and behavior policy. Keep `image.env` limited to the literal candidate image reference.

| File | Ownership |
|---|---|
| `profiles/<id>/profile.yaml` | Schema version, ID, engine, artifact references, capabilities, runtime environment, qualification policy, derived-cache policy |
| `profiles/<id>/image.env` | Candidate image only; resolve to an immutable digest at preparation |
| `profiles/<id>/vllm.yaml` | vLLM-only native model/serving configuration |
| `profiles/<id>/sglang.yaml` | Validated native SGLang argument mapping; render as an argument vector |
| `host.env` | Host paths, bind address, port, timeouts and memory/disk guardrails |
| `secrets.env` | Credentials only; never copy secret values into release identity |

`profile.yaml` must contain a closed, versioned schema. Required logical sections are:

- **Identity/artifacts:** profile ID, engine, primary checkpoint with full 40-character revision, tokenizer source/revision, optional named auxiliary artifacts, and source/license provenance.
- **Behavior:** text-only input, tool support, reasoning support/default, accepted reasoning response fields, history-preservation policy and documented sampling presets.
- **Runtime:** allowlisted environment, expected native entrypoint/invocation, engine-specific capabilities and optional derived-cache requirements.
- **Qualification:** configured combined context, required test cases, output reservations, scheduled concurrency, first-boot/daily-use status, and benchmark/recovery expectations.

Use one authoritative owner for each value. When the primary checkpoint reference remains in a native engine config for compatibility, the metadata loader must reference it rather than maintain a second independently editable copy. The frozen release contains the fully resolved representation. Reject mismatched duplicated values.

Reject unknown or duplicate keys, invalid types, unsafe profile IDs, path traversal, symlink escapes, unresolved substitutions, unknown engine names and arbitrary shell commands. Optional capabilities may be explicitly absent; do not require Qwen-only flags on Laguna or SGLang.

## 4. Two narrow runtime adapters

Add `lib/runtime_vllm.py` and `lib/runtime_sglang.py`, selected by a fixed engine-to-functions mapping. Shared orchestration stays in `lifecycle.py`.

Each adapter owns these operations: validate native configuration, inspect runtime version and CLI capabilities, render entrypoint/arguments, render environment and cache mounts, describe health/tokenization behavior, and extract resolved runtime information. Adapters return data; orchestration performs side effects.

For vLLM, preserve the inspected `vllm serve` entrypoint contract and pass only its arguments. For SGLang, explicitly render the inspected Python/module or supported CLI entrypoint; do not inherit vLLM's config-file invocation. Probe the selected image, not the host Python installation.

Model architecture support is not enough. Admission requires native ARM64 availability, current host driver compatibility, a real CUDA operation, parser availability, the quantization loader, required kernel paths, offline tokenizer/template loading and generation smoke tests. A successful matrix multiplication cannot certify an NVFP4 MoE kernel.

The SGLang candidate and additional requirements are documented in the Flash-Next guide. Do not use a generic `latest` image as an implicit fallback. If an adequate prebuilt image is unavailable, stop that profile at `blocked`; a reproducible custom build would be a separate, explicit scope decision.

## 5. Command and state semantics

Add `--profile <id>` to `validate`, `prepare`, `start` and `upgrade`. Keep the existing Qwen 27B profile as the default for backwards compatibility. Optionally add a read-only `profiles` listing command; it is not required for initial integration.

`prepare --profile P` records only `state/prepared/P.json`. `start --profile P` resolves that profile's prepared release, never another profile's last preparation. `start --release ID` loads the immutable release directly; reject an explicitly supplied, conflicting profile ID.

`status`, `logs`, `stop`, `restart`, `rollback`, `test` and `accept` operate on active/frozen state, not whichever mutable source profile was last edited. Parse host credentials separately so a broken or deleted candidate profile cannot prevent recovery. State-only reads should not require secrets; a stop operation should not require API authentication. An authenticated health check may legitimately require the current API secret.

Keep one global active service and one global accepted-history chain. On failed replacement, restore the last accepted release if one exists and still report failure. `rollback` means the previous accepted release, not the previous YAML edit. Expose a missing accepted target clearly; never manufacture acceptance for the current smoke-tested Qwen installation.

For legacy releases, infer engine/profile only in a compatibility reader. Do not rewrite their immutable manifests or hashes. A legacy prepared pointer may be mapped to the default profile only after checking its actual artifact identity. Back up mutable state before migration; migration must be repeatable.

## 6. Release identity and evidence

Extend the versioned identity with engine/adapter version, profile ID, resolved command/configuration, allowlisted runtime environment, image digest, all artifact revisions/hashes, template/tokenizer identity, qualification-policy hash, cache policy, and effective host configuration. Store the effective engine/kernel versions in runtime evidence.

Write all resolved files atomically and hash them, including the selected engine config, metadata, qualification policy, generated invocation and Compose file. A sampling change, new parser, additional draft model, changed context or cache precision requires a new release. Isolate compile caches by image and effective configuration.

Each report must identify its release, policy hash, test-harness revision/hash, timestamp and case coverage. Acceptance rejects copied reports, missing cases, partial runs, incompatible harness versions and evidence for another release. Report `configured_context_tokens` separately from `qualified_context_tokens`; before successful testing the latter is unknown, not the configuration value.

Profile acceptance must not weaken the original Qwen v1 requirement. That profile retains its 262,144-token qualification target. A newly configured 32K Laguna release can be accepted as a 32K release, not as a 256K release. Increasing its limit creates another candidate.

## 7. Artifacts, disks and unified memory

Replace the fixed model-size estimate with a manifest pass before download. Resolve file sizes and hashes for the selected full revisions. Include auxiliary drafts, image extraction, compilation files, partial-download headroom, configured free reserve and any derived PLE file. Account per backing filesystem and avoid double-counting shared Hub blobs or the same mount.

Keep immutable Hub snapshots read-only in the serving container. Never edit their `config.json` in place to fix an engine loader. A deliberate derived config needs its own recorded hash and provenance. Retain artifacts for accepted rollback targets; no automatic deletion of accepted weights or images.

Keep the existing 16 GiB `MemAvailable` guardrail as the initial proposal unless the actual host policy differs. A published recipe using less headroom does not authorize silently lowering it. If a new profile cannot meet the configured margin, it remains unqualified pending an explicit operator decision.

Treat GPU allocations, anonymous host memory, pinned memory, page cache, compilation and recurrent-state pools as consumers of one host budget. NVMe-backed PLE is not extra RAM; see Q1, Q2, Q5 and H1 in the source catalogue.

Add a bounded host-side monitor during startup and disruptive tests. Record single-sample dips, but stop the candidate only after 15 continuous seconds below the configured floor. During acceptance, classify swap as sustained after 15 continuous swap-active samples; retain shorter bursts in the report. It may stop only the candidate container identified by the owned Compose project and release label. Record the memory failure, retain diagnostics and use normal rollback. Avoid host-wide cache dropping, swap changes, killing unrelated processes, driver upgrades or disabling kernel/device checks.

## 8. Flash-Next derived-file lifecycle

Separate `runtime-cache/ple/<artifact-and-format-identity>/` from the read-only Hub cache and from compile caches. The directory must be local NVMe and explicitly owned by this service. It holds recomputable data, not a new model source.

The current upstream recipe warns that an already-populated PLE file can make subsequent startup substantially slower (Q2). Qualify fresh-file and existing-file startup independently. Prefer an upstream-fixed immutable image when available.

If the selected build requires resetting its derived file, use a recorded `fresh-file-before-start` policy: obtain the host lock, confirm all service processes are stopped and mappings released, validate the exact manifest-owned path, then remove only that regenerable file. Never use an unconstrained glob, follow symlinks, touch Hub weights, or reset a file mapped by an active engine. Apply the same policy to explicit restart and rollback. Include regeneration in the readiness timeout and recovery evidence.

The PLE RSS budget controls a mapping-management mechanism, not a hard whole-host page-cache limit. Observe `MemAvailable`, page faults, disk I/O and swap alongside engine pool sizes. A report showing a small mapping RSS is not proof of adequate system headroom (Q5).

## 9. API, reasoning and capacity qualification

Preserve `local-assistant` for trusted clients, but expose actual profile, model and release identity in status. Switching profiles is planned downtime and does not preserve an engine KV cache. Pause or drain clients; do not mix profile-specific reasoning histories accidentally after a switch.

Make tokenizer counting engine-aware. Prefer a verified native endpoint; otherwise run the exact pinned tokenizer/template in the pinned image. In either case require equality with inference `usage.prompt_tokens`. Template options, tool definitions and preserved reasoning must be identical for counting and generation.

Test the documented default and each reasoning mode that the pinned model/runtime actually supports. Record a declared unsupported mode as unsupported, not as passed; do not impose the existing Qwen 27B thinking toggle on every model. Preserve the engine's supported reasoning fields in the synthetic tool-return history. Test streamed tool-call argument assembly as well as streamed text. Bound tool retries and never execute model-generated shell commands.

Let `C` be combined context and `O` the requested output reserve. Boundary tests must use formatted input `C - O`, prove the sum does not exceed `C`, test `C + 1` rejection, and separate forced-length engine stress from retrieval quality. Derive soak prompts from `C`; 32,768 input tokens plus 512 output tokens are invalid for a 32,768-token release.

Unknown media requests must be rejected for these text-only profiles. Native Qwen multimodal support is not permission to download/process arbitrary remote media. Verify the selected runtime's disabling/rejection behavior; do not add a gateway merely to conceal missing enforcement.

## 10. Rollout

Deliver the profile/state/test refactor first, Laguna baseline second, SGLang/PLE integration third and Flash-Next baseline fourth. Defer DFlash, MTP, larger contexts and more concurrency until each non-speculative baseline is measured. No speed, memory-fit, 256K or million-token claim is established by documentation alone.
