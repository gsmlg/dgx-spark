# Adding Laguna S 2.1 NVFP4 and Qwen3.8-Flash-Next

**Repository:** `gsmlg/dgx-spark`  
**Date:** 2026-09-10  
**Status:** Proposed implementation and deployment documentation; not a deployment report.

## Decision

Extend the existing single-service lifecycle rather than add independent Docker scripts. Keep one active model, the `local-assistant` API alias, immutable releases, explicit qualification, and rollback. Implement Laguna first; introduce the narrowly scoped SGLang adapter for Flash-Next second.

| Proposed profile | Engine | Checkpoint selection | Initial qualification target |
|---|---|---|---|
| `laguna-s-2.1-nvfp4` | vLLM | `poolside/Laguna-S-2.1-NVFP4` | 32,768 combined tokens, one scheduled request |
| `qwen38-flash-next-nvfp4` | SGLang | `nvidia/Qwen3.8-Flash-Next-NVFP4` | 32,768 combined tokens, one scheduled request, file-backed PLE |

These are proposed conservative starting targets, not measured capacities. The checkpoint/runtime evidence and limitations are in the individual model guides. Neither new profile is qualified on Jonathan's machine by this document.

## Documents

| File | Purpose |
|---|---|
| [design.md](design.md) | Repository findings, profile contract, runtime adapters, release/state migration, safety boundaries |
| [prd.md](prd.md) | Scope, requirements and verifiable acceptance criteria |
| [implement_plan.md](implement_plan.md) | Ordered implementation tasks, file-level changes and test gates |
| [models/laguna-s-2.1-nvfp4.md](models/laguna-s-2.1-nvfp4.md) | Laguna artifact selection, vLLM settings, reasoning, DFlash and qualification |
| [models/qwen38-flash-next.md](models/qwen38-flash-next.md) | Flash-Next artifact selection, SGLang, NVMe PLE lifecycle and qualification |
| [sources.md](sources.md) | Primary sources, repository evidence and unresolved verification gates |

## Applying this documentation

Copy this `docs/model-deployment/` directory into the repository. Link it from the root README. Retain the original `DESIGN.md` and `PRD.md` as the v1 baseline; add a short scope-extension notice rather than silently replacing their requirements.

Read `design.md`, then execute the milestones in `implement_plan.md`. The proposed `--profile` option and the new profile directories do not exist in the reviewed implementation. The command sequences in the guides become usable only after the implementation milestones land and full model revisions/image digests are resolved.

No model weights, runtime image, source changes, GPU tests, or host configuration changes are included in this documentation package.
