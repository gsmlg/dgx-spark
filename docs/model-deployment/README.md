# Additional model profiles

**Repository:** `gsmlg/dgx-spark`  
**Date:** 2026-09-10  
**Status:** Profile switching and the candidate profiles are implemented; target-host preparation and qualification remain release-specific.

## Decision

Extend the existing single-service lifecycle rather than add independent Docker scripts. Keep one active model, the `local-assistant` API alias, immutable releases, explicit qualification, and rollback. Implement Laguna first; introduce the narrowly scoped SGLang adapter for Flash-Next second.

| Proposed profile | Engine | Checkpoint selection | Initial qualification target |
|---|---|---|---|
| `laguna-s-2.1-nvfp4` | vLLM | `poolside/Laguna-S-2.1-NVFP4` | 5,120 combined tokens, one scheduled request |
| `laguna-xs-2.1-nvfp4` | vLLM | `poolside/Laguna-XS-2.1-NVFP4` | 262,144 combined tokens, one scheduled request |
| `gpt-oss-120b-mxfp4` | vLLM | `openai/gpt-oss-120b` | 131,072 combined tokens, one scheduled request |
| `qwen38-flash-next-nvfp4` | SGLang | `nvidia/Qwen3.8-Flash-Next-NVFP4` | 32,768 combined tokens, one scheduled request, file-backed PLE |

These are configured starting targets, not measured capacities. The checkpoint/runtime evidence and limitations are in the individual model guides. Every profile requires qualification of its exact release.

## Documents

| File | Purpose |
|---|---|
| [design.md](design.md) | Repository findings, profile contract, runtime adapters, release/state migration, safety boundaries |
| [prd.md](prd.md) | Scope, requirements and verifiable acceptance criteria |
| [implement_plan.md](implement_plan.md) | Ordered implementation tasks, file-level changes and test gates |
| [models/laguna-s-2.1-nvfp4.md](models/laguna-s-2.1-nvfp4.md) | Laguna artifact selection, vLLM settings, reasoning, DFlash and qualification |
| [models/laguna-xs-2.1-nvfp4.md](models/laguna-xs-2.1-nvfp4.md) | Laguna XS artifact pin, serving defaults and qualification boundary |
| [models/gpt-oss-120b-mxfp4.md](models/gpt-oss-120b-mxfp4.md) | GPT-OSS artifact pin, parsers, serving policy and qualification boundary |
| [models/qwen38-flash-next.md](models/qwen38-flash-next.md) | Flash-Next artifact selection, SGLang, NVMe PLE lifecycle and qualification |
| [sources.md](sources.md) | Primary sources, repository evidence and unresolved verification gates |

## Applying this documentation

The repository now includes `--profile`, engine adapters, profile-scoped prepared state,
dynamic qualification budgets, and the candidate profile directories. The model commits are
pinned; `prepare` resolves each candidate image to an immutable ARM64 digest and checks
its runtime features before downloading the checkpoint.

Read `design.md`, then use the model guides. No new model weights or runtime images have
been pulled and no target-host GPU qualification is claimed by the source implementation.
