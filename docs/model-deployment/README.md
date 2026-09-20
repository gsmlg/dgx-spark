# Additional model profiles

**Repository:** `gsmlg/dgx-spark`  
**Date:** 2026-09-20

**Status:** Profile switching and the candidate profiles are implemented; target-host preparation and qualification remain release-specific.

## Decision

Extend the existing single-service lifecycle rather than add independent Docker scripts. Keep one active model, the `local-assistant` API alias, immutable releases, explicit qualification, and rollback. Implement Laguna first; introduce the narrowly scoped SGLang adapter for Flash-Next second.

| Proposed profile | Engine | Checkpoint selection | Initial qualification target |
|---|---|---|---|
| `laguna-s-2.1-nvfp4` | vLLM | `poolside/Laguna-S-2.1-NVFP4` | 5,120 combined tokens, one scheduled request |
| `laguna-xs-2.1-nvfp4` | vLLM | `poolside/Laguna-XS-2.1-NVFP4` | 262,144 combined tokens, one scheduled request |
| `gpt-oss-120b-mxfp4` | vLLM | `openai/gpt-oss-120b` | 131,072 combined tokens, one scheduled request |
| `gemma-4-26b-a4b-nvfp4` | vLLM | `nvidia/Gemma-4-26B-A4B-NVFP4` | 262,144 combined tokens, eight scheduled multimodal requests |
| `diffusiongemma-26b-a4b-it-nvfp4` | vLLM | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` | 262,144 combined tokens, eight scheduled diffusion requests |
| `mistral-small-4-119b-2603-nvfp4` | vLLM | `mistralai/Mistral-Small-4-119B-2603-NVFP4` | 262,144 combined tokens, one scheduled multimodal request |
| `muse-glimmer-30b-nvfp4` | vLLM + DFlash | `Inferact/Muse-Glimmer-30B-NVFP4-W4A4` + pinned DFlash assistant | 131,072 combined tokens, up to 32 scheduled multimodal requests |
| `ornith-1.5-35b-a3b-nvfp4` | vLLM | `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` | 262,144 combined tokens, one scheduled multimodal request |
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
| [models/gemma-4-26b-a4b-nvfp4.md](models/gemma-4-26b-a4b-nvfp4.md) | Gemma artifact and image pins, multimodal contract, parsers and qualification boundary |
| [models/diffusiongemma-26b-a4b-it-nvfp4.md](models/diffusiongemma-26b-a4b-it-nvfp4.md) | DiffusionGemma artifact pin, canvas policy, multimodal behavior and qualification boundary |
| [models/mistral-small-4-119b-2603-nvfp4.md](models/mistral-small-4-119b-2603-nvfp4.md) | Mistral artifact pin, single-Spark policy, multimodal and reasoning contract |
| [models/muse-glimmer-30b-nvfp4.md](models/muse-glimmer-30b-nvfp4.md) | Muse artifact pin, multimodal contract, parsers and qualification boundary |
| [models/ornith-1.5-35b-a3b-nvfp4.md](models/ornith-1.5-35b-a3b-nvfp4.md) | Ornith artifact pin, multimodal reasoning/tool policy and qualification boundary |
| [models/qwen38-flash-next.md](models/qwen38-flash-next.md) | Flash-Next artifact selection, SGLang, NVMe PLE lifecycle and qualification |
| [sources.md](sources.md) | Primary sources, repository evidence and unresolved verification gates |

## Applying this documentation

The repository now includes `--profile`, engine adapters, profile-scoped prepared state,
dynamic qualification budgets, and the candidate profile directories. The model commits are
pinned; `prepare` resolves each candidate image to an immutable ARM64 digest and checks
its runtime features before downloading the checkpoint.

Read `design.md`, then use the model guides. Ornith release
`af366d840c1b6ae81a5ecd3b` has been prepared and passed startup smoke on the target
GB10 host. No other candidate gains target-host qualification from the source changes.

## Token usage and prefix-cache verification

The vLLM profiles request `enable-prompt-tokens-details: true`; the SGLang profile
requests `enable-cache-report: true`. `prepare` checks those options against the
selected image's CLI; the Rust vLLM profile also checks the Rust frontend help.
The native Chat Completions API test checks input, output and total usage in
non-streaming and final streaming responses, then sends two sequential requests
with a long shared prefix and requires a reported positive hit on the second.
Run it on every newly prepared profile; source and CLI checks cannot prove API
behavior on a running model. A missing/null cache field is unreported, not zero.
The cache count is part of prompt tokens and does not change the context limit.
Responses API is outside this qualification scope.

The candidate vLLM Python `/metrics` endpoint needs no extra switch and exposes
`vllm:prefix_cache_hits_total`, `vllm:prefix_cache_queries_total`, and
`vllm:kv_cache_usage_perc`; verify the exact names on the resolved image and Rust
frontend. The candidate SGLang image requires `--enable-metrics` for `/metrics`;
the current profile leaves that option off. If enabled separately, its current
cache gauges include `sglang:cache_hit_rate`, `sglang:token_usage`, and
`sglang:full_token_usage`. Service-wide hit ratio, KV occupancy, and one request's
`cached_tokens` answer different questions.

Prepared releases are immutable. To apply a source or profile change, run
`bin/spark-llm prepare --profile <profile-id>`, review `bin/spark-llm status`,
then switch during a planned window with
`bin/spark-llm upgrade --profile <profile-id>` and verify using
`bin/spark-llm test --mode api`. Preparation
alone does not change the running release.

Verification on 2026-09-20: all ten profiles passed local validation and all
40 unit tests passed. The locally available vLLM v0.28.0, `gemma`, and
`nightly-20260704` candidate images contain the Python CLI option and native
Chat Completions cache-usage implementation; the Gemma candidate's Rust help
also lists its own `--enable-prompt-tokens-details` option. The locally
available `dev-qwen38-next-local` SGLang image contains the cache-report
option and usage implementation. Ornith release `af366d840c1b6ae81a5ecd3b`
passed startup smoke, including streaming and a tool-call round trip, after its
memory allocation was reduced to preserve the host guardrail. Its full API suite,
real cache hits, `/metrics`, 262K execution and release qualification remain pending.
