# Sources and verification boundaries

**Reviewed:** 2026-09-20. Sources are primary project/model documentation and repository code. Upstream pages are mutable; re-resolve revisions before implementation. Short identifiers below are citation keys used throughout this package.

## Repository evidence

The GitHub recursive tree endpoint reported tree identity `21ea8fe0f483be4e23b2d8e201ce3dd5c70f6e11`. This is recorded as a **tree identity**, not asserted to be a commit SHA. The reviewed content blobs include `f9f3b4ed823f69514b60595b48a32c88ab022ccb` for lifecycle, `443559588f208228a34ca5bdb9a356101d2e110a` for acceptance tests, and `8797c2169665410db4d7b4b83836aa168e39f2f9` for Compose.

| Key | Source | Used for |
|---|---|---|
| R1 | [Repository README](https://github.com/gsmlg/dgx-spark/blob/main/README.md) | Current model, host-validation statement, image-tag correction, pending qualification |
| R2 | [Lifecycle implementation](https://github.com/gsmlg/dgx-spark/blob/main/lib/lifecycle.py) | Profile constants, exact schema, entrypoint/parser probes, state, acceptance and disk assumptions |
| R3 | [Acceptance harness](https://github.com/gsmlg/dgx-spark/blob/main/tests/acceptance.py) | Reasoning/tool history, tokenizer, fixed context/soak and report handling |
| R4 | [Compose definition](https://github.com/gsmlg/dgx-spark/blob/main/compose.yaml) | vLLM invocation, private serving, offline mounts, cache and healthcheck |
| R5 | [Artifact preparer](https://github.com/gsmlg/dgx-spark/blob/main/lib/prepare_model.py) | Full snapshot and integrity checks |
| R6 | [Current Qwen profile](https://github.com/gsmlg/dgx-spark/blob/main/profiles/qwen38-27b-nvfp4/vllm.yaml) | Existing context, parser and sampling assumptions |
| R7 | [Original design](https://github.com/gsmlg/dgx-spark/blob/main/DESIGN.md) | v1 scope, host safety margin and architecture to preserve |

The original repository review was remote. The Laguna XS integration has local configuration and unit-test evidence, but no model download or GPU qualification.

## Laguna sources

| Key | Source | Used for |
|---|---|---|
| L1 | [Poolside NVFP4 model card](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4) | Model ID, updated checkpoint warning, native context, Spark/JIT guidance, optional DFlash |
| L2 | [Poolside NVFP4 file listing](https://huggingface.co/poolside/Laguna-S-2.1-NVFP4/tree/main) | Observed 99.7 GB repository total; revision-specific artifact planning |
| L3 | [Official vLLM Laguna recipe](https://recipes.vllm.ai/poolside/Laguna-S-2.1) | Native serving/parser and generation configuration guidance |
| L4 | [Poolside base model card](https://huggingface.co/poolside/Laguna-S-2.1) | Reasoning controls and interleaved/preserved reasoning |

**Conflict to resolve:** the NVFP4 card's approximate size and Spark recipe numbers should not be applied blindly to the current file listing. The card says weights changed. Inspect the selected full revision, weight index and actual loader allocation. No claim is made that 99.7 GB of repository files equals 99.7 GB of GPU-resident weights.

**Caution:** the model repository has revised its SGLang guidance. The proposed route remains vLLM to reuse the existing runtime; that choice is not a claim that current SGLang cannot run Laguna. Verify exact quantization and runtime compatibility rather than infer it from general architecture support.

## Laguna XS sources

| Key | Source | Used for |
|---|---|---|
| X1 | [Poolside Laguna XS 2.1 NVFP4 model card](https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4) | Model identity, runtime minimums, quantization, parsers and reasoning behavior |
| X2 | [Pinned checkpoint configuration](https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4/blob/d32afde8b09af1539b49ff96ff5551c674485f8e/config.json) | Native context, architecture and quantization metadata |
| X3 | [Pinned generation configuration](https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4/blob/d32afde8b09af1539b49ff96ff5551c674485f8e/generation_config.json) | Sampling, reasoning and parser defaults |

## Flash-Next sources

| Key | Source | Used for |
|---|---|---|
| Q1 | [Official Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Architecture/parameter breakdown, native context, reasoning and sampling controls |
| Q2 | [Official SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next) | Spark image candidate, single-node NVMe route, memory and startup caveats |
| Q3 | [NVIDIA NVFP4 export](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | Exact NVIDIA artifact identity and quantization documentation |
| Q4 | [RadixArk NVFP4 export](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | Explicit alternative artifact, not an automatic replacement |
| Q5 | [SGLang PR #37068](https://github.com/sgl-project/sglang/pull/37068) | File-backed PLE implementation, device capability, RSS/page-cache distinction |
| Q6 | [SGLang PR #38121](https://github.com/sgl-project/sglang/pull/38121) | NVIDIA mixed-precision loader and expert-backend requirements |
| Q7 | [SGLang model-support PR #36497](https://github.com/sgl-project/sglang/pull/36497) | Referenced model-support branch; verify inclusion in the chosen runtime |

PR #37068 was reported merged into `qwen4-main-squashed`, merge commit `3a09f089e068fe4295cccf246a94436b3215f22b`. PR #38121 was reported merged into the same branch, merge commit `9b2aee22836b2bfe620bf83861919870d6692660`. These are source-provenance observations, **not selected image digests**. Verify the actual candidate contains the complete required fixes.

**Startup conflict:** PR #37068's narrative contains inconsistent warm-start claims, while the current cookbook explicitly warns about rewriting an existing populated table. The design requires measuring both paths and does not promise file reuse makes restart fast.

**Memory clarification:** the cookbook's language about an 8 GiB bound must not be read as a whole-host page-cache quota. PR #37068 describes trimming mappings while pages can remain in cache. The design therefore uses independent host memory and I/O observation.

**Reasoning/sampling conflict:** the Qwen model card documents thinking controls and sampling guidance, but sections of the SGLang cookbook describe always-on reasoning and say sampling recommendations are unavailable. Resolve behavior against the pinned template and actual engine tests. Unsupported modes are recorded explicitly, not silently emulated or reported as passed.

## GPT-OSS sources

| Key | Source | Used for |
|---|---|---|
| G1 | [Official OpenAI GPT-OSS 120B model page](https://developers.openai.com/api/docs/models/gpt-oss-120b) | Model identity, parameter counts, context, reasoning and tool capabilities |
| G2 | [Pinned checkpoint configuration](https://huggingface.co/openai/gpt-oss-120b/blob/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json) | Architecture, native context and MXFP4 quantization metadata |
| G3 | [Official vLLM GPT-OSS recipe](https://github.com/vllm-project/recipes/blob/main/OpenAI/GPT-OSS.md) | Parser, FP8 KV cache, scheduler and prefix-cache guidance |
| G4 | [NVIDIA DGX Spark vLLM playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md) | DGX Spark support listing for the OpenAI MXFP4 checkpoint |
| G5 | [OpenAI Harmony encoding source](https://github.com/openai/harmony/blob/main/src/tiktoken_ext/public_encodings.rs) | Offline vocabulary filename, environment override and expected SHA256 |

The full revision and runtime image digest remain separate pins. The checkpoint commit
above was resolved from the Hub on the review date; `prepare` resolves and records the
actual ARM64 image digest. Source support statements do not replace GB10 startup,
parser, long-context, throughput, soak or recovery qualification.

## DiffusionGemma sources

| Key | Source | Used for |
|---|---|---|
| D1 | [Official vLLM DiffusionGemma recipe](https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it) | NVIDIA checkpoint, DGX Spark overrides, diffusion canvas, parsers, multimodal behavior and limitations |
| D2 | [Pinned NVIDIA NVFP4 checkpoint configuration](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4/blob/ec4ff3df205028f4e81c954c2227f9312b3ec2ea/config.json) | Architecture, native context and ModelOpt NVFP4 plus FP8 KV metadata |
| D3 | [Pinned generation configuration](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4/blob/ec4ff3df205028f4e81c954c2227f9312b3ec2ea/generation_config.json) | Denoising thresholds, entropy sampler and the fixed output cap that the recipe clears |
| D4 | [Original Google model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it) | Model identity, block-diffusion behavior and provenance |

The checkpoint revision and `vllm/vllm-openai:gemma` ARM64 manifest were resolved on
2026-09-15. The image tag remains mutable until `prepare` records its registry digest.
Recipe performance figures are upstream measurements, not qualification results for this
repository or host.

## Gemma 4 sources

| Key | Source | Used for |
|---|---|---|
| E1 | [Official vLLM Gemma 4 recipe](https://recipes.vllm.ai/Google/gemma-4-26B-A4B-it) | DGX Spark image and scheduler policy, NVIDIA checkpoint, parsers, multimodal behavior and MTP boundary |
| E2 | [Pinned NVIDIA NVFP4 checkpoint configuration](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4/blob/a19cfe00be84568a6867111c9a68c9c44fdcffe6/config.json) | Architecture, 262K context and ModelOpt NVFP4 W4A4 metadata |
| E3 | [Pinned generation configuration](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4/blob/a19cfe00be84568a6867111c9a68c9c44fdcffe6/generation_config.json) | Published stochastic sampling defaults |
| E4 | [Original Google model card](https://huggingface.co/google/gemma-4-26B-A4B-it) | Model identity, multimodal capabilities, reasoning and tool behavior |

The Hub revision and the recipe image manifest were resolved on 2026-09-15. The recipe
reports this checkpoint and ARM64 image as the tested single-DGX-Spark route. The image
tag remains mutable until `prepare` records its registry digest. MTP is deliberately not
configured because this lifecycle currently verifies one model snapshot per release; a
second assistant checkpoint must not bypass that integrity boundary.

## Muse Glimmer sources

| Key | Source | Used for |
|---|---|---|
| M1 | [Official vLLM Muse Glimmer recipe](https://recipes.vllm.ai/meta-models/Muse-Glimmer-30B) | DGX Spark checkpoint choice, vLLM version, context, parsers, sampling and NVFP4 behavior |
| M2 | [Pinned NVFP4 checkpoint configuration](https://huggingface.co/Inferact/Muse-Glimmer-30B-NVFP4-W4A4/blob/d35cb79050f419c457611b1cee5c5d15b176f285/config.json) | Architecture, native context, vision configuration and ModelOpt NVFP4 W4A4 metadata |
| M3 | [Pinned generation configuration](https://huggingface.co/Inferact/Muse-Glimmer-30B-NVFP4-W4A4/blob/d35cb79050f419c457611b1cee5c5d15b176f285/generation_config.json) | Published token IDs and generation boundary |
| M4 | [Original Muse Glimmer model card](https://huggingface.co/meta-models/Muse-Glimmer-30B) | Dense parameter count, image-text behavior and model provenance |
| M5 | [Pinned Muse Glimmer assistant checkpoint](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant/tree/e8192f3a8f617f74be2ce220360c89ef4789f39f) | Immutable DFlash draft configuration and weights |

The target and assistant Hub revisions were resolved on 2026-09-15. The recipe reports the NVFP4 checkpoint
as the 25.42 GB, Blackwell-only, single-DGX-Spark choice. That upstream support statement
does not replace this repository's image probe, startup, DFlash, multimodal, context,
concurrency, benchmark, soak or recovery qualification.

## Mistral Small 4 sources

| Key | Source | Used for |
|---|---|---|
| MS1 | [Official NVFP4 model card](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-NVFP4) | Model identity, vLLM settings, context, multimodal behavior, parsers and reasoning control |
| MS2 | [Pinned checkpoint parameters](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-NVFP4/blob/45331841b631f4e281df8e959ea3cc9beb84298a/params.json) | Architecture, context scaling, vision encoder and compressed-tensors NVFP4 metadata |
| MS3 | [Pinned chat template](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-NVFP4/blob/45331841b631f4e281df8e959ea3cc9beb84298a/chat_template.jinja) | `reasoning_effort` values, image/tool history and default non-reasoning behavior |

The Hub revision was resolved on 2026-09-15. The publisher's example uses two GPUs,
128 sequences and a 16,384-token scheduler batch. This repository deliberately starts
with TP1, one sequence and a 4,096-token batch for the single-GPU DGX Spark. That
adaptation, the v0.28.0 ARM64 image, 262K capacity, parsers, multimodal path, memory
headroom and performance all remain candidate settings until preparation and host tests.

## Ornith 1.5 sources

| Key | Source | Used for |
|---|---|---|
| O1 | [Official Ornith 1.5 35B-A3B model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) | Architecture, runtime minimums, native context, multimodal behavior, parsers, sampling and long-context warning |
| O2 | [Pinned official NVFP4 checkpoint configuration](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4/blob/94e431d9cc47fa1986a7a1a4e9a80f7f118b03aa/config.json) | Qwen3.5-MoE architecture, vision encoder, native context and ModelOpt mixed NVFP4/FP8 metadata |
| O3 | [Pinned official chat template](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4/blob/94e431d9cc47fa1986a7a1a4e9a80f7f118b03aa/chat_template.jinja) | Thinking toggle, reasoning history, multimodal markers and XML tool-call format |
| O4 | [vLLM 0.28.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.28.0) | Selected runtime release identity; exact ARM64 image digest remains preparation-time evidence |

The official NVFP4 repository revision was resolved from the Hub on 2026-09-20. Its
model card currently repeats serving commands for the BF16 repository, so artifact
selection comes from the explicit NVFP4 repository while behavioral settings come from
the base model card and the pinned NVFP4 config/template. Target-host release
`af366d840c1b6ae81a5ecd3b` proves that the pinned checkpoint loads on GB10 and that
health, streaming and a tool-call round trip pass. The selected vLLM path uses Marlin
weight-only FP4 rather than native FP4 compute. The 262K target, full API suite,
multimodal path, output quality and performance remain unqualified.

## Hardware

| Key | Source | Used for |
|---|---|---|
| H1 | [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) | Arm platform and 128 GB unified CPU/GPU memory |

## Not verified by this documentation task

The current driver/kernel and free storage on Jonathan's Spark; new model full revisions; container ARM64 digests and installed kernel libraries; checkpoint resident-memory allocation; PLE restart behavior on this host; tokenizer/stream/tool correctness in the chosen images; exact 32K or larger capacity; speed; eight-hour stability; restart/reboot recovery.

The documents specify how to verify these gates. None has been converted into a claimed local result. All first-release numeric defaults are proposed policy unless explicitly attributed to a source.
