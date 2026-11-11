# Qwen3.8 Flash-Next FP4-Local with B12X on one DGX Spark

**Reviewed:** 2026-10-23. **Local diagnosis:** 2026-11-09.
**Profile:** `qwen38-flash-next-fp4-local-b12x`.

**State:** Startup failed after checkpoint loading, before HTTP readiness. The pinned
release encountered NVIDIA driver memory-allocation failures and an unclean host
interruption. Generation and target-host long-context qualification remain untested.
Both B12X and the previous Ornith service are stopped; restoration was not requested.

## Artifact and runtime pins

The official checkpoint is `nvidia/Qwen3.8-Flash-Next-FP4-Local`, QAD FP4 1.0,
released on 2026-10-05. It is distinct from the older
`nvidia/Qwen3.8-Flash-Next-NVFP4` used by the SGLang profile. The model card reports
about 96 GiB on disk and validates single-Spark use with PLE offload. Its losslessly
compressed NVFP4 scales require the special runtime; ordinary upstream vLLM is not
a fallback. [F1](../sources.md)

| Component | Pin / configured policy |
|---|---|
| Checkpoint and tokenizer | `b9ce18a6887e18c089fdea8509a33e63429a3300` |
| Container tag used for resolution | `eugr/spark-vllm-b12x:nightly-20261015` |
| Immutable image | `eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595` |
| Registry platform | `linux/arm64`; GB10 / SM121 |
| Container CUDA | 13.0.2 in the published image configuration |
| Reported PyTorch | 2.13.0; verify the installed version during admission |
| Combined input and output budget | 262,144 tokens |
| Tensor / pipeline parallelism | 1 / 1 |
| Scheduler concurrency | At most 8 sequences; simultaneous full windows unqualified |
| Scheduler batch | 4,096 tokens, chunked prefill |
| Precision | BF16 activations, `modelopt_mixed`, explicit FP8 KV |
| Loader and kernels | `b12x` loader, linear/MoE/GDN backends |
| PLE | Disk, `io_uring`, direct reads from the pinned safetensors shards |
| MTP | Built-in checkpoint head, K=4, following NVIDIA's Spark example |
| Initial memory utilization | 0.80, subject to the existing 16 GiB host-available floor |
| Input contract | Text-only initial qualification |

The NVIDIA example and community solo recipe use these core runtime settings.
The profile also disables request/output logs, uses the existing private host binding,
and preserves offline, read-only checkpoint mounts. It enables prefix-cache usage
reporting, `qwen3` reasoning and `qwen3_xml` tool parsing. Thinking defaults on and
can be disabled through `chat_template_kwargs.enable_thinking`; sampling defaults
follow the pinned generation configuration (temperature 1.0, top-p 0.95, top-k 20).
The checkpoint includes vision, but image/video serving needs a separately qualified
multimodal release. [F1, F2](../sources.md)

This profile's B12X environment explicitly forces the A16 MoE path. Do not infer
native FP4 arithmetic from the stored NVFP4 weights; retain the resolved kernel and
activation-precision logs during runtime qualification.

## Driver, containment and storage

Keep NVIDIA R580. The target host currently reports **580.178.04**; adding this
profile does not install a driver, CUDA toolkit or host ML package. CUDA 13 family
compatibility and an ordinary CUDA matrix probe do not certify every B12X kernel.
No stricter minimum driver patch is asserted without publisher evidence.

Only this B12X runtime receives `seccomp=unconfined`, `apparmor=unconfined`, unlimited
memlock and a 64 MiB stack limit, matching the NVIDIA Spark container example.
The common host-network/host-IPC deployment and these exceptions do not provide a
strong security boundary. Use a trusted isolated test host or a controlled test window
with other clients paused. The container is not privileged; other profiles retain their
existing security settings. The image digest is fixed in source rather than relying on
`latest` at preparation. [F1](../sources.md)

Model weights remain under `HF_CACHE`, mounted read-only. In the reviewed B12X
implementation, `DiskTable.add_shard()` binds each PLE shard's checkpoint path and
byte offset; it does not materialize the SGLang-style derived table. Consequently
`derived-cache` is null and this profile does not mount or reset `/ple-cache`.
Confirm that contract in the pinned image during admission. Never remove checkpoint
blobs or hand-delete mapped files. [F5](../sources.md)

Compile caches are scoped to the image/configuration under `RUNTIME_CACHE`. B12X's
`/root/.cache` is bound to that same owned runtime cache; Triton uses
`/runtime-cache/triton`. The host home directory is not mounted. Fresh and repeated
startup both require measurement.

The profile sets `prepare-storage-gib: 160` as a conservative artifact/storage allowance; with the example
60 GiB extraction/cache headroom and 50 GiB free reserve, the initial disk gate is
270 GiB. Plan for the complete 96 GiB snapshot, image extraction,
compile artifacts and retained rollback files. Local NVMe is required for the intended
I/O behavior. PLE offload still consumes page cache and storage bandwidth.

## Stage without switching the running service

```sh
bin/spark-llm validate --profile qwen38-flash-next-fp4-local-b12x
bin/spark-llm download --profile qwen38-flash-next-fp4-local-b12x
bin/spark-llm prepare --profile qwen38-flash-next-fp4-local-b12x
bin/spark-llm status
```

Download uses a GPU-free ephemeral container. Preparation performs a small CUDA
operation and CLI/loader/parser checks, then freezes an immutable release. Check
available unified memory before that probe if another model is resident. These
operations do not replace the active model. Starting does replace it: pause clients
and choose an isolated test window before running:

```sh
bin/spark-llm start --profile qwen38-flash-next-fp4-local-b12x
bin/spark-llm test --mode api
# Run only after startup/API pass, with clients paused:
bin/spark-llm test --mode context --disruptive
```

Retain any admission, loader, kernel, memory-floor or PLE I/O failure. The October 15
image has community reports of transient loading/autotuning memory spikes on one
Spark; reducing the KV utilization does not prove that these transient allocations
are bounded. Do not weaken the host guardrail or patch packages inside a running
container to force a pass. [F4](../sources.md)

## Qualification sequence and evidence boundary

Run these gates in order on the exact prepared release:

1. Load, readiness, short generation, streaming, reasoning/tool round trips and API errors.
2. One approximately 128K combined session, then one near-256K combined session, each
   reserving room for output. Use tokenizer-measured input sizes and check retrieval
   correctness plus recovery with a short request afterward.
3. Two simultaneous approximately 128K sessions; record the actual overlap and both
   outputs before attempting two simultaneous near-256K sessions.
4. Measure throughput and repeated startup, then recovery and an eight-hour soak.

The existing context harness checks half-window and near-window requests sequentially;
its API overlap test uses short requests. Neither qualifies two long sessions running
at once. That concurrency gate remains separate and pending.

For each gate record release/digest/revision, driver/kernel versions, minimum
`MemAvailable`, swap-in/out, PLE file size and NVMe throughput/latency, prefill latency
and time to first token, decode rate, errors and post-request recovery. Use `vmstat 1`,
`iostat -dx 1` where available, the bounded engine logs and release reports. Preserve
the 16 GiB host memory floor throughout startup and testing.

NVIDIA's statement of approximately one million tokens of FP8 KV capacity with PLE
offload is a pool-capacity observation. It is not proof of one-million-token inference
or two complete 256K sessions. This profile leaves the native context at 262,144 and
does not enable YaRN extension. [F1](../sources.md)

The supplied community report quotes about 81 GiB runtime residency, 66.2 tok/s for
one decode stream and 166.4 tok/s aggregate for four short streams at MTP K=3.
Those measurements have not been reproduced here, do not establish four full-context
sessions, and are not guarantees for the K=4 candidate. Consult the reproducible solo
recipe and community discussion separately from local qualification. [F2, F3](../sources.md)

## Local integration checks

All 33 checkpoint files were downloaded and hashed: 95.24 GiB at the pinned revision.
The image's compressed layers, uncompressed layer identities, configuration ID and
native ARM64 repository digest were verified before preparation. All 51 repository
unit tests passed during integration. The runtime capability probe was extended to
check the quantization registry because CLI help omits `modelopt_mixed`.

Release `48bd69e896ef47f8de0b74b0` passed CUDA admission on driver **580.178.04** and
kernel **7.0.0-1019-nvidia**, using PyTorch **2.13.0+cu130** and vLLM
**0.1.dev21830+g8244d1c44.d20261015**. The image declares CUDA 13.0.2;
`torch.version.cuda` reports the family as `13.0`. Its PLE implementation binds
checkpoint paths and offsets directly with `DiskTable.add_shard()`.

The startup attempt on **2026-10-24 (HKT)** failed:

- **00:16:57:** all 21 checkpoint shards finished loading in 134.19 seconds;
  selected payload was 71.41 decimal GB. This is not total runtime residency.
- After loading, an interactive host sample showed approximately **10 GiB available**
  and **1.6 GiB swap used**, below the configured 16 GiB available-memory floor.
- **00:17:14–00:18:22:** the kernel recorded **149** NVIDIA
  `_memdescAllocInternal: NV_ERR_NO_MEMORY` failures.
- The previous boot's journal ended without an orderly shutdown record. The next
  boot began at approximately **00:20:58**; Docker recorded container exit **255**
  while restoring container state at **00:21:04**. That code does not identify a
  Python/application exception.

The failure is localized to the post-load interval, consistent with weight
finalization/repacking memory pressure. No KV-capacity, autotuning-start, HTTP-ready
or generation-success event was recorded. The bounded journal has no Linux OOM-kill,
NVRM Xid or panic/lockup record; Docker reports `OOMKilled=false`. The exact failing
allocation and whether a driver defect contributed remain unproven. The same NVIDIA
allocation-error signature also predates this model in the host's September journal.
Lowering KV utilization alone is not an established fix for a failure at this stage.

There is a separate, reproduced configuration defect in the pinned runtime:
`enable-flashinfer-autotune: false` becomes omitted while the top-level CLI parser
loads YAML, because its negative option belongs to the `serve` subparser. The engine
therefore logged `enable_flashinfer_autotune=True`. A GPU-free reproduction confirmed
the root/subparser difference; nested JSON/YAML under
`kernel-config.enable_flashinfer_autotune` preserves `false`. Supporting that nested
setting in this repository is a follow-up, not an applied fix or proof that autotuning
caused this failure. The unchanged profile should not be treated as a verified recipe.

Evidence and diagnostic limits are preserved in
[`reports/48bd69e896ef47f8de0b74b0/diagnosis.md`](../../../reports/48bd69e896ef47f8de0b74b0/diagnosis.md)
with the container log, bounded journal and structured summary. No GPU startup was
retried and neither service was restored during diagnosis. Smoke, API, context,
concurrency, benchmark and soak qualification remain unrun for this release.
