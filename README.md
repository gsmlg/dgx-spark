# Spark LLM

One native ARM64 vLLM service for `Inferact/Qwen3.8-27B-NVFP4`, exposed as
`local-assistant`. The configured combined input/output context is 262,144 tokens.
**Configured capacity is not qualified capacity.** Check status and reports for actual results.

Host validation on 2026-09-08: release `34d2325c4774623a9d1aef00` passed the
CUDA probe, startup smoke and basic API suite on this GB10 host. Streaming,
English/Chinese, three tool round trips, thinking, cancellation, overlapping requests,
and malformed/unknown-model/media errors passed. The native reasoning field is
`reasoning`; it is empty by default and populated for explicit thinking requests.
An exact 4,096-token formatted fixture was verified through the deployed tokenizer.
256K execution, benchmark, recovery and eight-hour soak qualification are pending.
The running candidate has restart policy `no` until acceptance; it will not
automatically return after a reboot yet. These observations are not full v1 acceptance.

## Setup and use

Requires the existing Docker Engine, Compose, NVIDIA Container Toolkit, Python 3,
PyYAML (`python3-yaml`), and `flock`. No custom image or host driver changes are made.
Run commands from this directory:

```sh
cp -n host.env.example host.env
# Edit host.env to match your host and existing Hugging Face hub cache.
# Optional credentials:
cp -n secrets.env.example secrets.env
chmod 600 secrets.env
bin/spark-llm validate
bin/spark-llm doctor
bin/spark-llm prepare
bin/spark-llm start
bin/spark-llm status
bin/spark-llm test --mode api
```

`host.env` uses literal, unquoted values. Unknown/duplicate keys and substitutions
are rejected. The selected image downloads during preparation; the full model
snapshot is downloaded into the existing cache and checked against Hub file
sizes and SHA256/Git blob identities. The downloader runs in an ephemeral,
GPU-free candidate-image container so no host Python ML packages are needed.
Preparation also executes a small CUDA matrix operation, validates the image's
native ARM64 architecture, entrypoint, parsers and CLI options, and records a
resolved release. Image/model execution compatibility still needs generation tests.

The default cache is `/home/gao/.cache/huggingface/hub`. Docker can write this
existing root-owned cache; the inference container mounts it read-only. Compilation
artifacts use a separate persistent directory under `/home/gao/.cache/spark-llm`.
Do not remove accepted release snapshots or pinned images if rollback is required.
Preparation reserves 30 GiB for this snapshot, 60 GiB for image/cache/extraction,
and a configurable 50 GiB free-space reserve on each relevant filesystem.

The original `v0.28.0-cu130` image alias returned `manifest unknown` during setup.
The profile explicitly uses the published default CUDA 13.0 tag `v0.28.0` instead.
Every prepared release resolves that tag to a real image digest; startup never pulls.
See the [vLLM release artifacts](https://github.com/vllm-project/vllm/releases/tag/v0.28.0).

`prepare` does not stop an existing model. `start` refuses competing GPU containers,
compute processes or an occupied API port. Stop their owner explicitly before launch.
The lifecycle command does not kill unrelated workloads or alter their restart policy.

## Clients

Base URL: `http://127.0.0.1:8000/v1`; model: `local-assistant`.
Native endpoints include `/health`, `/metrics`, `/v1/models`, `/v1/chat/completions`
and the tokenizer endpoint used by the qualification harness. Bind a specific private
IP in `host.env` and set `VLLM_API_KEY` in permission-restricted `secrets.env` for LAN use.
Authentication does not necessarily protect operational routes; keep the entire
endpoint on a trusted private boundary.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-assistant","messages":[{"role":"user","content":"Explain binary search."}],"max_tokens":4096,"stream":true}'
```

Add an Authorization bearer header if a key is configured. Tools execute in your
client, after inspecting the returned tool call. The service does not run tools.
Only text input is supported. Responses API, embeddings and media are outside scope.

Non-thinking is the server default: temperature 0.7, top-p 0.8, top-k 20,
presence penalty 1.5, min-p 0, repetition penalty 1.0. To enable thinking, send
`chat_template_kwargs: {"enable_thinking": true}` together with temperature 1.0,
top-p 0.95, top-k 20 and presence penalty 0. API tests record the actual message
field names in the pinned runtime; do not assume a specific reasoning field before
those tests pass.

Always specify `max_tokens`. The suggested client default is 4,096; it is not a
server-wide output cap. A 16,384 output reservation is valid with at most 245,760
formatted input tokens. Count system/history/tool/template tokens too. One scheduled
sequence means overlapping HTTP requests can wait; it does not bound the HTTP queue.

## Lifecycle

```sh
bin/spark-llm stop
bin/spark-llm restart
bin/spark-llm logs                     # bounded tail with known credentials redacted
bin/spark-llm prepare                  # after an explicit profile edit
bin/spark-llm upgrade                  # use latest prepared release
bin/spark-llm start --release <id>      # select an immutable prepared release
bin/spark-llm rollback                 # restore previous accepted release
```

Mutations share a nonblocking host lock. Prepared releases contain a frozen Compose
file, literal vLLM configuration, artifact hashes, image identity and runtime metadata.
`state/active.json` changes atomically. Source edits do not change a running release.
A candidate is marked running only after health and smoke generation/tool tests pass.
A failed replacement restores the last accepted release when available and still exits
with failure. First installation has no accepted rollback target.

Candidate containers have restart policy `no` while qualification is pending.
Acceptance enables `unless-stopped`: running services return after restart/reboot,
and intentionally stopped services stay stopped. An unhealthy healthcheck alone does
not restart a process. Use `status` and an explicit `restart`. Shutdown allows 300
seconds inside vLLM and 330 seconds in Docker; an exit code of 137 is recorded as a
possible forced termination. This cannot guarantee every long request drains.

## Qualification

Routine `test` defaults to synthetic smoke requests. Pause clients and use the explicit
stress flag for capacity, performance or eight-hour soak tests:

```sh
bin/spark-llm test --mode api
bin/spark-llm test --mode context --disruptive
bin/spark-llm test --mode benchmark --disruptive
bin/spark-llm test --mode soak --disruptive
```

Reports are local, ignored by Git and exclude request/response bodies. They distinguish
requested output from generated tokens, include host memory/swap/OOM observations,
and preserve individual runs. Context fixtures use the deployed pinned tokenizer and
chat template through `/tokenize`, then compare counts with inference usage. Three
seeded near-limit retrieval trials use distinct cache salts. A separate forced-length
stress request reaches exactly 262,144 total tokens; forced output is not a quality test.
Short measurements use 20 warm 4K-input requests with 512-token output budgets.

Recovery checks requiring operator-controlled host events are documented in
[tests/RECOVERY.md](tests/RECOVERY.md). The harness never automatically reboots the host
or injects process faults. Those results and your responsiveness review must be recorded
before acceptance. Run the following only after reviewing the evidence:

```sh
bin/spark-llm accept --evidence reports/reviewed.json
```

The JSON must contain the active `release` ID and true values for
`operator_accepted_responsiveness`, `recovery_suite_passed`, and
`regression_review_passed`. Acceptance additionally requires passing API, context,
benchmark and soak reports for that release. Retain recovery observations with that
review. Compare equivalent benchmark conditions with the prior accepted baseline;
a >20% median TTFT increase or decode-rate decrease needs an explicit review decision.

## Troubleshooting

- **Configuration error:** run `validate`; inspect profile versus host ownership.
- **Image unavailable:** verify the explicit tag exists; never insert a fake digest.
- **GPU busy / port used:** inspect `doctor`, then explicitly stop the conflicting owner.
- **Startup failure:** inspect `logs` and `reports/last-failure.json`. Do not install
  packages inside the container, bypass driver checks or silently reduce context.
- **Cache incomplete:** rerun `prepare`; it verifies and resumes the pinned snapshot.
- **OOM / swap / 256K failure:** retain failed reports. Any backend/cache precision
  adjustment needs a new resolved release and requalification.

The full v1 acceptance criteria remain in [PRD.md](PRD.md) and [DESIGN.md](DESIGN.md).
