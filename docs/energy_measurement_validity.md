# Energy measurement validity (peer-review reference)

What our `joules` field actually means, what's isolated, and how to handle
outliers. Triggered by post-hoc analysis of matrix
`swebench_25_vanilla_vs_sharedfs_vs_boad_seed0` where two `vanilla`
episodes had a single ~602-second LLM call inflating their energy by
40-77%.

## Measurement chain

Per LLM call, `LLMCallLogHooks` snapshots the vLLM host's `/energy`
endpoint at `on_llm_start` and `on_llm_end`. The diff (mJ summed across
NVML-visible GPUs) is recorded as `joules` on the `LLMCallRecord`. The
endpoint is implemented by `scripts/serving/vllm_with_energy.py` and
reads `nvmlDeviceGetTotalEnergyConsumption` — a per-GPU monotonic
cumulative-mJ counter exposed by NVML on Ampere+ hardware.

## What is isolated (verified)

| Source | Isolation mechanism |
|---|---|
| Other GPUs on same physical node | `nvmlDeviceGetTotalEnergyConsumption(handle)` is per-GPU. Each GPU has its own hardware counter. |
| Other vLLM tenants on the same node | The serving process only sees the GPUs exposed to it (`CUDA_VISIBLE_DEVICES`, or the device cgroup when it runs under a scheduler or container). Other tenants' GPUs are invisible to our `/energy` endpoint. |
| CPU activity (any user) | NVML measures GPU chip + memory + voltage regulator power only. CPU power is on RAPL and not read. |
| Network / PCIe noise | Negligible (microwatts) impact on GPU power envelope. |
| Other physical nodes | Independent hardware. |

Verified empirically across the matrix run: every one of 2,507 LLM
calls reported exactly 1 GPU in `joules_per_gpu`, confirming TP=1
allocation and per-GPU isolation throughout.

## What is *not* isolated (must be flagged)

| Source | Mechanism | Mitigation in this codebase |
|---|---|---|
| Concurrent requests to the **same vLLM** | vLLM's continuous batcher executes neighbor requests' forward passes on our GPU during our call's wall-time window — counted in our `joules`. | `scripts/run_matrix.py` uses one OS thread per URL (`worker_loop`, line 224); each thread runs episodes serially — at most one `Runner.run` per URL at a time. Verified in [Concurrency check](#concurrency-check) below. |
| **Idle baseline** (~60-70 W per A100) | NVML counts all energy regardless of whether useful inference is happening. A call that stalls in vLLM still accumulates baseline energy proportional to wall time. | None at the framework level — analysis must inspect outliers and report median / outlier-stripped statistics (see [Long-call anomaly](#long-call-anomaly)). |
| vLLM warmup / first-prefill compile | CUDA graph compile in vLLM startup adds ~30-60s of extra GPU activity to the first request after launch. | The startup window is before any matrix episode; subsequent calls see steady state. |

## Concurrency check (matrix #8 reference)

For both 602s outlier calls in matrix #8, dispatched via
`run_matrix.py` with `--reset-prefix-cache`:

- **`django__django-14034 vanilla`** — 602.3s call window 11:05:36-11:15:39.
  URL 18061's serial dispatch order at that time (per
  `/tmp/swe25_3agents.log`): `django-12304 vanilla → astropy-14539 vanilla →
  django-14034 vanilla`. URL was DEDICATED to django-14034 during the
  long call.
- **`django__django-12125 vanilla`** — 602.9s call window 11:19:07-11:29:10.
  its worker URL next in queue after django-14034: `→ django-12125 vanilla
  (started 11:17:52)`. Again single-tenant.

The `worker_loop`'s `for idx, run in queue:` blocks on
`subprocess.run()`, so episodes within a URL run strictly serially.
Other vLLM workers ran independent episodes on separate GPUs.

**Pitfall hit during analysis**: `/tmp/run_matrix_runs/` accumulates
log files across matrix runs without clearing. A first-pass attempt to
infer "URL of episode" by scanning these log filenames produced false
overlaps because stale files from a killed prior matrix attempt
(matrix #7) appeared as if they were from the current matrix. Always
cross-reference with the dispatcher log (`/tmp/<title>.log`) for
ground truth, not the per-run log directory.

## Long-call anomaly

Two `vanilla` episodes had a single LLM call of nearly identical
duration (602.3s and 602.9s — suggesting a specific timeout boundary
rather than random slowness). Both produced structurally normal
output: a valid `str_replace_editor view` action with 177 / 289
output tokens, followed by a normal observation and a normal next
turn. No `error` field was set. GPU drew ~210 W throughout.

Effective decode rate: ~0.5 tok/s. Expected vLLM decode rate for
Qwen3.5-35B-A3B-FP8: 50-100 tok/s. The gap is unexplained from
client-side logs alone (no vLLM server logs collected). Possible
causes — KV-cache eviction storm, scheduler edge case, internal
recompile.

These two calls account for 75% of those episodes' total energy
(125,876 J = 44% on django-12125, 129,280 J = 77% on django-14034).
None of the 14 `sharedfs` episodes exhibited the pattern.

**Reporting convention**: in aggregate comparisons, report both
median per-episode energy AND outlier-stripped mean. The two
outliers above should be flagged as "vLLM operational anomaly,
not an intrinsic property of the agent."

## Per-call vs aggregate interpretation

`joules` per call is "GPU energy spent during this LLM call's wall
time." It is correctly attributed when the URL is single-tenant
(verified above). It is NOT the same as "compute cost of this call's
tokens" — when a call's wall time is inflated by stalls, the per-call
J inflates proportionally even though useful compute is small.

For per-call efficiency comparisons (which agent uses less compute
per useful unit?), prefer:
- `J / output_tokens` (compute-normalized)
- `J / turn` (request-normalized)
- median rather than mean across episodes

For total-energy comparisons (how much GPU energy did the run
consume?), the raw sum is correct but should be reported alongside
the medians and any outlier callouts.
