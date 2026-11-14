# Plan judging memory protection

`python judge_treatment_plans.py RUN_DIRECTORY` enables this guard automatically; existing concurrency/context options remain available.

The command runs under a lightweight memory supervisor that retains the run
lock and owns the dedicated Ollama process group. It samples macOS pressure,
swap-in/out, compression, approximate reclaimable pages and owned-process RSS
every five seconds. RSS can include shared/GPU pages; these measurements are
diagnostics, not independent quantities to add together. Brief warnings continue. It stops after 60
seconds of warning plus 64 MiB of new swap-out (also covering a 60-second window
with at least 80% warning samples despite brief normal blips), critical pressure,
or three failed monitoring samples. Startup is labeled for diagnostics and never
exempts critical pressure or growing swap from protection.

At 180 seconds of continuous warning, continuing requires complete telemetry
over the last minute, zero new swap-out, less than 64 MiB each of swap-in,
compressor growth and reclaimable-page loss, less than 256 MiB of compression
traffic, and at least 1 GiB of estimated reclaimable pages. Otherwise judging
stops. Substantial churn with less than that reserve can stop after 60 seconds;
churn with ample reserve does not shorten the original 180-second review window.
These are conservative protections, not benchmark-calibrated performance values.
Missing auxiliary readings retain the original strict warning timeout.

On a warning stop, new scalar calls stop and existing work gets up to 30 seconds
to drain; critical pressure skips that grace. Native TERM/KILL and worker-exit
waits are bounded even if a provider call hangs. A forced worker exit may leave
results needing reconstruction, but completed fsynced scalar replies remain
reusable. Only owned process groups are signaled. A redundant signal's macOS
permission error is accepted only if process-table inspection proves the group
has no live members. Genuine or unverifiable failures retain ownership for cleanup.

The supervisor waits at most five minutes for 30 continuous normal seconds with
no new swap-out and at least 1 GiB of reclaimable reserve when available, then
retries once. Both client concurrency and native server slots fall from more
than two to two, from two to one, or stay at one. It never promotes concurrency.
The reserve is an approximate safety floor, not a model-specific admission estimate;
arbitrary model aliases do not provide a reliable pre-load memory budget. A second stop,
monitor failure, or unrecovered headroom exits 75 with progress saved for manual
resume. Every sample, policy, pressure transition and stop/retry decision is recorded
in `provenance/plan-judge-memory-<id>.jsonl`; stage metadata links to that trace.
This protects unattended operation; fallback can extend runtime and cannot
promise completion by morning.
