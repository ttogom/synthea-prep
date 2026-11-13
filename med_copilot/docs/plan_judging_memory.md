# Plan judging memory protection

`python judge_treatment_plans.py RUN_DIRECTORY` enables this guard automatically; existing concurrency/context options remain available.

The command runs under a lightweight memory supervisor that retains the run
lock and owns the dedicated Ollama process group. It samples macOS pressure and
new swap-out every five seconds. Brief warnings continue. It stops after 60
seconds of warning plus 64 MiB of new swap-out (also covering a 60-second window
with at least 80% warning samples despite brief normal blips), 180 seconds of warning alone,
critical pressure, or three failed monitoring samples. Request latency remains
diagnostic, not a stop trigger.

On a warning stop, new scalar calls stop and existing work gets up to 30 seconds
to drain; critical pressure skips that grace. Native TERM/KILL and worker-exit
waits are bounded even if a provider call hangs. A forced worker exit may leave
results needing reconstruction, but completed fsynced scalar replies remain
reusable. Only owned process groups are signaled.

The supervisor waits at most five minutes for 30 continuous normal seconds with
no new swap-out, then retries once with both client concurrency and native server
slots reduced to two. It never promotes itself back to four slots. A second stop,
monitor failure, or unrecovered headroom exits 75 with progress saved for manual
resume. The policy, pressure transitions and stop/retry decisions are recorded
in `provenance/plan-judge-memory-<id>.jsonl`; stage metadata links to that trace.
This protects unattended operation; fallback can extend runtime and cannot
promise completion by morning.
