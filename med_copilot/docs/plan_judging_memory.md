# Plan judging memory protection

`python judge_treatment_plans.py RUN_DIRECTORY` enables this guard automatically; existing concurrency/context options remain available.

The command runs under a lightweight memory supervisor that retains the run
lock and owns the dedicated Ollama process group. It samples macOS pressure,
swap-in/out, compression, approximate reclaimable pages and owned-process RSS
every five seconds. RSS can include shared/GPU pages; these measurements are
diagnostics, not independent quantities to add together. Critical pressure stops
immediately, without waiting for auxiliary readings. Three consecutive failed
monitoring samples also stop judging. Startup is labeled for diagnostics and
receives the same protection.

Warning pressure, compression traffic and an isolated paging burst do not by
themselves establish exhaustion. Under sustained or predominantly warning
pressure, paging protection requires at least 64 MiB plus distributed activity:
either activity over at least half a minute spanning 40 seconds in the latest
minute, or three episodes separated by at least ten quiet seconds and spanning
at least a minute within two minutes. Sub-MiB interval increments do not count
as activity. A single 15-second burst dominating over 80% of the window cannot
turn background trickle into sustained paging.

Known reclaimable reserve below 1 GiB for 30 continuous warning seconds stops
independently. After 180 seconds of warning, a complete recent minute of auxiliary
telemetry is required; missing readings or sampling gaps cannot authorize blind
continuation. Counter resets discard the prior activity evidence. Compression
traffic remains diagnostic even when high, and low-rate paging may continue with
adequate reserve. These are conservative protection thresholds, not a guarantee
of a particular runtime or of preventing every OS intervention.

On a warning stop, new scalar calls stop and existing work gets up to 30 seconds
to drain; critical pressure skips that grace. Native TERM/KILL and worker-exit
waits are bounded even if a provider call hangs. A forced worker exit may leave
results needing reconstruction, but completed fsynced scalar replies remain
reusable. Only owned process groups are signaled. A redundant signal's macOS
permission error is accepted only if process-table inspection proves the group
has no live members. Genuine or unverifiable failures retain ownership for cleanup.

Completed replies are durably saved independently of the guard. Even after a
crash, they remain reusable; a few in-flight requests may need repeating.

Reply lookup uses a disposable SQLite file in the system temporary directory,
with a 2 MiB page-cache target and memory mapping disabled. It stores request
keys, byte offsets, lengths and checksums, rather than decoded reply payloads.
Each hit reads one audit record on demand and rechecks its checksum and parser.
Opening the cache still streams and validates every complete JSONL line; it
rebuilds the index, repairs only an unterminated tail after validation, and
fsyncs recovered evidence before reuse. The last valid duplicate wins.

JSONL remains the sole durable authority: appending/fsyncing precedes index
publication. The derived index does not need durable commits and is never reused
on restart. An index/write failure or interrupted append stops that cache owner
from publishing or appending more replies; a fresh owner reconstructs progress
from the audit. Normal cleanup deletes the temporary index. Abrupt process exit
can leave disposable scratch files, which are not used for recovery.

The supervisor waits at most five minutes for 30 continuous normal seconds with
no new swap-out and at least 1 GiB of reclaimable reserve when available, then
retries once. Both client concurrency and native server slots fall from more
than two to two, from two to one, or stay at one. It never promotes concurrency.
The reserve is an approximate safety floor, not a model-specific admission estimate;
arbitrary model aliases do not provide a reliable pre-load memory budget. A second stop,
monitor failure, or unrecovered headroom exits 75 with progress saved for manual
resume. Every sample, policy, pressure transition and stop/retry decision is recorded
in `provenance/plan-judge-memory-<id>.jsonl`; stage metadata links to that trace.
The versioned policy metadata records every effective detection threshold.
This protects unattended operation; fallback can extend runtime and cannot
promise completion by morning.
