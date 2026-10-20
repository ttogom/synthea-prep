#!/usr/bin/env python3
"""Convert patient records to SOAP JSON using `claude -p` on a Claude subscription.

  corpus -> subjective, objective, assessment, plan   (CORPUS_PROMPT)
  dev/test -> subjective, objective only              (DEV_TEST_PROMPT)

Prompts are hard-coded below, SOAP_CONVERSION.md verbatim; the record replaces {text} and the
whole prompt is sent as the user message (empty system prompt).

Usage:
  python3 soap.py pilot [--workers N]   # ~53 stratified records -> soap_pilot/
  python3 soap.py run [--workers N]     # everything -> soap/   (resumable; just rerun)
                                        # outputs: <split>-soap.jsonl
  python3 soap.py status                # progress counts for soap/
  python3 soap.py check [--dir soap]    # verify outputs against inputs -> <dir>/check_report.md

  --split corpus|dev|test [...]         # any command: limit to those splits, e.g. --split dev test (default: all)
          corpus_nota|dev_nota|test_nota  # opt-in: reads <split>.jsonl, writes <split>-soap.jsonl with the
                                        # base split's prompt/fields, e.g. --split dev_nota test_nota corpus_nota
  --data DIR                            # any command: release folder to read/write (default: data/release-v2)

Unattended: runs N calls at once, skips records already done, and on a usage
limit pauses all workers, waits, and continues by itself. Ctrl-C stops cleanly
after in-flight calls finish (press twice to quit immediately).
"""
import argparse, csv, fcntl, json, os, random, re, signal, subprocess, sys, threading, time, traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO_ROOT, "data", "release-v2")  # inputs + outputs; override with --data
MODEL = "claude-sonnet-4-6"  # pinned per SOAP_CONVERSION.md
EFFORT = "low"
SPLITS = ["dev", "test", "corpus"]  # dev/test first: needed soonest
NOTA_SPLITS = [f"{s}_nota" for s in SPLITS]  # opt-in via --split; same prompt/fields as their base split
ALL_SPLITS = SPLITS + NOTA_SPLITS
KEYS = {"corpus": ["subjective", "objective", "assessment", "plan"],
        "dev": ["subjective", "objective"], "test": ["subjective", "objective"]}

CORPUS_PROMPT = """\
Convert the following patient record into a SOAP note.
Return a JSON object with exactly four keys:
  "subjective"  — what the patient reports: chief complaints, symptoms,
                  and relevant history as stated in the record
  "objective"   — measurable or observed findings: vital signs, lab values,
                  exam findings, and relevant past diagnoses
  "assessment"  — the diagnosis, copied verbatim from the DIAGNOSIS section
  "plan"        — the treatment, as a list of items from the TREATMENT section

Rules:
  - Restate only what the record says. Do not infer, expand, or add information.
  - Do not mention the patient's name or any identifying detail.
  - Output valid JSON only, no prose outside the JSON object.

RECORD:
{text}
"""

DEV_TEST_PROMPT = """\
Convert the following patient record into a partial SOAP note.
Return a JSON object with exactly two keys:
  "subjective"  — what the patient reports: chief complaints, symptoms,
                  and relevant history as stated in the record
  "objective"   — measurable or observed findings: vital signs, lab values,
                  exam findings, and relevant past diagnoses

Rules:
  - Restate only what the record says. Do not infer, expand, or add information.
  - Do NOT name, suggest, hint at, or rank any diagnosis in either field.
  - Do not mention the patient's name or any identifying detail.
  - Output valid JSON only, no prose outside the JSON object.

RECORD:
{text}
"""

PROMPTS = {"corpus": CORPUS_PROMPT, "dev": DEV_TEST_PROMPT, "test": DEV_TEST_PROMPT}

MAX_FORMAT_RETRIES = 2       # extra attempts after a bad reply or unrecognised CLI error
MAX_UNCOUNTED_RETRIES = 10   # per record: retries after transient / circuit-breaker pauses
LIMIT_WAIT_DEFAULT = 30 * 60 # usage limit with no reset time in the message
TRANSIENT_WAIT = 60          # 429 / 5xx / overloaded / timeouts
CIRCUIT_BREAKER = 5          # consecutive CLI errors (any worker) before a global pause
MAX_BLIND_PAUSES = 3         # breaker pauses in a row with no success -> stop the run
FAILURE_STREAK_ABORT = 25    # records in a row sent to failures -> stop the run
CALL_TIMEOUT = 600
USAGE_KEYS = ["input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens"]

LIMIT_RE = re.compile(r"usage limit|limit reached|hit your limit|limit will reset|\bresets\b", re.I)
TRANSIENT_RE = re.compile(r"\b429\b|\b5\d\d\b|overloaded|rate.?limit|timed? ?out|timeout|"
                          r"ECONNRESET|ETIMEDOUT|ENOTFOUND|socket hang up|network", re.I)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


# ---------------------------------------------------------------- helpers

def load(split):
    with open(os.path.join(ROOT, f"{split}.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


def out_path(out_dir, split):
    return os.path.join(out_dir, f"{split}-soap.jsonl")


def base(split):
    """dev_nota -> dev: *_nota splits reuse their base split's prompt and fields."""
    return split.removesuffix("_nota")


def fields(split):
    return KEYS[base(split)]


def read_prompt(split):
    return PROMPTS[base(split)]


def parse_reply(text, keys):
    if not isinstance(text, str):
        raise ValueError("reply is not text")
    t = text.strip()
    m = FENCE_RE.search(t)
    if m:
        t = m.group(1).strip()
    d = json.loads(t)
    if not isinstance(d, dict) or sorted(d) != sorted(keys):
        raise ValueError(f"wrong keys: {list(d) if isinstance(d, dict) else type(d).__name__}")
    for k in keys:
        if k == "plan":
            # [] is valid: 40% of corpus records have TREATMENT "none recorded"
            if not isinstance(d[k], list) or not all(isinstance(x, str) and x.strip() for x in d[k]):
                raise ValueError("plan must be a list of non-empty strings")
        elif not isinstance(d[k], str) or not d[k].strip():
            raise ValueError(f"empty or non-string field: {k}")
    return d


def section_items(text, name):
    """Bullet lines of a record section, without the leading '- '."""
    m = re.search(rf"^{name}\n(.*?)(?:\n\n|\Z)", text, re.M | re.S)
    return [l[2:].strip() for l in m.group(1).splitlines() if l.startswith("- ")] if m else []


def field_text(v):
    return " ".join(v) if isinstance(v, list) else str(v)


def check_model(cli):
    """The record must have been served by MODEL and nothing else."""
    used = cli.get("modelUsage")
    if not isinstance(used, dict) or set(used) != {MODEL}:
        raise ValueError(f"served by {list(used) if isinstance(used, dict) else used!r}, expected {MODEL}")


def limit_wait_seconds(msg, now=None):
    now = time.time() if now is None else now
    m = re.search(r"\|(\d{10})", msg)  # "...limit reached|<epoch>"
    if m:
        return max(60, int(m.group(1)) - now + 60)
    # "...resets 9:50am" / "resets at 10am" / "resets 21:50" (local time; next occurrence)
    m = re.search(r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", msg, re.I)
    if m:
        h, mins, ap = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
        if ap == "pm" and h != 12:
            h += 12
        elif ap == "am" and h == 12:
            h = 0
        if h < 24 and mins < 60:
            t = datetime.fromtimestamp(now).replace(hour=h, minute=mins, second=0, microsecond=0).timestamp()
            if t <= now:
                t += 24 * 3600
            return max(60, t - now + 60)
    return LIMIT_WAIT_DEFAULT


def call_claude(prompt, text):
    cmd = ["claude", "-p", "--model", MODEL, "--effort", EFFORT,
           "--system-prompt", "", "--tools", "", "--strict-mcp-config",
           "--setting-sources", "", "--no-session-persistence", "--output-format", "json"]
    # Own session: Ctrl-C on the script doesn't kill in-flight calls, so they finish and get saved.
    p = subprocess.run(cmd, input=prompt.replace("{text}", text), capture_output=True, text=True,
                       timeout=CALL_TIMEOUT, start_new_session=True)
    try:
        d = json.loads(p.stdout)
    except Exception:
        return None, (p.stdout + p.stderr).strip()[:500] or f"no output (exit code {p.returncode})"
    if not isinstance(d, dict):
        return None, f"unexpected CLI output: {str(d)[:200]}"
    if d.get("is_error"):
        return None, str(d.get("result", ""))[:500] or "is_error with no message"
    return d, None


# ---------------------------------------------------------------- run state

class Run:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.abort_reason = None
        self.resume_at = 0.0
        self.paused_total = 0.0
        self.ok = self.fail = 0
        self.err_streak = 0
        self.blind_pauses = 0
        self.fail_streak = 0
        self.start = time.time()
        self.fail_path = os.path.join(out_dir, "failures.jsonl")
        self.usage = Counter()  # summed over every call that returned usage, incl. bad replies
        self.calls = 0

    def add_usage(self, usage):
        if not isinstance(usage, dict):
            return
        with self.lock:
            self.calls += 1
            for k in USAGE_KEYS:
                if isinstance(usage.get(k), int):
                    self.usage[k] += usage[k]

    def usage_line(self):
        """Console only (not written to progress.log)."""
        with self.lock:
            n, u = self.calls, dict(self.usage)
        if not n:
            return "usage: no calls reported usage yet"
        return (f"usage over {n} calls: " + ", ".join(f"{k} {u.get(k, 0):,} (avg {u.get(k, 0) // n:,})"
                                                      for k in USAGE_KEYS))

    def log(self, msg):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(os.path.join(self.out_dir, "progress.log"), "a") as f:
            f.write(line + "\n")

    def abort(self, reason):
        with self.lock:
            if self.abort_reason is None:
                self.abort_reason = reason
                self.log(f"STOPPING RUN: {reason}")
        self.stop.set()

    def wait_ready(self):
        """Sleep through any global pause. False if the run is stopping."""
        while not self.stop.is_set():
            w = self.resume_at - time.time()
            if w <= 0:
                return True
            self.stop.wait(min(w, 5))
        return False

    def pause(self, kind, detail, streak):
        with self.lock:
            now = time.time()
            if self.resume_at > now:
                return  # already paused by another worker
            if kind == "limit":
                secs = limit_wait_seconds(detail)
            elif kind == "transient":
                secs = TRANSIENT_WAIT
            else:  # circuit breaker on unrecognised errors
                secs = LIMIT_WAIT_DEFAULT
                self.blind_pauses += 1
            self.resume_at = now + secs
            self.paused_total += secs
            self.log(f"Pausing all workers until {datetime.fromtimestamp(self.resume_at):%H:%M} "
                     f"({kind}, error streak {streak}): {detail[:150]}")
        if kind == "breaker" and self.blind_pauses > MAX_BLIND_PAUSES:
            self.abort(f"{self.blind_pauses} pauses in a row on unrecognised errors with no success "
                       f"(last: {detail[:150]}). Fix the cause, then rerun.")

    def note_error(self):
        with self.lock:
            self.err_streak += 1
            return self.err_streak

    def write_result(self, rec, split, fields):
        row = {"id": rec["id"], **fields}
        with self.lock:
            with open(out_path(self.out_dir, split), "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.ok += 1
            self.err_streak = self.blind_pauses = self.fail_streak = 0

    def write_failure(self, rec, split, reason):
        with self.lock:
            with open(self.fail_path, "a") as f:
                f.write(json.dumps({"id": rec["id"], "split": split, "reason": reason}) + "\n")
            self.fail += 1
            self.fail_streak += 1
            streak = self.fail_streak
            self.log(f"FAILED {rec['id']}: {reason[:150]}")
        if streak >= FAILURE_STREAK_ABORT:
            self.abort(f"{streak} records in a row failed (last: {reason[:150]}). "
                       "Something systematic is wrong; fix it, then rerun.")


def process(st, rec, split, prompt):
    counted = uncounted = 0
    while True:
        if not st.wait_ready():
            return  # stopping: record stays undone and is picked up on the next run
        try:
            d, err = call_claude(prompt, rec["text"])
        except subprocess.TimeoutExpired:
            d, err = None, f"timeout: no reply within {CALL_TIMEOUT}s"

        if err is None:
            st.add_usage(d.get("usage"))
            try:
                fields = parse_reply(d.get("result"), fields(split))
                check_model(d)
            except Exception as e:
                text = str(d.get("result", ""))
                # A usage-limit notice can come back as an ordinary (non-error) reply.
                if len(text) < 500 and not text.lstrip().startswith(("{", "`")) and LIMIT_RE.search(text):
                    kind, detail = "limit", text
                else:
                    kind, detail = "bad", f"bad reply: {e}"
            else:
                st.write_result(rec, split, fields)
                return
        elif LIMIT_RE.search(err):
            kind, detail = "limit", err
        elif TRANSIENT_RE.search(err):
            kind, detail = "transient", err
        else:
            kind, detail = "error", f"cli error: {err}"

        streak = st.note_error() if kind != "bad" else 0
        if kind == "error" and streak >= CIRCUIT_BREAKER:
            kind = "breaker"

        if kind == "limit":
            st.pause(kind, detail, streak)
            continue  # real limits always clear; never fail a record because of one
        if kind in ("transient", "breaker"):
            st.pause(kind, detail, streak)
            uncounted += 1
            if uncounted <= MAX_UNCOUNTED_RETRIES:
                continue
            st.write_failure(rec, split, f"gave up after {MAX_UNCOUNTED_RETRIES} pauses: {detail}")
            return

        counted += 1
        if counted > MAX_FORMAT_RETRIES:
            st.write_failure(rec, split, detail)
            return
        st.stop.wait(5 * counted)


def safe_process(st, rec, split, prompt):
    try:
        process(st, rec, split, prompt)
    except Exception as e:
        st.log(traceback.format_exc())
        try:
            st.write_failure(rec, split, f"internal error: {type(e).__name__}: {e}")
        except Exception:
            st.abort(f"cannot write failures file: {e}")


# ---------------------------------------------------------------- startup safety

def acquire_lock(out_dir):
    fd = open(os.path.join(out_dir, ".run.lock"), "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"Another soap.py run is already using {out_dir}. Stop it first.")
    return fd  # keep open for the life of the process


def repair_outputs(st, valid_ids):
    """Drop truncated/invalid/duplicate lines (e.g. from a kill mid-write) before resuming."""
    done = {}
    for split in valid_ids:
        path = out_path(st.out_dir, split)
        ids = set()
        if not os.path.exists(path):
            done[split] = ids
            continue
        with open(path) as f:
            raw = f.read()
        keep, dropped = [], []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                ok = (isinstance(row, dict) and row.get("id") in valid_ids[split]
                      and row["id"] not in ids and sorted(row) == sorted(["id"] + fields(split)))
            except Exception:
                ok = False
            if ok:
                ids.add(row["id"])
                keep.append(line)
            else:
                dropped.append(line)
        if dropped or (raw and not raw.endswith("\n")):
            with open(path + ".dropped", "a") as f:
                f.writelines(l + "\n" for l in dropped)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.writelines(l + "\n" for l in keep)
            os.replace(tmp, path)
            st.log(f"{split}: removed {len(dropped)} invalid/duplicate line(s) "
                   f"(saved to {os.path.basename(path)}.dropped); they will be redone")
        done[split] = ids
    return done


def sort_outputs(st, records_by_split):
    """Rewrite each output file in input-file order (workers append in completion order)."""
    for split, recs in records_by_split.items():
        path = out_path(st.out_dir, split)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            by_id = {json.loads(l)["id"]: l for l in f if l.strip()}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.writelines(by_id[r["id"]] for r in recs if r["id"] in by_id)
        os.replace(tmp, path)


def check_manifest(st, records_by_split):
    path = os.path.join(st.out_dir, "manifest.json")
    current = {"model": MODEL, "effort": EFFORT, "fields": KEYS,
               "prompts": {s: read_prompt(s) for s in SPLITS}}
    now = datetime.now().isoformat(timespec="seconds")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        changed = [k for k in current if old.get(k) != current[k]]
        if changed:
            sys.exit(f"Settings changed since this folder was started ({', '.join(changed)}). "
                     f"Resuming would mix versions. Use a new output folder or restore the old settings.")
        manifest = {**old, "runs": old.get("runs", [old.get("started")]) + [now],
                    "input_counts": {**old.get("input_counts", {}),
                                     **{s: len(records_by_split[s]) for s in records_by_split}}}
    else:
        manifest = {"tool": "claude -p (Claude Code, subscription)", **current,
                    "temperature": "not settable via Claude Code (spec: 0)",
                    "max_output_tokens": "not settable via Claude Code (spec: 1024)",
                    "started": now, "runs": [now],
                    "input_counts": {s: len(records_by_split[s]) for s in records_by_split}}
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------- commands

def pilot_sample(splits=SPLITS):
    """2 per condition per split, plus the 3 longest and 1 shortest per split."""
    rng = random.Random(0)
    picks = {}
    for split in SPLITS + [s for s in splits if s not in SPLITS]:  # base splits first keeps their picks stable
        recs = load(split)
        by_cond = {}
        for r in recs:
            by_cond.setdefault(r["id"].rsplit("_", 1)[0], []).append(r)
        chosen = {}
        for cond, rs in sorted(by_cond.items()):
            for r in rng.sample(rs, min(2, len(rs))):
                chosen[r["id"]] = r
        by_len = sorted(recs, key=lambda r: len(r["text"]))
        for r in by_len[-3:] + by_len[:1]:
            chosen[r["id"]] = r
        picks[split] = [r for r in recs if r["id"] in chosen]  # input-file order
    return {s: picks[s] for s in splits}  # sample all splits so the picks don't depend on --split


def run(out_dir, records_by_split, workers):
    os.makedirs(out_dir, exist_ok=True)
    lock_fd = acquire_lock(out_dir)
    st = Run(out_dir)
    check_manifest(st, records_by_split)
    splits = [s for s in ALL_SPLITS if s in records_by_split]
    done = repair_outputs(st, {s: {r["id"] for r in records_by_split[s]} for s in splits})
    if os.path.exists(st.fail_path):  # previous failures of these splits get another try
        with open(st.fail_path) as f:
            lines = f.readlines()
        retry = [l for l in lines if json.loads(l).get("split") in splits]
        if retry:
            with open(st.fail_path + f".{int(time.time())}.old", "w") as f:
                f.writelines(retry)
            with open(st.fail_path + ".tmp", "w") as f:
                f.writelines(l for l in lines if l not in retry)
            os.replace(st.fail_path + ".tmp", st.fail_path)

    jobs = []
    for split in splits:
        todo = [r for r in records_by_split[split] if r["id"] not in done[split]]
        st.log(f"{split}: {len(done[split])} already done, {len(todo)} to go")
        prompt = read_prompt(split)
        jobs += [(r, split, prompt) for r in todo]
    n = len(jobs)
    st.log(f"Starting {n} conversions with {workers} parallel workers (model {MODEL})")

    def on_signal(signum, frame):
        if st.stop.is_set():
            os._exit(130)
        st.log("Stopping: finishing in-flight calls, then exiting (press Ctrl-C again to quit now)")
        st.stop.set()
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    finished = threading.Event()

    def reporter():
        while not finished.wait(60):
            active = max(1.0, time.time() - st.start - st.paused_total)
            k = st.ok + st.fail
            rate = k / active
            eta = (n - k) / rate / 3600 if rate else float("inf")
            st.log(f"progress {k}/{n} (ok {st.ok}, failed {st.fail}), ~{rate*3600:.0f}/hr while running, "
                   f"~{eta:.1f}h of processing left (plus any limit pauses)")
            print(st.usage_line(), flush=True)
    threading.Thread(target=reporter, daemon=True).start()

    with ThreadPoolExecutor(workers) as ex:
        futures = [ex.submit(safe_process, st, *j) for j in jobs]
        for fu in futures:
            while not fu.done():
                time.sleep(0.5)  # short sleeps keep Ctrl-C responsive
    finished.set()
    print(st.usage_line(), flush=True)
    sort_outputs(st, records_by_split)

    left = n - st.ok - st.fail
    if st.abort_reason:
        st.log(f"Run stopped early: {st.abort_reason}")
    elif st.stop.is_set():
        st.log("Run stopped by user.")
    st.log(f"ok {st.ok}, failed {st.fail}, not attempted {left}. "
           + ("Rerun the same command to continue/retry." if (st.fail or left) else "All records done."))
    lock_fd.close()
    return 1 if (st.fail or left) else 0


def status(out_dir, splits=SPLITS):
    for split in splits:
        total = len(load(split))
        path = out_path(out_dir, split)
        done = sum(1 for _ in open(path)) if os.path.exists(path) else 0
        print(f"{split:7s} {done:5d} / {total}")
    fp = os.path.join(out_dir, "failures.jsonl")
    if os.path.exists(fp):
        print("failures:", sum(1 for l in open(fp) if json.loads(l).get("split") in splits))


# Terms that name each condition; used to spot diagnoses the model added.
COND_TERMS = {
    "copd": r"copd|chronic obstructive|emphysema|chronic bronchitis",
    "hypertension": r"hypertension|hypertensive|high blood pressure",
    "strep_throat": r"strep|streptococc",
    "viral_pharyngitis": r"pharyngitis",
    "bacterial_sinusitis": r"sinusitis",
    "cystitis": r"cystitis|urinary tract infection|\buti\b",
    "heart_failure": r"heart failure|\bchf\b|\bhfref\b|\bhfpef\b|cardiomyopathy",
}
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def check(out_dir, splits=SPLITS):
    cond = {r["example_id"]: r["condition"] for r in csv.DictReader(open(os.path.join(ROOT, "index.csv")))}
    hard, review, lines = [], [], [f"# SOAP check report: `{os.path.basename(out_dir)}/`", "",
                                   f"Generated {datetime.now():%Y-%m-%d %H:%M}", ""]
    fp = os.path.join(out_dir, "failures.jsonl")
    n_fail = sum(1 for l in open(fp) if json.loads(l).get("split") in splits) if os.path.exists(fp) else 0
    for split in splits:
        src = {r["id"]: r["text"] for r in load(split)}
        path = out_path(out_dir, split)
        rows, invalid, seen, dups = [], 0, set(), set()
        if os.path.exists(path):
            for line in open(path):
                try:
                    row = json.loads(line)
                    row["id"]
                except Exception:
                    invalid += 1
                    continue
                (dups if row["id"] in seen else seen).add(row["id"])
                rows.append(row)
        unknown = seen - set(src)
        pos = {i: n for n, i in enumerate(src)}
        order = [pos[r["id"]] for r in rows if r["id"] in pos]
        in_order = order == sorted(order)
        missing = set(src) - seen
        bad_keys = [r["id"] for r in rows if sorted(r) != sorted(["id"] + fields(split))]
        empty = [r["id"] for r in rows if any(not field_text(r.get(k, "")).strip()
                                              for k in fields(split) if k != "plan")]

        added, wrong_assess, unsupported_nums, not_verbatim, plan_diff = [], [], [], [], []
        none_forms = Counter()
        for r in rows:
            if r["id"] not in src or r["id"] in bad_keys:
                continue
            s_low = src[r["id"]].lower()
            so = (r["subjective"] + " " + r["objective"]).lower()
            new = sorted({m.group(0) for pat in COND_TERMS.values()
                          for m in re.finditer(pat, so) if m.group(0) not in s_low})
            if new:
                added.append((r["id"], new))
            if base(split) == "corpus":
                c = cond.get(r["id"])
                if c and not re.search(COND_TERMS[c], r["assessment"].lower()):
                    wrong_assess.append((r["id"], c, r["assessment"][:120]))
                diag = section_items(src[r["id"]], "DIAGNOSIS")
                if r["assessment"].strip() not in diag:
                    not_verbatim.append((r["id"], r["assessment"][:120], "; ".join(diag)))
                treat = section_items(src[r["id"]], "TREATMENT")
                none_forms[json.dumps(r["plan"])] += treat == ["none recorded"]
                if [x.strip() for x in r["plan"]] != treat and not (treat == ["none recorded"] and r["plan"] == []):
                    plan_diff.append((r["id"], r["plan"], treat))
            out_text = " ".join(field_text(r[k]) for k in fields(split))
            nums = sorted({x for x in NUM_RE.findall(out_text) if x not in src[r["id"]]})
            if nums:
                unsupported_nums.append((r["id"], nums[:8]))

        lines += [f"## {split}", "",
                  f"- output records: {len(seen)} of {len(src)} inputs",
                  f"- missing: {len(missing)} · duplicates: {len(dups)} · unknown ids: {len(unknown)} · "
                  f"unreadable lines: {invalid}",
                  f"- wrong keys: {len(bad_keys)} · empty fields: {len(empty)} · "
                  f"in input order: {'yes' if in_order else 'NO'}",
                  f"- condition terms added in S/O (not in source): {len(added)}",
                  *([f"- corpus assessment doesn't name the labelled condition: {len(wrong_assess)}",
                     f"- assessment not verbatim from DIAGNOSIS: {len(not_verbatim)}",
                     f"- plan not exactly the TREATMENT items: {len(plan_diff)}",
                     "- plan for TREATMENT 'none recorded': "
                     + (", ".join(f"`{k}` ×{v}" for k, v in none_forms.items() if v) or "n/a")]
                    if base(split) == "corpus" else []),
                  f"- records with numbers not found in source: {len(unsupported_nums)}", ""]
        for label, items in [("Missing ids", sorted(missing)), ("Duplicate ids", sorted(dups)),
                             ("Unknown ids", sorted(unknown)), ("Wrong keys", bad_keys), ("Empty fields", empty)]:
            if items:
                hard.append(f"{split}: {len(items)} {label.lower()}")
                lines += [f"**{label}** (first 20): " + ", ".join(items[:20]), ""]
        if invalid:
            hard.append(f"{split}: {invalid} unreadable lines")
        if not in_order:
            hard.append(f"{split}: not in input order (rerun to re-sort)")
        if added:
            review.append(f"{split}: {len(added)} records with added condition terms")
            lines += ["**Condition terms added in S/O** (first 30):", ""] + \
                     [f"- `{i}`: {', '.join(t)}" for i, t in added[:30]] + [""]
        if wrong_assess:
            review.append(f"corpus: {len(wrong_assess)} assessments not naming the labelled condition")
            lines += ["**Corpus assessment vs label** (first 30):", ""] + \
                     [f"- `{i}` ({c}): {a}" for i, c, a in wrong_assess[:30]] + [""]
        if not_verbatim:
            review.append(f"corpus: {len(not_verbatim)} assessments not verbatim from DIAGNOSIS")
            lines += ["**Assessment vs DIAGNOSIS line** (first 30):", ""] + \
                     [f"- `{i}`: got `{a}`, source `{d}`" for i, a, d in not_verbatim[:30]] + [""]
        if plan_diff:
            review.append(f"corpus: {len(plan_diff)} plans differing from TREATMENT")
            lines += ["**Plan vs TREATMENT lines** (first 30):", ""] + \
                     [f"- `{i}`: got {json.dumps(p, ensure_ascii=False)}, source {json.dumps(t, ensure_ascii=False)}"
                      for i, p, t in plan_diff[:30]] + [""]
        if unsupported_nums:
            review.append(f"{split}: {len(unsupported_nums)} records with numbers not in source")
            lines += ["**Numbers not found in source** (first 30; often harmless, e.g. unit conversions):", ""] + \
                     [f"- `{i}`: {', '.join(x)}" for i, x in unsupported_nums[:30]] + [""]
    if n_fail:
        hard.append(f"{n_fail} records in failures.jsonl (rerun to retry)")
    summary = ["## Summary", "",
               "**Must fix:** " + ("; ".join(hard) if hard else "none"), "",
               "**To review:** " + ("; ".join(review) if review else "none"), ""]
    lines[4:4] = summary
    report = os.path.join(out_dir, "check_report.md")
    with open(report, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(summary))
    print(f"Full report: {report}")
    return 1 if hard else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "run", "status", "check"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dir", default="soap", help="output folder for status/check")
    ap.add_argument("--split", nargs="+", choices=ALL_SPLITS,
                    help="limit to these splits (default: corpus dev test; *_nota splits only when named)")
    ap.add_argument("--data", default=ROOT, help="release folder with <split>.jsonl, index.csv (default: data/release-v2)")
    a = ap.parse_args()
    ROOT = os.path.abspath(a.data)
    splits = [s for s in ALL_SPLITS if s in a.split] if a.split else SPLITS
    if a.cmd == "pilot":
        sys.exit(run(os.path.join(ROOT, "soap_pilot"), pilot_sample(splits), a.workers))
    elif a.cmd == "run":
        sys.exit(run(os.path.join(ROOT, "soap"), {s: load(s) for s in splits}, a.workers))
    elif a.cmd == "status":
        status(os.path.join(ROOT, a.dir), splits)
    else:
        sys.exit(check(os.path.join(ROOT, a.dir), splits))
