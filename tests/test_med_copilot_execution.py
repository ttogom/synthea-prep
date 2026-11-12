"""Offline tests for durable independent jobs and bounded concurrent execution."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from threading import Event, Lock
import time
import unittest
from unittest.mock import patch

# Direct execution puts tests/ on sys.path; the project imports below need its root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.experiment_runtime.execution import JobStore, bounded_map, write_results
from scripts.med_copilot_run_file import RunMetadataError


class JobStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "jobs.sqlite3"
        self.key = ("generation", "patient-1", "llm_only", "local")
        self.identity = {"config_sha256": "abc", "patients": ["patient-1"]}

    def test_new_store_has_full_synchronous_wal_and_no_jobs(self):
        with JobStore(self.path, identity=self.identity) as store:
            self.assertEqual(store._connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            self.assertEqual(store._connection.execute("PRAGMA synchronous").fetchone()[0], 2)
            self.assertEqual(store.entries(), [])
            self.assertIsNone(store.load(self.key))
            self.assertIsNone(store.state(self.key))

    def test_received_reply_survives_reopen_before_parsing(self):
        reply = {"_generation_reply_complete": True, "assessment_plan_raw": '{"predictions":[]}',
                 "model_responses": [{"token_usage": {"total_tokens": 123}}]}
        with JobStore(self.path, identity=self.identity) as store:
            self.assertEqual(store.save(self.key, reply, status="received"), 1)
        with JobStore(self.path, identity=self.identity) as resumed:
            self.assertEqual(resumed.load(self.key), reply)
            self.assertEqual(resumed.state(self.key), {"status": "received", "revision": 1, "payload": reply})
            parsed = {"status": "completed", "predictions": [{"condition": "x", "plan": ["y"]}]}
            self.assertEqual(resumed.save(self.key, parsed, status="completed"), 2)
        with JobStore(self.path) as resumed:
            self.assertEqual(resumed.load(self.key), parsed)
            self.assertEqual(resumed.state(self.key)["revision"], 2)

    def test_detached_reads_and_independent_variant_model_identity(self):
        with JobStore(self.path) as store:
            for variant in (None, "llm_only", "guidelines"):
                for model in (None, "a", "b"):
                    store.save(("generation", "patient", variant, model), {"items": [variant, model]})
            loaded = store.load(("generation", "patient", None, None))
            loaded["items"].append("changed")
            self.assertEqual(store.load(("generation", "patient", None, None)), {"items": [None, None]})
            self.assertEqual(len(store.entries("generation")), 9)
            self.assertEqual(store.entries("missing"), [])
            self.assertEqual(store.state(("generation", "patient", "guidelines", "a"))["revision"], 1)

    def test_concurrent_checkpoints_do_not_mix_patient_payloads(self):
        with JobStore(self.path) as store:
            def save(index):
                key = ("generation", f"p{index}", "both", "model")
                for attempt in range(3):
                    store.save(key, {"patient": index, "attempt": attempt}, status="completed")
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(save, range(24)))
            entries = store.entries()
            self.assertEqual(len(entries), 24)
            for key, state in entries:
                self.assertEqual(state["revision"], 3)
                self.assertEqual(state["payload"], {"patient": int(key[1][1:]), "attempt": 2})

    def test_exception_scope_still_closes_and_keeps_completed_job(self):
        store = JobStore(self.path)
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with store:
                store.save(self.key, {"saved": True})
                raise RuntimeError("stop")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            store.load(self.key)
        store.close()
        with JobStore(self.path) as reopened:
            self.assertEqual(reopened.load(self.key), {"saved": True})

    def test_mismatched_identity_is_rejected_without_modifying_existing_database(self):
        with JobStore(self.path, identity=self.identity) as store:
            store.save(self.key, {"kept": True})
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "identity"):
            JobStore(self.path, identity={"config_sha256": "changed"})
        self.assertEqual(self.path.read_bytes(), before)
        with JobStore(self.path, identity=self.identity) as store:
            self.assertEqual(store.load(self.key), {"kept": True})

    def test_invalid_payload_never_replaces_a_completed_reply(self):
        with JobStore(self.path) as store:
            store.save(self.key, {"kept": True})
            for payload in (None, [], {"bad": float("nan")}, {"bad": float("inf")}, {"bad": object()}):
                with self.subTest(payload=type(payload).__name__), self.assertRaises((TypeError, ValueError)):
                    store.save(self.key, payload)
                self.assertEqual(store.load(self.key), {"kept": True})
                self.assertEqual(store.state(self.key)["revision"], 1)

    def test_invalid_keys_status_and_identity_are_rejected(self):
        with JobStore(self.path) as store:
            for key in ([], ("generation", "p"), ("", "p", None, None),
                        ("generation", None, None, None), ("generation", "p", "", None),
                        ("generation", "p", None, 2)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    store.save(key, {})
            for status in (None, "", "  ", 1):
                with self.subTest(status=status), self.assertRaises(ValueError):
                    store.save(self.key, {}, status=status)
            for stage in ("", 1, []):
                with self.subTest(stage=stage), self.assertRaises(ValueError):
                    store.entries(stage)
        for identity in ([], "text", 1):
            with self.assertRaises(TypeError):
                JobStore(self.root / "other.sqlite3", identity=identity)

    def test_redaction_happens_before_checkpoint_is_published(self):
        with JobStore(self.path, redact=lambda text: text.replace("private-token", "[REDACTED]")) as store:
            store.save(self.key, {"error": "endpoint echoed private-token", "nested": ["private-token"]})
            self.assertEqual(store.load(self.key), {"error": "endpoint echoed [REDACTED]", "nested": ["[REDACTED]"]})
        self.assertNotIn(b"private-token", self.path.read_bytes())

    def test_broken_redaction_cannot_ack_or_replace_a_checkpoint(self):
        with JobStore(self.path) as store:
            store.save(self.key, {"kept": True})
            store._redact = lambda text: "not JSON"
            with self.assertRaises(ValueError):
                store.save(self.key, {"new": True})
            self.assertEqual(store.load(self.key), {"kept": True})
        with self.assertRaises(TypeError):
            JobStore(self.path, redact="invalid")

    def test_failed_commit_rolls_back_and_raises_fatal_checkpoint_error(self):
        with JobStore(self.path) as store:
            store.save(self.key, {"kept": True})
            real = store._connection
            class FailCommit:
                def __getattr__(self, name):
                    return getattr(real, name)
                def execute(self, sql, *args):
                    if sql == "COMMIT":
                        raise sqlite3.OperationalError("disk full")
                    return real.execute(sql, *args)
            store._connection = FailCommit()
            with self.assertRaises(RunMetadataError):
                store.save(self.key, {"new": True})
            self.assertEqual(store.load(self.key), {"kept": True})
            self.assertEqual(store.state(self.key)["revision"], 1)

    def test_malformed_saved_json_is_rejected_before_reopening_for_writes(self):
        with JobStore(self.path) as store:
            store.save(self.key, {"kept": True})
        for payload in ('{"same":1,"same":2}', '{"number":NaN}', '{"number":1e999}', "[]"):
            with sqlite3.connect(self.path) as connection:
                connection.execute("UPDATE jobs SET payload=?", (payload,))
            before = self.path.read_bytes()
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                JobStore(self.path)
            self.assertEqual(self.path.read_bytes(), before)

    def test_corrupt_database_is_rejected(self):
        self.path.write_bytes(b"not a sqlite database")
        with self.assertRaisesRegex(ValueError, "Invalid job store"):
            JobStore(self.path)
        self.assertEqual(self.path.read_bytes(), b"not a sqlite database")

    def test_process_exit_without_close_keeps_durable_reply(self):
        code = ("from scripts.experiment_runtime.execution import JobStore\n"
                "import os,sys\n"
                "store=JobStore(sys.argv[1])\n"
                "store.save(('generation','p','v','m'), {'raw':'completed reply'}, status='received')\n"
                "os._exit(0)\n")
        result = subprocess.run([sys.executable, "-c", code, str(self.path)], cwd=ROOT, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        with JobStore(self.path) as store:
            self.assertEqual(store.load(("generation", "p", "v", "m")), {"raw": "completed reply"})


class BoundedMapTests(unittest.TestCase):
    def test_independent_calls_overlap_and_keep_output_identity(self):
        active = maximum = 0
        lock = Lock()
        def work(item):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(active, maximum)
            time.sleep(0.015)
            with lock:
                active -= 1
            return {"patient": item}
        results = list(bounded_map(work, range(12), workers=3))
        self.assertEqual(maximum, 3)
        self.assertEqual({item: result for item, result, _ in results}, {n: {"patient": n} for n in range(12)})
        self.assertTrue(all(seconds >= 0.01 for _, _, seconds in results))

    def test_submission_is_bounded_even_with_an_unbounded_source(self):
        produced = []
        release = Event()
        def source():
            for item in range(1000):
                produced.append(item)
                yield item
        def work(item):
            release.wait(1)
            return item
        iterator = bounded_map(work, source(), workers=2)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(next, iterator)
            deadline = time.monotonic() + 1
            while len(produced) < 4 and time.monotonic() < deadline:
                time.sleep(0.002)
            self.assertEqual(len(produced), 4)
            release.set()
            future.result()
        iterator.close()
        self.assertEqual(len(produced), 4)

    def test_contextvars_are_isolated_and_inherited_per_job(self):
        context = ContextVar("test_job", default="unset")
        context.set("parent")
        def work(item):
            before = context.get()
            context.set(item)
            return before
        results = list(bounded_map(work, range(8), workers=2))
        self.assertEqual([result for _, result, _ in results], ["parent"] * 8)
        self.assertEqual(context.get(), "parent")

    def test_failure_stops_submission_and_waits_for_running_checkpoint(self):
        saved = Event()
        def work(item):
            if item == 0:
                time.sleep(0.01)
                raise RuntimeError("request failed")
            time.sleep(0.03)
            saved.set()
            return item
        with self.assertRaisesRegex(RuntimeError, "request failed"):
            list(bounded_map(work, range(100), workers=2))
        self.assertTrue(saved.is_set())

    def test_interrupts_propagate(self):
        def work(item):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            list(bounded_map(work, [1], workers=1))

    def test_empty_source_and_argument_validation(self):
        self.assertEqual(list(bounded_map(lambda item: item, [], workers=2)), [])
        for workers in (0, -1, True, "2"):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                list(bounded_map(lambda item: item, [], workers=workers))
        for limit in (0, 1, True, "4"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                list(bounded_map(lambda item: item, [], workers=2, max_pending=limit))


class ResultsExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "results.jsonl"
        self.path.write_text("original\n")

    def test_export_streams_unicode_records_in_order(self):
        records = [{"id": "a", "text": "café\u2028preserved"}, {"id": "b", "score": 0.5}]
        write_results(iter(records), self.path)
        self.assertEqual([json.loads(line) for line in self.path.read_text().split("\n") if line], records)
        self.assertEqual(list(self.path.parent.glob(".results-*.tmp")), [])

    def test_serialization_failure_preserves_existing_file_and_cleans_temporary(self):
        with self.assertRaises(ValueError):
            write_results([{"id": "a"}, {"bad": float("nan")}], self.path)
        self.assertEqual(self.path.read_text(), "original\n")
        self.assertEqual(list(self.path.parent.glob(".results-*.tmp")), [])

    def test_failed_flush_and_replace_preserve_existing_file(self):
        for target in ("scripts.experiment_runtime.execution.os.fsync", "scripts.experiment_runtime.execution.os.replace"):
            with self.subTest(target=target), patch(target, side_effect=OSError("disk full")), self.assertRaises(OSError):
                write_results([{"id": "a"}], self.path)
            self.assertEqual(self.path.read_text(), "original\n")
            self.assertEqual(list(self.path.parent.glob(".results-*.tmp")), [])

    def test_failed_directory_sync_reports_failure_with_a_complete_new_file(self):
        with patch("scripts.experiment_runtime.execution._fsync_directory", side_effect=OSError("directory flush")), \
                self.assertRaises(OSError):
            write_results([{"id": "a"}], self.path)
        self.assertEqual(json.loads(self.path.read_text()), {"id": "a"})
        self.assertEqual(list(self.path.parent.glob(".results-*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
