"""Offline tests for the standard-library run.json owner used before heavy imports.

Run directly: python tests/test_med_copilot_run_file.py
Or discover: python -m unittest discover -s tests -p 'test_*.py'
"""

import builtins
from contextlib import redirect_stderr
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
RUN_FILE_MODULE = ROOT / "scripts/med_copilot_run_file.py"
R = runpy.run_path(str(RUN_FILE_MODULE))
STARTED = R["pacific_time"](datetime(2026, 10, 17, 22, 30, 45, tzinfo=timezone.utc))


class StandardLibraryTests(unittest.TestCase):
    def test_module_imports_only_the_standard_library(self):
        original_import = builtins.__import__

        def stdlib_only(name, *args, **kwargs):
            if name.split(".")[0] not in sys.stdlib_module_names:
                raise AssertionError(f"run.json startup code must not import {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=stdlib_only):
            runpy.run_path(str(RUN_FILE_MODULE))


class RunFileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "experiments/output"

    def read(self, run):
        return json.loads((run.run_dir / "run.json").read_text(encoding="utf-8"))

    def test_begin_creates_provisional_running_run_json(self):
        run = R["RunFile"].begin(self.output, started_at=STARTED)
        self.assertEqual(run.run_dir.parent, self.output)
        self.assertTrue(run.run_dir.name.startswith("2026-10-17_153045PDT__starting-"))
        self.assertEqual(run.started_at, STARTED)
        self.assertEqual([path.name for path in run.run_dir.iterdir()], ["run.json"])
        self.assertEqual(self.read(run), {
            "run_name": run.run_dir.name, "started_at": "2026-10-17T15:30:45-07:00",
            "finished_at": None, "status": "running", "branch": None, "input": None,
            "git_commit": None, "reference_data": None, "errors": [],
        })

    def test_move_names_the_run_and_keeps_its_errors(self):
        run = R["RunFile"].begin(self.output, started_at=STARTED)
        provisional = run.run_dir
        R["record_error"](RuntimeError("kept"), run=run, stage="startup")
        run.move(self.output, "final-name")
        self.assertEqual(run.run_dir, self.output / "final-name")
        self.assertFalse(provisional.exists())
        metadata = self.read(run)
        self.assertEqual(metadata["run_name"], "final-name")
        self.assertEqual([error["message"] for error in metadata["errors"]], ["kept"])

    def test_move_to_another_root_removes_the_provisional_root_it_created(self):
        run = R["RunFile"].begin(self.output, started_at=STARTED)
        run.move(self.root / "elsewhere/runs", "final-name")
        self.assertFalse((self.root / "experiments").exists())
        self.assertTrue((self.root / "elsewhere/runs/final-name/run.json").is_file())

    def test_move_never_replaces_an_existing_run(self):
        (self.output / "taken").mkdir(parents=True)
        run = R["RunFile"].begin(self.output, started_at=STARTED)
        with self.assertRaises(FileExistsError):
            run.move(self.output, "taken")
        self.assertTrue((run.run_dir / "run.json").is_file())
        self.assertEqual(list((self.output / "taken").iterdir()), [])

    def test_discard_removes_the_run_and_only_directories_it_created(self):
        run = R["RunFile"].begin(self.output, started_at=STARTED)
        run.move(self.output, "final-name")
        run.discard()
        self.assertFalse((self.root / "experiments").exists())
        other = self.output / "previous-run"
        other.mkdir(parents=True)
        run = R["RunFile"].begin(self.output, started_at=STARTED)
        run.discard()
        self.assertEqual(list(self.output.iterdir()), [other])

    def test_fail_records_the_error_and_finishes_failed_incomplete(self):
        run = R["RunFile"].begin(self.output, started_at=STARTED)
        try:
            raise ValueError("bad config")
        except ValueError as exc:
            run.fail(exc, stage="startup")
        metadata = self.read(run)
        self.assertEqual(metadata["status"], "failed_incomplete")
        self.assertIsNotNone(metadata["finished_at"])
        [error] = metadata["errors"]
        self.assertEqual((error["patient_id"], error["stage"], error["exception_type"], error["message"]),
                         (None, "startup", "ValueError", "bad config"))

    def test_fail_does_not_record_a_metadata_write_failure_again(self):
        run = R["RunFile"].begin(self.output, started_at=STARTED)
        run.fail(R["RunMetadataError"]("already reported"))
        metadata = self.read(run)
        self.assertEqual((metadata["status"], metadata["errors"]), ("failed_incomplete", []))

    def test_default_redaction_hides_shared_credentials(self):
        run = R["RunFile"].begin(self.output, started_at=STARTED)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-shared-SECRET"}):
            R["record_error"](RuntimeError("bad sk-shared-SECRET"), run=run)
        self.assertNotIn("SECRET", (run.run_dir / "run.json").read_text(encoding="utf-8"))

    def test_stop_reports_a_failed_final_write(self):
        run = R["RunFile"].begin(self.output, started_at=STARTED)
        stderr = io.StringIO()
        with patch.object(os, "replace", side_effect=OSError("disk full")), redirect_stderr(stderr):
            run.stop(STARTED)
        self.assertIn("Could not mark run.json as stopped", stderr.getvalue())
        self.assertEqual(self.read(run)["status"], "running")


if __name__ == "__main__":
    unittest.main()
