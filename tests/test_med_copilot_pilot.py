"""Offline runtime-pilot sampling and weighted-extrapolation regression tests."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

# Direct execution starts in tests/; load the project without model dependencies.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.med_copilot_pilot import STRATA, project_plan_runtime, project_runtime, select_runtime_pilot


def population(count=200):
    patients, labels = [], []
    for index in range(count):
        events = "2020-01-01: recorded event\n" * 20 if index % 10 == 0 else ""
        patients.append({"id": f"p{index:04}", "subjective": events + "word " * (index + 1),
                         "objective": "Objective unchanged café\u2028section"})
        labels.append({"id": f"p{index:04}", "condition": "reference", "plan": ["Action"]})
    return patients, labels


class PilotSelectionTests(unittest.TestCase):
    def test_64_patients_are_equal_disjoint_strata_and_original_objects(self):
        patients, labels = population()
        original = deepcopy(patients)
        selected, manifest = select_runtime_pilot(patients, labels)
        self.assertEqual(len(selected), 64)
        self.assertEqual(patients, original)
        self.assertEqual(len({patient["id"] for patient in selected}), 64)
        for patient in selected:
            self.assertIs(patient, next(row for row in patients if row["id"] == patient["id"]))
        self.assertEqual(manifest["selected_patient_ids"], [row["id"] for row in selected])
        self.assertEqual([patient["id"] for patient in selected],
                         [patient["id"] for patient in patients if patient in selected])
        self.assertEqual(set(manifest["strata"]), set(STRATA))
        self.assertEqual(sum(row["weight"] for row in manifest["strata"].values()), 1)
        for name, stratum in manifest["strata"].items():
            self.assertEqual(stratum["sample_count"], 16)
            self.assertTrue(all(manifest["patient_strata"][value] == name for value in stratum["selected_patient_ids"]))
        complex_ids = {value for value, name in manifest["patient_strata"].items() if name == "complex"}
        self.assertEqual(complex_ids, {patient["id"] for index, patient in enumerate(patients) if index % 10 == 0})
        self.assertEqual(manifest["strata"]["complex"]["population_count"], 20)
        self.assertEqual(manifest["strata"]["complex"]["weight"], 0.1)

    def test_48_patient_option_is_reproducible_and_labels_can_be_mapping(self):
        patients, labels = population()
        selected, manifest = select_runtime_pilot(patients, {row["id"]: row for row in labels}, count=48)
        again, repeated = select_runtime_pilot(patients, labels, count=48)
        self.assertEqual(selected, again)
        self.assertEqual(manifest, repeated)
        self.assertEqual([row["sample_count"] for row in manifest["strata"].values()], [12] * 4)

    def test_seed_changes_random_members_but_retains_forced_extremes(self):
        patients, labels = population()
        _, first = select_runtime_pilot(patients, labels, seed=42)
        _, second = select_runtime_pilot(patients, labels, seed=99)
        self.assertNotEqual(first["selected_patient_ids"], second["selected_patient_ids"])
        for name in STRATA:
            anchors = first["strata"][name]["forced_patient_ids"]
            self.assertTrue(set(anchors) <= set(second["strata"][name]["selected_patient_ids"]))
            self.assertTrue(all(first["strata"][name]["inclusion_probabilities"][value] == 1 for value in anchors))

    def test_population_order_changes_only_return_order(self):
        patients, labels = population()
        first, _ = select_runtime_pilot(patients, labels)
        second, _ = select_runtime_pilot(list(reversed(patients)), labels)
        self.assertEqual({row["id"] for row in first}, {row["id"] for row in second})
        self.assertEqual([row["id"] for row in first], list(reversed([row["id"] for row in second])))

    def test_reference_actions_and_artifacts_are_selection_features_only(self):
        patients, labels = population()
        labels[1]["plan"] = ["one", "two", "three"]
        labels[2]["plan"] = ["( Record Artifact ) observation"]
        _, manifest = select_runtime_pilot(patients, labels)
        self.assertEqual(manifest["population_features"]["p0001"]["reference_action_count"], 3)
        self.assertFalse(manifest["population_features"]["p0002"]["judgeable_reference"])
        self.assertEqual(labels[2]["plan"], ["( Record Artifact ) observation"])

    def test_bad_population_count_seed_and_labels_fail_before_sampling(self):
        patients, labels = population()
        for count in (0, 12, 49, True, 204):
            with self.subTest(count=count), self.assertRaises(ValueError):
                select_runtime_pilot(patients, labels, count=count)
        with self.assertRaises(ValueError):
            select_runtime_pilot(patients, labels, seed=True)
        with self.assertRaises(ValueError):
            select_runtime_pilot(patients + [patients[0]], labels)
        with self.assertRaises(ValueError):
            select_runtime_pilot(patients, labels[1:])
        with self.assertRaises(ValueError):
            select_runtime_pilot(patients, labels + [labels[0]])
        with self.assertRaises(ValueError):
            select_runtime_pilot(population(40)[0], population(40)[1], count=32)


class RuntimeProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        patients, labels = population()
        selected, manifest = select_runtime_pilot(patients, labels)
        self.manifest = manifest
        self.selected = selected
        self.models = ["hosted", "local_a", "local_b", "local_c"]
        self.variants = ["llm_only", "patient_retrieval", "guidelines", "both"]
        self.costs = {"short": 1, "typical": 2, "long": 4, "complex": 8}
        generation, preparation, scoring = [], [], []
        self.records = []
        for patient in selected:
            patient_id = patient["id"]
            cost = self.costs[manifest["patient_strata"][patient_id]]
            preparation.append({"patient_id": patient_id, "seconds": 2 * cost})
            variants = {}
            for variant in self.variants:
                outputs = {}
                for model in self.models:
                    generation.append({"patient_id": patient_id, "variant": variant, "model": model,
                                       "seconds": cost, "status": "completed"})
                    outputs[model] = {"evaluation": [{"rank": rank,
                        "condition": {"status": "scored"}, "plan": {"status": "scored"},
                        "plan_correctness": {"status": "scored", "reference_actions": ["Action"],
                            "generated_actions": ["Action"], "judges": [
                            {"call": "action", "status": "scored"}, {"call": "extras", "status": "scored"}]}}
                        for rank in (1, 2, 3)]}
                variants[variant] = {"assessment_plans": outputs}
            self.records.append({"id": patient_id, "variants": variants})
        for index in range(0, len(selected), 8):
            scoring.append({"patient_ids": [row["id"] for row in selected[index:index + 8]], "seconds": 8})
        self.metadata = {
            "status": "completed", "started_at": "2026-11-11T10:00:00-08:00",
            "finished_at": "2026-11-11T10:15:00-08:00",
            "pilot_selection": manifest,
            "experiment": {"selected_models": self.models, "selected_variants": self.variants,
                           "execution": {"local_concurrency": 4, "hosted_concurrency": 4, "preparation_workers": 2}},
            "assessment_plan_models": {"models": {model: {"config": {"api_base":
                None if model == "hosted" else "http://127.0.0.1:11435/v1"}} for model in self.models}},
            "runtime_timings": {"generation": generation, "preparation": preparation,
                "scoring": scoring, "scoring_seconds": 64, "preparation_and_generation_seconds": 800,
                "model_phases": [{"model": "local_a", "seconds": 300}, {"model": "local_b", "seconds": 240},
                                 {"model": "local_c", "seconds": 240}]},
        }
        self.write()

    def write(self):
        (self.run_dir / "run.json").write_text(json.dumps(self.metadata))
        (self.run_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.records))

    def project(self):
        self.write()
        return project_runtime(self.run_dir, root=self.root)

    def test_projection_weights_complex_oversampling_to_population(self):
        result = self.project()
        expected_service = sum(self.manifest["strata"][name]["population_count"] * self.costs[name] * 4 for name in STRATA)
        self.assertAlmostEqual(expected_service, 2260)
        for model in self.models:
            estimate = result["generation_by_model"][model]
            self.assertAlmostEqual(estimate["weighted_service_seconds"], expected_service)
            self.assertAlmostEqual(estimate["wall_time"]["seconds"], 565)
        self.assertAlmostEqual(result["preparation"]["wall_time"]["seconds"], 565)
        self.assertAlmostEqual(result["scoring"]["wall_time"]["seconds"], 200)
        # Preparation and the first model overlap: they are not summed twice.
        self.assertAlmostEqual(result["pipeline"]["fixed_preparation_generation_overhead_seconds"], 80)
        self.assertAlmostEqual(result["pipeline"]["fixed_startup_report_overhead_seconds"], 36)
        self.assertAlmostEqual(result["pipeline"]["wall_time"]["seconds"], 2011)

    def test_first_model_preparation_wait_does_not_inflate_its_throughput(self):
        self.metadata["runtime_timings"]["model_phases"][0]["seconds"] = 700
        result = self.project()
        self.assertEqual(result["generation_by_model"]["local_a"]["phase_utilization_factor"], 1)
        self.assertAlmostEqual(result["generation_by_model"]["local_a"]["wall_time"]["seconds"], 565)

    def test_hosted_lane_overlaps_all_local_model_phases(self):
        for row in self.metadata["runtime_timings"]["generation"]:
            if row["model"] == "hosted":
                row["seconds"] *= 10
        self.metadata["runtime_timings"]["preparation_and_generation_seconds"] = 2480
        result = self.project()
        self.assertAlmostEqual(result["generation_by_model"]["hosted"]["wall_time"]["seconds"], 5650)
        self.assertAlmostEqual(result["pipeline"]["fixed_preparation_generation_overhead_seconds"], 80)
        # Remaining local phases finish inside the slower hosted lane.
        self.assertAlmostEqual(result["pipeline"]["wall_time"]["seconds"], 5930)

    def test_overlapped_scoring_is_not_added_to_wall_time_twice(self):
        self.metadata["experiment"]["execution"]["overlap_scoring"] = True
        timings = self.metadata["runtime_timings"]
        final = deepcopy(timings["scoring"])
        for row in final:
            row["seconds"] = 1
        timings["scoring"].extend(final)
        timings["scoring_seconds"] = 8
        result = self.project()
        self.assertAlmostEqual(result["scoring"]["overlapped_service_wall_time"]["seconds"], 200)
        self.assertAlmostEqual(result["scoring"]["final_service_wall_time"]["seconds"], 25)
        self.assertEqual(result["scoring"]["fixed_cold_start_seconds"], 0)
        self.assertAlmostEqual(result["pipeline"]["fixed_preparation_generation_overhead_seconds"], 80)
        self.assertAlmostEqual(result["pipeline"]["wall_time"]["seconds"], 1892)
        self.assertAlmostEqual(result["pipeline"]["conservative_serial_scoring_bound"]["seconds"], 2092)

    def test_slow_overlapped_scoring_starts_after_first_local_and_preparation(self):
        self.metadata["experiment"]["execution"]["overlap_scoring"] = True
        timings = self.metadata["runtime_timings"]
        final = deepcopy(timings["scoring"])
        for row in timings["scoring"]:
            row["seconds"] = 100
        for row in final:
            row["seconds"] = 0
        timings["scoring"].extend(final)
        timings["scoring_seconds"] = 0
        timings["preparation_and_generation_seconds"] = 1120
        self.metadata["finished_at"] = "2026-11-11T10:20:00-08:00"
        result = self.project()
        self.assertAlmostEqual(result["scoring"]["wall_time"]["seconds"], 2500)
        self.assertAlmostEqual(result["pipeline"]["fixed_preparation_generation_overhead_seconds"], 80)
        self.assertAlmostEqual(result["pipeline"]["wall_time"]["seconds"], 565 + 2500 + 80 + 80)

    def test_scoring_after_last_local_is_scaled_as_tail_instead_of_fixed_overhead(self):
        self.metadata["experiment"]["execution"]["overlap_scoring"] = True
        timings = self.metadata["runtime_timings"]
        for row in timings["generation"]:
            row["end_seconds"] = 700 if row["model"] == "local_c" else 200
        post_gpu, final = deepcopy(timings["scoring"]), deepcopy(timings["scoring"])
        for index, row in enumerate(timings["scoring"]):
            row.update(start_seconds=600 + index * 8, end_seconds=608 + index * 8)
        for index, row in enumerate(post_gpu):
            row.update(start_seconds=700 + index * 8, end_seconds=708 + index * 8)
        for index, row in enumerate(final):
            row.update(seconds=1, start_seconds=864 + index, end_seconds=865 + index)
        timings["scoring"].extend(post_gpu + final)
        timings["scoring_seconds"] = 8
        timings["preparation_and_generation_seconds"] = 864
        self.metadata["finished_at"] = "2026-11-11T10:16:04-08:00"
        result = self.project()
        self.assertEqual(result["scoring"]["timestamped_batch_count"], 24)
        self.assertEqual(result["scoring"]["overlapped_batch_count"], 8)
        self.assertEqual(result["scoring"]["measured_post_generation_service_inside_prepare_seconds"], 64)
        self.assertAlmostEqual(result["scoring"]["overlapped_service_wall_time"]["seconds"], 200)
        self.assertAlmostEqual(result["scoring"]["post_generation_service_wall_time"]["seconds"], 225)
        # The pilot's exposed64s tail is removed before deriving fixed cost;
        # its population-weighted200s cost is added as work, not just64s fixed.
        self.assertAlmostEqual(result["pipeline"]["fixed_preparation_generation_overhead_seconds"], 80)
        self.assertAlmostEqual(result["pipeline"]["wall_time"]["seconds"], 2092)

    def test_scoring_batch_crossing_last_local_end_is_split_proportionally(self):
        self.metadata["experiment"]["execution"]["overlap_scoring"] = True
        timings = self.metadata["runtime_timings"]
        for row in timings["generation"]:
            row["end_seconds"] = 700
        final = deepcopy(timings["scoring"])
        for row in timings["scoring"]:
            row.update(start_seconds=696, end_seconds=704)
        for index, row in enumerate(final):
            row.update(seconds=1, start_seconds=832 + index, end_seconds=833 + index)
        timings["scoring"].extend(final)
        timings["scoring_seconds"] = 8
        timings["preparation_and_generation_seconds"] = 832
        self.metadata["finished_at"] = "2026-11-11T10:15:32-08:00"
        result = self.project()
        self.assertAlmostEqual(result["scoring"]["overlapped_service_wall_time"]["seconds"], 100)
        self.assertAlmostEqual(result["scoring"]["post_generation_service_wall_time"]["seconds"], 125)
        self.assertAlmostEqual(result["scoring"]["measured_post_generation_service_inside_prepare_seconds"], 32)
        self.assertAlmostEqual(result["pipeline"]["fixed_preparation_generation_overhead_seconds"], 80)
        self.assertAlmostEqual(result["pipeline"]["wall_time"]["seconds"], 1992)

    def test_reverse_scoring_timestamps_are_rejected(self):
        self.metadata["experiment"]["execution"]["overlap_scoring"] = True
        for row in self.metadata["runtime_timings"]["generation"]:
            row["end_seconds"] = 700
        self.metadata["runtime_timings"]["scoring"][0].update(start_seconds=8, end_seconds=7)
        with self.assertRaisesRegex(ValueError, "Scoring batch end precedes"):
            self.project()

    def test_output_field_coverage_is_independent_of_completed_calls(self):
        output = self.records[0]["variants"][self.variants[0]]["assessment_plans"]["local_a"]
        output["predictions"] = [{"condition": "C", "plan": ["Action"]} for _ in range(3)]
        salvaged = self.records[1]["variants"][self.variants[0]]["assessment_plans"]["local_a"]
        salvaged["predictions"] = [{"condition": None, "plan": ["Action"]},
                                   {"condition": "C", "plan": []}, {"condition": "C", "plan": None}]
        result = self.project()["generation_by_model"]["local_a"]
        self.assertEqual(result["completed_slots"], 64 * 4)
        self.assertEqual(result["output_coverage"], {"expected_ranked_pairs": 64 * 4 * 3,
            "available_conditions": 5, "valid_plans": 5, "nonempty_plans": 4, "complete_nonempty_slots": 1})

    def test_scoring_cost_uses_new_fields_in_each_snapshot(self):
        fields = {patient["id"]: self.costs[self.manifest["patient_strata"][patient["id"]]]
                  for patient in self.selected}
        self.metadata["runtime_timings"]["scoring"] = [{"patient_ids": list(fields),
            "seconds": sum(fields.values()), "newly_scored_fields_by_patient": fields,
            "start_seconds": 800, "end_seconds": 1040}]
        result = self.project()
        self.assertEqual(result["scoring"]["allocation"], "newly_scored_fields")
        self.assertAlmostEqual(result["scoring"]["wall_time"]["seconds"], 565)

    def test_noop_scoring_snapshot_retains_bookkeeping_time(self):
        self.metadata["runtime_timings"]["scoring"] = [{"patient_ids": [row["id"] for row in self.selected],
            "seconds": 64, "newly_scored_fields_by_patient": {row["id"]: 0 for row in self.selected}}]
        result = self.project()
        self.assertAlmostEqual(result["scoring"]["wall_time"]["seconds"], 200)

    def test_invalid_new_field_counts_are_rejected(self):
        self.metadata["runtime_timings"]["scoring"][0]["newly_scored_fields_by_patient"] = {}
        with self.assertRaisesRegex(ValueError, "Scoring batch field counts"):
            self.project()

    def test_resumed_pilot_cannot_project_from_reset_timings_and_downtime(self):
        self.metadata["resumptions"] = [{"started_at": "2026-11-11T10:10:00-08:00"}]
        self.write()
        before = {path.name: path.read_bytes() for path in self.run_dir.iterdir()}
        with self.assertRaisesRegex(ValueError, "fresh uninterrupted pilot"):
            project_runtime(self.run_dir, root=self.root)
        self.assertEqual({path.name: path.read_bytes() for path in self.run_dir.iterdir()}, before)

    def test_nonfirst_phase_utilization_calibrates_underfilled_concurrency(self):
        for phase in self.metadata["runtime_timings"]["model_phases"][1:]:
            phase["seconds"] = 300
        result = self.project()
        self.assertEqual(result["generation_by_model"]["local_b"]["phase_utilization_factor"], 1.25)
        self.assertAlmostEqual(result["generation_by_model"]["local_a"]["wall_time"]["seconds"], 706.25)

    def test_forced_extremes_do_not_get_random_member_population_weights(self):
        first = self.manifest["strata"]["short"]["forced_patient_ids"][0]
        for row in self.metadata["runtime_timings"]["generation"]:
            if row["patient_id"] == first:
                row["seconds"] += 100
        result = self.project()
        # One deterministically included boundary case adds its actual four-call
        # cost, not 45/16 times that cost as an unweighted sample mean would.
        self.assertAlmostEqual(result["generation_by_model"]["hosted"]["weighted_service_seconds"], 2660)

    def test_missing_stratum_timings_prevent_unjustified_full_model_estimate(self):
        self.metadata["runtime_timings"]["generation"] = [row for row in self.metadata["runtime_timings"]["generation"]
            if row["model"] != "local_a" or self.manifest["patient_strata"][row["patient_id"]] != "complex"]
        result = self.project()
        self.assertIsNone(result["generation_by_model"]["local_a"]["wall_time"])
        self.assertIsNone(result["pipeline"]["wall_time"])
        self.assertTrue(any("Cannot project local_a" in message for message in result["warnings"]))

    def test_failures_are_weighted_within_strata_instead_of_sample_mix(self):
        failed_patient = self.manifest["strata"]["complex"]["forced_patient_ids"][0]
        for row in self.metadata["runtime_timings"]["generation"]:
            if row["model"] == "hosted" and row["patient_id"] == failed_patient:
                row["status"] = "failed_incomplete"
        result = self.project()
        estimate = result["generation_by_model"]["hosted"]
        # Complex has population20. This forced case represents exactly one
        # person, so its retry adjustment uses19/20 success, not15/16.
        expected = (2260 - 20 * 4 * 8 + (20 * 4 * 8) / (19 / 20)) / 4
        self.assertAlmostEqual(estimate["wall_time"]["seconds"], expected)

    def test_cold_scoring_start_is_added_once_instead_of_scaled_by_cohort(self):
        self.metadata["runtime_timings"]["scoring"][0]["seconds"] += 100
        self.metadata["runtime_timings"]["scoring_seconds"] += 100
        self.metadata["finished_at"] = "2026-11-11T10:16:40-08:00"
        result = self.project()
        self.assertEqual(result["scoring"]["fixed_cold_start_seconds"], 100)
        self.assertAlmostEqual(result["scoring"]["wall_time"]["seconds"], 300)

    def test_plan_projection_separates_cache_assumption_and_actual_calls(self):
        logical = 64 * 4 * 4 * 3 * 2
        self.metadata["plan_evaluation"] = {"status": "completed", "calls": logical // 2,
            "cache_hits": logical // 2, "started_at": "2026-11-11T11:00:00-08:00",
            "finished_at": "2026-11-11T11:10:00-08:00"}
        result = self.project()
        judge = result["plan_judging"]
        self.assertEqual(judge["observed_cache_hit_fraction"], 0.5)
        self.assertAlmostEqual(judge["weighted_full_logical_scalar_requests"], 200 * 4 * 4 * 3 * 2)
        self.assertAlmostEqual(judge["observed_cache_reuse"]["seconds"], 600 * 200 / 64)
        self.assertAlmostEqual(judge["without_cache_reuse"]["seconds"], 2 * 600 * 200 / 64)
        self.assertEqual(judge["theoretical_full_requests_before_empty_generation_skips"], 19200)

    def test_cache_only_resume_cannot_claim_free_full_cohort_judging(self):
        self.metadata["plan_evaluation"] = {"status": "completed", "calls": 0, "cache_hits": 0,
            "started_at": "2026-11-11T11:00:00-08:00", "finished_at": "2026-11-11T11:00:00-08:00"}
        result = self.project()
        self.assertIsNone(result["plan_judging"]["observed_cache_reuse"])
        self.assertNotIn("including_plan_judging", result["pipeline"])
        self.assertTrue(any("cache-only" in message for message in result["warnings"]))

    def test_separate_plan_projection_accepts_generation_resume_without_its_timings(self):
        logical = 64 * 4 * 4 * 3 * 2
        self.metadata["resumptions"] = [{"started_at": "2026-11-11T10:14:00-08:00"}]
        del self.metadata["runtime_timings"]
        self.metadata["plan_evaluation"] = {"status": "completed", "calls": logical // 2,
            "cache_hits": logical // 2, "started_at": "2026-11-11T11:00:00-08:00",
            "finished_at": "2026-11-11T11:10:00-08:00"}
        self.write()
        before = {path.name: path.read_bytes() for path in self.run_dir.iterdir()}
        result = project_plan_runtime(self.run_dir, root=self.root)
        self.assertEqual(result["provenance"]["generation_resumption_count"], 1)
        self.assertFalse(result["provenance"]["generation_timings_used"])
        self.assertNotIn("pipeline", result)
        self.assertAlmostEqual(result["plan_judging"]["observed_cache_reuse"]["seconds"], 600 * 200 / 64)
        for name, source in result["provenance"]["source_artifacts"].items():
            self.assertEqual(source["path"], str((self.run_dir / name).resolve()))
            self.assertEqual(source["sha256"], hashlib.sha256(before[name]).hexdigest())
        self.assertEqual({path.name: path.read_bytes() for path in self.run_dir.iterdir()}, before)
        # No weakened guard or synthetic uninterrupted run is necessary.
        with self.assertRaisesRegex(ValueError, "fresh uninterrupted pilot"):
            project_runtime(self.run_dir, root=self.root)

    def test_full_attempt_with_invalid_judge_replies_is_distinct_from_valid_completion(self):
        logical, entries = 64 * 4 * 4 * 3 * 2, 64 * 4 * 4 * 3
        for evaluation in self.records[0]["variants"][self.variants[0]]["assessment_plans"]["hosted"]["evaluation"][:2]:
            evaluation["plan_correctness"]["status"] = "error"
            evaluation["plan_correctness"]["judges"][1].update(status="invalid_response",
                error={"type": "JudgeResponseError", "message": "extras count exceeds generated actions"})
        self.metadata["plan_evaluation"] = {"status": "failed_incomplete", "calls": logical - 9,
            "cache_hits": 9, "counts": {"scored": entries - 2, "error": 2, "pending": 0, "interrupted": 0},
            "started_at": "2026-11-11T11:00:00-08:00", "finished_at": "2026-11-11T11:10:00-08:00"}
        self.write()
        projection = project_plan_runtime(self.run_dir, root=self.root)
        judge = projection["plan_judging"]
        self.assertIsNone(judge["observed_cache_reuse"])
        self.assertIsNone(judge["without_cache_reuse"])
        self.assertEqual(judge["plan_entry_counts"]["error"], 2)
        self.assertEqual(judge["scalar_status_counts"], {"scored": logical - 2, "invalid_response": 2})
        self.assertAlmostEqual(judge["attempted_stage_with_observed_cache_reuse"]["seconds"], 600 * 200 / 64)
        self.assertAlmostEqual(judge["attempted_stage_without_cache_reuse"]["seconds"], 600 * (200 * 4 * 4 * 3 * 2) / (logical - 9))
        self.assertTrue(any("extra work needed" in warning for warning in projection["warnings"]))
        # An inconsistent completed label must not conceal invalid scalar rows.
        self.metadata["plan_evaluation"]["status"] = "completed"
        self.write()
        judge = project_plan_runtime(self.run_dir, root=self.root)["plan_judging"]
        self.assertIsNone(judge["observed_cache_reuse"])
        self.assertIsNotNone(judge["attempted_stage_with_observed_cache_reuse"])

    def test_attempt_projection_rejects_partial_interrupted_pending_and_cache_only_stages(self):
        logical, entries = 64 * 4 * 4 * 3 * 2, 64 * 4 * 4 * 3
        base = {"status": "failed_incomplete", "calls": logical, "cache_hits": 0,
            "counts": {"scored": entries}, "started_at": "2026-11-11T11:00:00-08:00",
            "finished_at": "2026-11-11T11:10:00-08:00"}
        cases = [("partial", {"calls": logical - 1}), ("interrupted", {"status": "interrupted"}),
                 ("pending entries", {"counts": {"scored": entries - 1, "pending": 1}}),
                 ("interrupted entries", {"counts": {"scored": entries - 1, "interrupted": 1}}),
                 ("cache only", {"calls": 0, "cache_hits": logical}), ("missing counts", {"counts": None})]
        for label, changes in cases:
            with self.subTest(case=label):
                self.metadata["plan_evaluation"] = {**deepcopy(base), **changes}
                self.write()
                judge = project_plan_runtime(self.run_dir, root=self.root)["plan_judging"]
                self.assertIsNone(judge["observed_cache_reuse"])
                self.assertIsNone(judge["attempted_stage_with_observed_cache_reuse"])
                self.assertIsNone(judge["attempted_stage_without_cache_reuse"])
        self.metadata["plan_evaluation"] = base
        scalar = self.records[0]["variants"][self.variants[0]]["assessment_plans"]["hosted"]["evaluation"][0]["plan_correctness"]["judges"][0]
        for status in ("pending", "interrupted", "unknown"):
            with self.subTest(scalar_status=status):
                scalar["status"] = status
                self.write()
                judge = project_plan_runtime(self.run_dir, root=self.root)["plan_judging"]
                self.assertIsNone(judge["attempted_stage_with_observed_cache_reuse"])
        del scalar["status"]
        self.write()
        self.assertIsNone(project_plan_runtime(self.run_dir, root=self.root)["plan_judging"]["attempted_stage_without_cache_reuse"])
        scalar["status"] = "scored"
        plan = self.records[0]["variants"][self.variants[0]]["assessment_plans"]["hosted"]["evaluation"][0]["plan_correctness"]
        plan["judges"].pop()  # An unattempted extras call has no scalar row.
        self.write()
        judge = project_plan_runtime(self.run_dir, root=self.root)["plan_judging"]
        self.assertFalse(judge["all_required_scalar_requests_present"])
        self.assertIsNone(judge["attempted_stage_with_observed_cache_reuse"])

    def test_successful_stage_can_also_report_attempt_throughput(self):
        logical, entries = 64 * 4 * 4 * 3 * 2, 64 * 4 * 4 * 3
        self.metadata["plan_evaluation"] = {"status": "completed", "calls": logical,
            "cache_hits": 0, "counts": {"scored": entries},
            "started_at": "2026-11-11T11:00:00-08:00", "finished_at": "2026-11-11T11:10:00-08:00"}
        self.write()
        judge = project_plan_runtime(self.run_dir, root=self.root)["plan_judging"]
        self.assertEqual(judge["observed_cache_reuse"], judge["attempted_stage_with_observed_cache_reuse"])
        self.assertEqual(judge["without_cache_reuse"], judge["attempted_stage_without_cache_reuse"])

    def test_separate_plan_projection_does_not_extrapolate_partial_or_stale_stage(self):
        logical = 64 * 4 * 4 * 3 * 2
        self.metadata["plan_evaluation"] = {"status": "completed", "calls": 1,
            "cache_hits": 0, "started_at": "2026-11-11T11:00:00-08:00",
            "finished_at": "2026-11-11T11:10:00-08:00"}
        self.write()
        result = project_plan_runtime(self.run_dir, root=self.root)
        self.assertIsNone(result["plan_judging"]["observed_cache_reuse"])
        self.assertTrue(any("partial resume" in warning for warning in result["warnings"]))
        # Even a nearly complete earlier invocation omits work when this
        # invocation handles only the remaining requests.
        self.metadata["plan_evaluation"]["calls"] = logical - 1
        self.write()
        self.assertIsNone(project_plan_runtime(self.run_dir, root=self.root)["plan_judging"]["observed_cache_reuse"])
        self.metadata["plan_evaluation"]["calls"] = logical
        self.metadata["finished_at"] = "2026-11-11T12:00:00-08:00"
        self.write()
        result = project_plan_runtime(self.run_dir, root=self.root)
        self.assertIsNone(result["plan_judging"]["observed_cache_reuse"])
        self.assertTrue(any("predates generation" in warning for warning in result["warnings"]))

    def test_separate_plan_projection_reports_absent_stage_and_validates_inputs(self):
        result = project_plan_runtime(self.run_dir, root=self.root)
        self.assertIsNone(result["plan_judging"])
        self.assertTrue(any("No finished" in warning for warning in result["warnings"]))
        self.metadata["plan_evaluation"] = {"status": "completed", "calls": True,
            "cache_hits": 0, "started_at": "2026-11-11T11:00:00-08:00",
            "finished_at": "2026-11-11T11:10:00-08:00"}
        self.write()
        with self.assertRaisesRegex(ValueError, "nonnegative integers"):
            project_plan_runtime(self.run_dir, root=self.root)
        del self.metadata["pilot_selection"]
        self.write()
        with self.assertRaisesRegex(ValueError, "pilot_selection"):
            project_plan_runtime(self.run_dir, root=self.root)

    def test_projection_does_not_change_run_artifacts(self):
        before = {path.name: path.read_bytes() for path in self.run_dir.iterdir()}
        project_runtime(self.run_dir, root=self.root)
        self.assertEqual({path.name: path.read_bytes() for path in self.run_dir.iterdir()}, before)

    def test_missing_manifest_invalid_concurrency_and_nonfinite_times_fail(self):
        del self.metadata["pilot_selection"]
        with self.assertRaises(ValueError):
            self.project()
        self.metadata["pilot_selection"] = self.manifest
        self.metadata["experiment"]["execution"]["local_concurrency"] = True
        with self.assertRaises(ValueError):
            self.project()
        self.metadata["experiment"]["execution"]["local_concurrency"] = 4
        self.metadata["runtime_timings"]["generation"][0]["seconds"] = float("nan")
        with self.assertRaises(ValueError):
            self.project()


if __name__ == "__main__":
    unittest.main()
