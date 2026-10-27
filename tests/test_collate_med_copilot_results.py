"""Offline tests for the standalone, read-only MED-COPILOT results report."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
from io import StringIO
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# Direct execution puts tests/ on sys.path; project imports need the repository root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collate_med_copilot_results import (
    METRICS, format_top_rank_summary, generate_report, load_report_data, main, render_report,
)


SCRIPT = ROOT / "scripts/collate_med_copilot_results.py"


def score(value=0.5, status="scored"):
    return {"status": status,
            "metrics": dict.fromkeys(METRICS, value if status == "scored" else None),
            "error": None}


def output(value=0.5):
    return {
        "status": "completed", "error": None,
        "predictions": [{"condition": f"Condition {rank}",
                         "plan": [f"Action {rank}.1", f"Action {rank}.2", f"Action {rank}.3"]}
                        for rank in range(1, 4)],
        "evaluation": [{"rank": rank, "condition": score(value), "plan": score(value / 2)}
                       for rank in range(1, 4)],
        "model_responses": [{"reported_model": "reported/model-version", "response_id": "PRIVATE_RESPONSE_ID",
                             "token_usage": {"total_tokens": 123}}],
    }


class HtmlNode:
    def __init__(self, tag, attrs=()):
        self.tag = tag
        self.attrs = dict(attrs)
        self.children = []

    def find_all(self, tag, **attrs):
        found = []
        for child in self.children:
            if isinstance(child, HtmlNode):
                if child.tag == tag and all(child.attrs.get(key) == value for key, value in attrs.items()):
                    found.append(child)
                found.extend(child.find_all(tag, **attrs))
        return found

    def find_one(self, tag, **attrs):
        return next(iter(self.find_all(tag, **attrs)), None)

    @property
    def text(self):
        return ''.join(child.text if isinstance(child, HtmlNode) else child for child in self.children)


class Document(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.text = []
        self.root = HtmlNode('document')
        self.stack = [self.root]
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        node = HtmlNode(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in ('meta', 'input', 'link', 'br', 'hr', 'img'):
            self.stack.append(node)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.text.append(data)
        self.stack[-1].children.append(data)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run = self.root / "experiments/output/example"
        self.run.mkdir(parents=True)
        self.labels_path = self.root / "labels.jsonl"
        self.labels = [{"id": "patient-1", "condition": "reference_condition", "plan": ["Reference action one", "Reference action two"]}]
        self.metadata = {
            "run_name": "PRIVATE_RUN_NAME", "git_commit": "PRIVATE_COMMIT", "status": "completed",
            "started_at": "PRIVATE_TIMESTAMP", "api_key": "PRIVATE_API_KEY",
            "evaluation": {"labels_path": "labels.jsonl", "selected_patient_ids": ["patient-1"]},
        }
        self.rows = [{"id": "patient-1", **output()}]
        self.rows[0]["model_responses"] = {
            "keyword_extraction": [{"reported_model": "wrong-stage-model"}],
            "assessment_plan": [{"reported_model": "legacy-final-version", "response_id": "PRIVATE_RESPONSE_ID"}],
        }
        self.write()

    def write(self, *, fingerprint=True):
        self.labels_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in self.labels), encoding="utf-8")
        if fingerprint:
            self.metadata["evaluation"]["labels_sha256"] = hashlib.sha256(self.labels_path.read_bytes()).hexdigest()
        else:
            self.metadata["evaluation"].pop("labels_sha256", None)
        (self.run / "run.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        (self.run / "results.jsonl").write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in self.rows), encoding="utf-8")

    def load(self):
        return load_report_data(self.run, root=self.root)

    def native_v3(self):
        self.metadata["schema_version"] = 3
        self.metadata["experiment"] = {"selected_variants": ["llm_only", "guidelines"]}
        self.metadata["assessment_plan_models"] = {"selected_models": ["mini", "llama"], "models": {
            "mini": {"config": {"model": "mini"}}, "llama": {"config": {"model": "llama"}},
        }}
        self.rows = [{"id": "patient-1", "status": "completed", "variants": {
            variant: {"status": "completed", "error": None,
                      "assessment_plan_prompts": {"user": "PRIVATE_PROMPT"},
                      "assessment_plans": {model: output(value) for model in ("mini", "llama")}}
            for variant, value in (("llm_only", 0.4), ("guidelines", 0.8))
        }}]

    def test_v3_normalizes_one_patient_variant_model_matrix_without_duplicate_shared_content(self):
        self.native_v3()
        self.write()
        before = [(self.run / name).read_bytes() for name in ("run.json", "results.jsonl")]
        data = self.load()
        self.assertEqual(len(data["patients"]), 1)
        outputs = data["patients"][0]["outputs"]
        self.assertEqual([(v["variant"], v["model_id"]) for v in outputs],
                         [(variant, model) for variant in ("llm_only", "guidelines") for model in ("mini", "llama")])
        self.assertEqual(len(data["summary"]), 24)  # 2 variants * 2 models * 3 ranks * 2 fields
        self.assertEqual(len(data["comparisons"]), 12)
        html = generate_report(self.run, root=self.root).read_text()
        self.assertIn("Best evidence variant by model", html)
        self.assertNotIn("PRIVATE_PROMPT", html)
        self.assertEqual(before, [(self.run / name).read_bytes() for name in ("run.json", "results.jsonl")])

    def test_v3_missing_patients_variants_and_models_remain_explicitly_unprocessed(self):
        self.native_v3()
        self.labels.append({"id": "patient-2", "condition": "second", "plan": ["action"]})
        self.metadata["evaluation"]["selected_patient_ids"].append("patient-2")
        del self.rows[0]["variants"]["guidelines"]
        del self.rows[0]["variants"]["llm_only"]["assessment_plans"]["llama"]
        self.write()
        data = self.load()
        self.assertEqual(len(data["patients"]), 2)
        self.assertEqual(len(data["patients"][1]["outputs"]), 4)
        outputs = [v for patient in data["patients"] for v in patient["outputs"]]
        self.assertEqual(sum(v["status"] == "not_processed" for v in outputs), 7)
        self.assertTrue(all(pair["scores"]["condition"]["metrics"]["BERTScore_F1"] is None
                            for v in outputs if v["status"] == "not_processed" for pair in v["predictions"]))
        summary = format_top_rank_summary(data)
        self.assertIn("Best variant: llm_only", summary)
        self.assertIn("guidelines: condition: N/A (n=0); plan: N/A (n=0)", summary)

    def native_v4(self):
        self.native_v3()
        self.metadata["schema_version"] = 4
        self.metadata["experiment"]["selected_variants"] = ["llm_only", "custom_evidence"]
        self.metadata["variant_definitions"] = {
            "llm_only": {"description": "Saved S/O-only description"},
            "custom_evidence": {"description": "Saved custom evidence description",
                                "components": ["custom_component"], "prompt_key": "guidelines",
                                "private_implementation_detail": "PRIVATE_SOURCE_PATH"},
        }
        self.rows[0]["components"] = {
            "custom_component": {"status": "completed", "error": None,
                                 "guideline_evidence": "PRIVATE_COMPONENT_EVIDENCE",
                                 "trace": ["PRIVATE_AGENT_TRACE"]},
        }
        custom = self.rows[0]["variants"].pop("guidelines")
        custom["components"] = ["custom_component"]
        self.rows[0]["variants"]["custom_evidence"] = custom

    def test_v4_saved_variant_is_reportable_without_a_current_registry_entry(self):
        self.native_v4()
        self.write()
        before = [(self.run / name).read_bytes() for name in ("run.json", "results.jsonl")]
        data = self.load()
        self.assertEqual(data["selected_variants"], ["llm_only", "custom_evidence"])
        self.assertEqual([(item["variant"], item["model_id"]) for item in data["patients"][0]["outputs"]],
                         [(variant, model) for variant in ("llm_only", "custom_evidence")
                          for model in ("mini", "llama")])
        self.assertEqual(data["variant_labels"]["custom_evidence"], "Saved custom evidence description")
        html = generate_report(self.run, root=self.root).read_text()
        self.assertIn('<option value="custom_evidence">Saved custom evidence description</option>', html)
        self.assertIn("Saved S/O-only description", html)
        self.assertIn('<tr data-variant="custom_evidence" data-best="true">', html)
        for private in ("PRIVATE_SOURCE_PATH", "PRIVATE_COMPONENT_EVIDENCE", "PRIVATE_AGENT_TRACE", "PRIVATE_PROMPT"):
            self.assertNotIn(private, html)
        self.assertEqual(before, [(self.run / name).read_bytes() for name in ("run.json", "results.jsonl")])

    def test_v4_component_failures_do_not_override_saved_final_model_statuses(self):
        self.native_v4()
        self.rows[0]["components"]["custom_component"].update(status="failed_incomplete",
                                                            error={"type": "RuntimeError", "message": "PRIVATE_ERROR"})
        custom = self.rows[0]["variants"]["custom_evidence"]
        custom.update(status="failed_incomplete", error={"type": "EvidenceUnavailable"})
        custom["assessment_plans"] = {model: {"status": "not_processed"} for model in ("mini", "llama")}
        self.write()
        data = self.load()
        outputs = data["patients"][0]["outputs"]
        self.assertEqual([item["status"] for item in outputs],
                         ["completed", "completed", "not_processed", "not_processed"])
        self.assertIn("Best variant: llm_only", format_top_rank_summary(data))
        self.assertNotIn("PRIVATE_ERROR", render_report(data))

    def test_v4_missing_selected_patient_variant_and_model_slots_stay_explicit(self):
        self.native_v4()
        self.labels.append({"id": "patient-2", "condition": "second", "plan": ["action"]})
        self.metadata["evaluation"]["selected_patient_ids"].append("patient-2")
        del self.rows[0]["variants"]["custom_evidence"]
        del self.rows[0]["variants"]["llm_only"]["assessment_plans"]["llama"]
        self.write()
        data = self.load()
        self.assertEqual([patient["id"] for patient in data["patients"]], ["patient-1", "patient-2"])
        outputs = [item for patient in data["patients"] for item in patient["outputs"]]
        self.assertEqual(len(outputs), 8)
        self.assertEqual(sum(item["status"] == "not_processed" for item in outputs), 7)
        self.assertEqual(len(data["comparisons"]), 12)
        self.assertTrue(all(row["paired_patients"] == 0 for row in data["comparisons"]))
        self.assertIn("custom_evidence: condition: N/A (n=0)", format_top_rank_summary(data))

    def test_v4_comparisons_keep_same_patient_pairing_and_configured_tie_order(self):
        self.native_v4()
        for variant in self.rows[0]["variants"].values():
            for slot in variant["assessment_plans"].values():
                slot["evaluation"][0]["condition"] = score(0)
        self.metadata["experiment"]["selected_variants"].reverse()
        self.write()
        data = self.load()
        self.assertTrue(all(row["paired_patients"] == 1 for row in data["comparisons"]))
        top = [row for row in data["comparisons"] if row["rank"] == 1 and row["field"] == "condition"]
        self.assertTrue(all(row["differences"]["BERTScore_F1"] == 0 for row in top))
        self.assertIn("Best variant: custom_evidence", format_top_rank_summary(data))

    def test_saved_variant_descriptions_are_escaped_in_html_and_embedded_json(self):
        self.native_v4()
        attack = '</script><img src=x onerror="alert(1)"> & \u2028'
        self.metadata["variant_definitions"]["custom_evidence"]["description"] = attack
        self.write()
        data = self.load()
        html = render_report(data)
        self.assertNotIn(attack, html)
        self.assertNotIn('<img src=x', html)
        self.assertIn('&lt;img src=x', html)
        payload = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S)[1]
        self.assertEqual(json.loads(payload)["variant_labels"]["custom_evidence"], attack)
        self.assertEqual(html.count('</script>'), 2)

    def test_current_registry_supplies_new_variant_labels_without_reporter_changes(self):
        self.native_v4()
        self.metadata.pop("variant_definitions")
        registry = {"llm_only": SimpleNamespace(description="Registered baseline"),
                    "custom_evidence": SimpleNamespace(description="Registered custom")}
        self.write()
        with patch("scripts.collate_med_copilot_results.REGISTERED_VARIANTS", registry):
            data = self.load()
        self.assertEqual(data["variant_labels"]["custom_evidence"], "Registered custom")
        self.assertIn("Registered custom", render_report(data))

    def test_saved_legacy_definition_without_description_uses_stable_identity(self):
        self.native_v4()
        self.metadata["variant_definitions"]["custom_evidence"].pop("description")
        self.metadata["schema_version"] = 3
        self.write()
        data = self.load()
        self.assertEqual(data["variant_labels"]["custom_evidence"], "custom_evidence")
        self.assertEqual(data["patients"][0]["outputs"][2]["variant"], "custom_evidence")

    def test_v4_rejects_malformed_saved_definitions_and_undeclared_variants(self):
        for definitions in ([], {"": {}}, {"not_recorded": {}}, {"custom_evidence": []},
                            {"custom_evidence": {"description": " "}},
                            {"custom_evidence": {"description": 1}}, {}, None):
            with self.subTest(definitions=definitions):
                self.native_v4()
                self.metadata["variant_definitions"] = definitions
                self.write()
                with self.assertRaises(ValueError):
                    self.load()

    def test_v4_requires_variant_matrix_metadata_and_unique_patient_records(self):
        for case in ("empty_selection", "missing_variants", "duplicate_patient"):
            with self.subTest(case=case):
                self.native_v4()
                if case == "empty_selection": self.metadata["experiment"]["selected_variants"] = []
                if case == "missing_variants": self.rows[0].pop("variants")
                if case == "duplicate_patient": self.rows.append(deepcopy(self.rows[0]))
                self.write()
                with self.assertRaisesRegex(ValueError, "Schema v4 requires"):
                    self.load()

    def test_v3_rejects_invalid_variant_metadata_rows_and_duplicate_patients_cleanly(self):
        cases = ["bad_selection", "duplicate_selection", "unknown_variant", "invalid_variants", "invalid_slot",
                 "missing_variants", "duplicate_patient", "invalid_patient", "invalid_selected_patient", "empty_selection"]
        for case in cases:
            with self.subTest(case=case):
                self.native_v3()
                self.metadata["evaluation"]["selected_patient_ids"] = ["patient-1"]
                if case == "bad_selection": self.metadata["experiment"]["selected_variants"] = [{}]
                if case == "duplicate_selection": self.metadata["experiment"]["selected_variants"] = ["llm_only"] * 2
                if case == "unknown_variant": self.rows[0]["variants"]["bad"] = {}
                if case == "invalid_variants": self.rows[0]["variants"] = []
                if case == "invalid_slot": self.rows[0]["variants"]["llm_only"] = {"assessment_plans": []}
                if case == "missing_variants": del self.rows[0]["variants"]
                if case == "duplicate_patient": self.rows *= 2
                if case == "invalid_patient": self.rows[0]["id"] = []
                if case == "invalid_selected_patient": self.metadata["evaluation"]["selected_patient_ids"] = [{}]
                if case == "empty_selection": self.metadata["experiment"]["selected_variants"] = []
                self.write()
                with self.assertRaises(ValueError): self.load()

    def test_best_variant_uses_only_top_condition_bert_mean_and_plan_statistics_are_separate(self):
        self.native_v3()
        self.labels.append({"id": "patient-2", "condition": "second", "plan": ["action"]})
        self.metadata["evaluation"]["selected_patient_ids"].append("patient-2")
        later = deepcopy(self.rows[0])
        later["id"] = "patient-2"
        self.rows.append(later)
        for row, value in zip(self.rows, (0.0, 1.0)):
            baseline = row["variants"]["llm_only"]["assessment_plans"]["mini"]
            baseline["evaluation"][0]["condition"] = score(value)
            baseline["evaluation"][0]["plan"] = score(0.9)
            baseline["evaluation"][1]["condition"] = score(100)  # rank 2 cannot choose the winner
        self.write()
        summary = format_top_rank_summary(self.load())
        mini, llama = summary.split("Model: mini\n", 1)[1].split("Model: llama", 1)
        self.assertIn("llm_only: condition: 0.5000 (n=2); plan: 0.9000 (n=2)", mini)
        self.assertIn("Best variant: guidelines", mini)
        self.assertIn("Top-ranked condition: mean=0.8000, population_sd=0.0000, min=0.8000, max=0.8000, n=2", mini)
        self.assertIn("Top-ranked plan: mean=0.4000", mini)
        self.assertIn("Best variant: guidelines", llama)

    def test_summary_zero_missing_scores_ties_and_counts_are_not_conflated(self):
        self.native_v3()
        for slot in self.rows[0]["variants"].values():
            slot["assessment_plans"]["mini"] = output(0)
            slot["assessment_plans"]["mini"]["evaluation"][0]["plan"] = score(status="skipped_no_recorded_plan")
            slot["assessment_plans"]["llama"] = {"status": "failed_incomplete", "error": {"type": "Error"}}
        self.write()
        summary = format_top_rank_summary(self.load())
        self.assertIn("llm_only: condition: 0.0000 (n=1); plan: N/A (n=0)", summary)
        self.assertIn("Best variant: llm_only", summary)
        self.assertIn("Top-ranked plan: N/A (n=0)", summary)
        self.assertIn("Best variant: unavailable (no scored top-ranked conditions)", summary)
        self.metadata["experiment"]["selected_variants"].reverse()
        self.write()
        summary = format_top_rank_summary(self.load())
        self.assertIn("Best variant: guidelines", summary)

    def test_summary_population_spread_and_html_escaping_and_empty_cohort(self):
        self.native_v3()
        self.labels.append({"id": "patient-2", "condition": "second", "plan": []})
        self.metadata["evaluation"]["selected_patient_ids"].append("patient-2")
        second = deepcopy(self.rows[0])
        second["id"] = "patient-2"
        self.rows.append(second)
        self.metadata["assessment_plan_models"]["selected_models"] = ["<script>mini</script>"]
        for row, value in zip(self.rows, (0.0, 1.0)):
            for variant in row["variants"].values():
                variant["assessment_plans"] = {"<script>mini</script>": output(value)}
        self.write()
        data = self.load()
        summary = format_top_rank_summary(data)
        self.assertIn("mean=0.5000, population_sd=0.5000, min=0.0000, max=1.0000, n=2", summary)
        html = render_report(data)
        self.assertNotIn("Model: <script>mini</script>", html)
        self.assertIn('<h3>&lt;script&gt;mini&lt;/script&gt;</h3>', html)
        self.rows = []
        self.metadata["evaluation"]["selected_patient_ids"] = []
        self.write()
        summary = format_top_rank_summary(self.load())
        self.assertIn("Model: <script>mini</script>", summary)
        self.assertIn("Best variant: unavailable", summary)
        self.assertIn("condition: N/A (n=0)", summary)

    def test_summary_of_large_finite_saved_scores_does_not_overflow(self):
        self.native_v3()
        self.labels.append({"id": "patient-2", "condition": "second", "plan": []})
        self.metadata["evaluation"]["selected_patient_ids"].append("patient-2")
        second = deepcopy(self.rows[0])
        second["id"] = "patient-2"
        self.rows.append(second)
        for row in self.rows:
            for variant in row["variants"].values():
                for model in variant["assessment_plans"]:
                    variant["assessment_plans"][model] = output(1e308)
        self.write()
        summary = format_top_rank_summary(self.load())
        self.assertIn("Best variant: llm_only", summary)
        self.assertNotIn("inf", summary)
        self.assertNotIn("nan", summary)
        self.assertIn("population_sd=0.0000", summary)

    def test_best_variant_section_has_sorted_scores_and_statistics_for_every_variant(self):
        self.native_v3()
        self.write()
        document = Document(render_report(self.load())).root
        cards = document.find_all('article', **{'class':'top-rank-card'})
        self.assertEqual([card.attrs['data-model'] for card in cards], ['mini','llama'])
        for card in cards:
            table = card.find_one('table', **{'class':'top-rank-table'})
            groups = table.find_all('tbody', **{'class':'top-rank-variant'})
            self.assertEqual([group.attrs['data-variant'] for group in groups], ['guidelines','llm_only'])
            self.assertEqual([group.attrs['data-condition-score'] for group in groups], ['0.8','0.4'])
            self.assertEqual([group.attrs['data-plan-score'] for group in groups], ['0.4','0.2'])
            self.assertEqual(len(table.find_all('button', **{'class':'score-sort'})), 2)
            for group in groups:
                details = group.find_one('details', **{'class':'variant-statistics'})
                self.assertNotIn('open', details.attrs)
                statistics = details.find_one('table', **{'class':'variant-statistics-table'})
                self.assertEqual([body.attrs['data-rank'] for body in statistics.find_all('tbody')], ['1','2','3'])
                self.assertEqual(len([row for row in statistics.find_all('tr') if 'data-metric' in row.attrs]), 36)
        self.assertNotIn('pre', [tag for tag,_ in Document(render_report(self.load())).tags])

    def test_each_variant_statistics_preserve_all_rank_population_spread_and_plan_scores(self):
        self.native_v3()
        self.labels.append({'id':'patient-2','condition':'second','plan':['action']})
        self.metadata['evaluation']['selected_patient_ids'].append('patient-2')
        second=deepcopy(self.rows[0]); second['id']='patient-2'; self.rows.append(second)
        for row,value in zip(self.rows,(0.0,1.0)):
            for variant in row['variants'].values():
                for model in variant['assessment_plans']:
                    variant['assessment_plans'][model]=output(value)
        self.write()
        document=Document(render_report(self.load())).root
        for card in document.find_all('article', **{'class':'top-rank-card'}):
            for details in card.find_all('details', **{'class':'variant-statistics'}):
                table=details.find_one('table', **{'class':'variant-statistics-table'})
                self.assertEqual([cell.text for cell in table.find_one('thead').find_all('th')],
                    ['Rank','Field / coverage','Metric','Mean','Population SD','Min','Max','Scored patients'])
                for body in table.find_all('tbody'):
                    for field,expected in [('condition',['0.5000','0.5000','0.0000','1.0000','2']),
                                           ('plan',['0.2500','0.2500','0.0000','0.5000','2'])]:
                        row=next(row for row in body.find_all('tr') if row.attrs.get('data-field')==field
                                 and row.attrs.get('data-metric')=='BERTScore_F1')
                        self.assertEqual([cell.text for cell in row.find_all('td')], expected)

    def test_best_variant_cards_keep_zero_missing_scores_and_failed_models_distinct(self):
        self.native_v3()
        for variant in self.rows[0]["variants"].values():
            variant["assessment_plans"]["mini"] = output(0)
            variant["assessment_plans"]["mini"]["evaluation"][0]["plan"] = score(status="skipped_no_recorded_plan")
            variant["assessment_plans"]["llama"] = {"status": "failed_incomplete"}
        self.write()
        html = render_report(self.load())
        cards = dict(re.findall(r'<article class="top-rank-card" data-model="([^"]*)">(.*?)</article>', html, re.S))
        self.assertIn('<tr data-variant="llm_only" data-best="true">', cards["mini"])
        self.assertIn('title="0">0.0000', cards["mini"])
        self.assertIn('class="top-rank-count">n=0', cards["mini"])
        self.assertIn('class="na">N/A', cards["mini"])
        self.assertNotIn('data-best="true"', cards["llama"])
        self.assertIn("best variant unavailable", cards["llama"])
        self.assertEqual(cards["llama"].count('class="variant-statistics"'), 2)
        self.metadata["experiment"]["selected_variants"].reverse()
        self.write()
        html = render_report(self.load())
        mini = re.search(r'<article class="top-rank-card" data-model="mini">(.*?)</article>', html, re.S)[1]
        self.assertIn('<tr data-variant="guidelines" data-best="true">', mini)

    def test_top_rank_cards_handle_an_empty_cohort_with_and_without_selected_models(self):
        self.rows = []
        self.metadata["evaluation"]["selected_patient_ids"] = []
        self.write()
        html = render_report(self.load())
        self.assertIn("No model results available", html)
        self.native_v3()
        self.rows = []
        self.metadata["evaluation"]["selected_patient_ids"] = []
        self.write()
        html = render_report(self.load())
        cards = re.findall(r'<article class="top-rank-card" data-model="([^"]*)">(.*?)</article>', html, re.S)
        self.assertEqual(len(cards), 2)
        for _, card in cards:
            self.assertIn("best variant unavailable", card)
            self.assertNotIn('data-best="true"', card)
            self.assertIn('class="top-rank-count">n=0', card)

    def test_combined_evidence_description_spells_out_sources_without_changing_saved_labels(self):
        self.native_v3()
        self.metadata['experiment']['selected_variants']=['patient_retrieval_guidelines']
        self.rows[0]['variants']={'patient_retrieval_guidelines':self.rows[0]['variants']['guidelines']}
        self.write()
        data=self.load()
        original=deepcopy(data)
        document=Document(render_report(data)).root
        heading=document.find_one('article', **{'class':'top-rank-card'}).find_one('div', **{'class':'top-rank-best'})
        self.assertIn('S/O + similar-patient example + clinical guidelines', heading.text)
        self.assertNotIn('both evidence sources', heading.text)
        self.assertEqual(data,original)

    def test_custom_variants_get_all_rank_all_metric_statistics_without_registry_changes(self):
        self.native_v4()
        self.write()
        data=self.load()
        document=Document(render_report(data)).root
        for card in document.find_all('article', **{'class':'top-rank-card'}):
            panels=card.find_all('details', **{'class':'variant-statistics'})
            self.assertEqual({panel.attrs['data-variant'] for panel in panels},{'llm_only','custom_evidence'})
            custom=next(panel for panel in panels if panel.attrs['data-variant']=='custom_evidence')
            self.assertIn('Saved custom evidence description',custom.find_one('summary').text)
            for rank,body in enumerate(custom.find_one('table').find_all('tbody'),1):
                self.assertEqual(body.attrs['data-rank'],str(rank))
                for field in ('condition','plan'):
                    rows=[row for row in body.find_all('tr') if row.attrs['data-field']==field]
                    self.assertEqual(len(rows),6)
                    for row in rows:
                        expected=0.8 if field=='condition' else 0.4
                        self.assertEqual(row.find_all('td')[0].text,f'{expected:.4f}')
                        self.assertEqual(row.find_all('td')[-1].text,'1')

    def test_default_variant_order_sorts_numeric_condition_scores_and_keeps_missing_last(self):
        self.native_v3()
        for slot in self.rows[0]['variants']['guidelines']['assessment_plans'].values():
            slot['evaluation'][0]['condition']=score(status='missing_generation')
        for slot in self.rows[0]['variants']['llm_only']['assessment_plans'].values():
            slot['evaluation'][0]['condition']=score(0)
        self.write()
        document=Document(render_report(self.load())).root
        for table in document.find_all('table', **{'class':'top-rank-table'}):
            groups=table.find_all('tbody', **{'class':'top-rank-variant'})
            self.assertEqual([group.attrs['data-variant'] for group in groups],['llm_only','guidelines'])
            self.assertEqual([group.attrs['data-condition-score'] for group in groups],['0',''])
            self.assertEqual(table.attrs['data-sort-direction'],'descending')

    @unittest.skipUnless(shutil.which('node'), 'Node.js is required for offline sorting checks')
    def test_javascript_score_sorting_is_numeric_stable_and_preserves_expanded_statistics(self):
        javascript=re.findall(r'<script>(.*?)</script>',render_report(self.load()),re.S)[0]
        harness=r'''
const assert=require('node:assert/strict'), vm=require('node:vm');
class Element {
  constructor(dataset={}) { this.dataset=dataset; this.value=''; this.listeners={}; this.attributes={}; this.hidden=false; }
  addEventListener(name,callback) { this.listeners[name]=callback; }
  fire(name) { this.listeners[name](); }
  setAttribute(name,value) { this.attributes[name]=value; }
}
function table() {
  const groups=[['0','0.9'],['0.8','0.1'],['','0.7'],['0.8','0.3'],['-0.2','']].map((values,index)=>{
    const group=new Element({variant:String(index),order:String(index),conditionScore:values[0],planScore:values[1]});
    group.statistics={open:index===3}; return group;
  });
  const result=new Element({sortField:'condition',sortDirection:'descending'});
  result.groups=[groups[1],groups[3],groups[0],groups[4],groups[2]];
  result.buttons=['condition','plan'].map(field=>{
    const button=new Element({field}); button.parentElement=new Element(); button.indicator=new Element();
    button.querySelector=()=>button.indicator; return button;
  });
  result.querySelectorAll=selector=>selector==='button.score-sort'?result.buttons:result.groups;
  result.appendChild=group=>{result.groups.splice(result.groups.indexOf(group),1);result.groups.push(group);};
  return result;
}
const tables=[table(),table()], elements={};
for(const id of ['model-filter','variant-filter','patient-search','patient-count','patients-empty','show-more',
                'show-less','expand-all','reset','collapse-controls','collapse-current','collapse-all']) elements[id]=new Element();
const document={getElementById:id=>elements[id],querySelectorAll:selector=>selector==='table.top-rank-table'?tables:[]};
vm.runInNewContext(SCRIPT,{document,window:{innerHeight:900}});
const order=target=>target.groups.map(group=>group.dataset.variant).join(',');
assert.equal(order(tables[0]),'1,3,0,4,2');
tables[0].buttons[0].fire('click'); assert.equal(order(tables[0]),'4,0,1,3,2'); // Ascending, missing stays last.
assert.equal(tables[0].buttons[0].parentElement.attributes['aria-sort'],'ascending');
assert.equal(tables[0].buttons[0].indicator.textContent,'↑');
tables[0].buttons[1].fire('click'); assert.equal(order(tables[0]),'0,2,3,1,4');
assert.equal(tables[0].buttons[0].parentElement.attributes['aria-sort'],'none');
tables[0].buttons[1].fire('click'); assert.equal(order(tables[0]),'1,3,2,0,4');
tables[0].buttons[0].fire('click'); assert.equal(order(tables[0]),'1,3,0,4,2');
assert.equal(tables[0].groups[1].statistics.open,true); // Reordering moves the same expanded DOM group.
assert.equal(order(tables[1]),'1,3,0,4,2'); // Other models retain their own sort order.
'''
        program='const SCRIPT='+json.dumps(javascript)+';\n'+harness
        result=subprocess.run([shutil.which('node'),'-e',program],text=True,capture_output=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_legacy_results_use_only_the_final_response_model_and_preserve_all_pairs(self):
        data = self.load()
        patient = data["patients"][0]
        self.assertEqual(patient["id"], "patient-1")
        self.assertEqual(patient["reference_condition"], "reference_condition")
        self.assertEqual(patient["reference_plan"], self.labels[0]["plan"])
        model = patient["outputs"][0]
        self.assertEqual(model["model_id"], "legacy-final-version")
        self.assertEqual(model["model_name"], "legacy-final-version")
        self.assertEqual(model["variant"], "not_recorded")
        self.assertEqual([pair["rank"] for pair in model["predictions"]], [1, 2, 3])
        for pair, original in zip(model["predictions"], self.rows[0]["predictions"]):
            self.assertEqual(pair["condition"], original["condition"])
            self.assertEqual(pair["plan"], original["plan"])
            for field in ("condition", "plan"):
                self.assertEqual(set(pair["scores"][field]["metrics"]), set(METRICS))
        self.assertEqual(data["comparisons"], [])
        self.assertIn("not recorded", data["notices"][0])

    def test_no_evidence_variant_is_inferred_from_saved_retrieval_or_guidelines(self):
        self.rows[0].update(selected_reference_patient={"id": "example"}, guideline_evidence="Some evidence")
        self.metadata["reference_data"] = {"patient_retrieval": {}, "guidelines": {}}
        self.write()
        self.assertEqual(self.load()["patients"][0]["outputs"][0]["variant"], "not_recorded")

    def test_multi_model_metadata_supplies_aliases_and_missing_model_names(self):
        self.metadata["assessment_plan_models"] = {
            "selected_models": ["model_a", "model_b"],
            "models": {"model_a": {"config": {"model": "configured/a"}},
                       "model_b": {"config": {"model": "configured/b"}}},
        }
        a, b = output(), output(0.8)
        b["model_responses"] = []
        self.rows = [{"id": "patient-1", "status": "completed", "assessment_plans": {"model_a": a, "model_b": b}}]
        self.write()
        outputs = self.load()["patients"][0]["outputs"]
        self.assertEqual([item["model_id"] for item in outputs], ["model_a", "model_b"])
        self.assertEqual([item["model_name"] for item in outputs], ["reported/model-version", "configured/b"])
        self.assertEqual(outputs[1]["predictions"][0]["scores"]["condition"]["metrics"]["BLEU"], 0.8)

    def test_last_reported_final_model_is_used_without_reading_current_configs(self):
        self.rows[0]["model_responses"]["assessment_plan"].extend([{}, {"reported_model": "newest-final-model"}])
        self.write()
        with patch("builtins.open", side_effect=AssertionError("No configuration reads")):
            data = self.load()
        self.assertEqual(data["patients"][0]["outputs"][0]["model_name"], "newest-final-model")

    def test_absolute_label_paths_do_not_depend_on_root_or_working_directory(self):
        self.metadata["evaluation"]["labels_path"] = str(self.labels_path)
        self.write()
        data = load_report_data(self.run, root=self.root / "unrelated")
        self.assertEqual(data["patients"][0]["id"], "patient-1")

    def test_label_fingerprint_mismatch_prevents_silently_joining_changed_labels(self):
        self.labels_path.write_text(self.labels_path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, "no longer match.*labels_sha256"):
            self.load()

    def test_missing_label_fingerprint_is_explicitly_explained(self):
        self.write(fingerprint=False)
        self.assertTrue(any("could not be verified" in note for note in self.load()["notices"]))

    def test_missing_label_path_missing_file_and_unmatched_result_id_fail(self):
        original = deepcopy(self.metadata)
        for evaluation in ({}, {"labels_path": "not-found.jsonl"}):
            with self.subTest(evaluation=evaluation):
                self.metadata["evaluation"] = evaluation
                self.write()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.load()
        self.metadata = original
        self.rows[0]["id"] = "no-label"
        self.write()
        with self.assertRaisesRegex(ValueError, "No reference label.*no-label"):
            self.load()

    def test_duplicate_or_invalid_labels_are_rejected(self):
        original = deepcopy(self.labels)
        for labels in ([*original, original[0]], [{"id": "", "condition": "x", "plan": []}],
                       [{"id": "patient-1", "condition": None, "plan": []}],
                       [{"id": "patient-1", "condition": "x", "plan": "not an array"}],
                       [{"id": "patient-1", "condition": "x", "plan": [5]}]):
            with self.subTest(labels=labels):
                self.labels = labels
                self.write()
                with self.assertRaises(ValueError):
                    self.load()

    def test_blank_lines_and_unicode_line_separators_in_text_are_preserved(self):
        self.rows[0]["predictions"][0]["condition"] = "Condition\u2028with\u2029separators"
        self.write()
        path = self.run / "results.jsonl"
        path.write_text('\n \n' + path.read_text() + '\n')
        condition = self.load()["patients"][0]["outputs"][0]["predictions"][0]["condition"]
        self.assertEqual(condition, "Condition\u2028with\u2029separators")

    def test_invalid_json_objects_duplicate_keys_and_nonfinite_literals_name_the_source(self):
        path = self.run / "results.jsonl"
        for value in ('[]', '{', '{"id":"patient-1","id":"patient-1"}', '{"value":NaN}'):
            with self.subTest(value=value):
                path.write_text('\n' + value + '\n')
                with self.assertRaisesRegex(ValueError, r"results.jsonl:2"):
                    self.load()

    def test_duplicate_patient_variant_is_rejected_but_different_variants_are_allowed(self):
        self.rows.append(deepcopy(self.rows[0]))
        self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate result"):
            self.load()
        self.rows[0]["evidence_variant"] = "llm_only"
        self.rows[1]["evidence_variant"] = "guidelines"
        self.write()
        self.assertEqual(len(self.load()["patients"][0]["outputs"]), 2)

    def test_explicit_record_variant_takes_precedence_over_experiment_variant(self):
        self.metadata["experiment"] = {"evidence_variant": "llm_only"}
        self.write()
        self.assertEqual(self.load()["patients"][0]["outputs"][0]["variant"], "llm_only")
        self.rows[0]["evidence_variant"] = "patient_retrieval_guidelines"
        self.write()
        data = self.load()
        self.assertEqual(data["patients"][0]["outputs"][0]["variant"], "patient_retrieval_guidelines")
        self.assertEqual(data["notices"], [])

    def test_unknown_variants_are_rejected_instead_of_being_guessed(self):
        for variant in ("bad_variant", 12, []):
            with self.subTest(variant=variant):
                self.rows[0]["evidence_variant"] = variant
                self.write()
                with self.assertRaisesRegex(ValueError, "Unsupported evidence_variant"):
                    self.load()

    def test_summaries_keep_ranks_fields_and_models_separate_and_include_zero(self):
        self.labels.append({"id": "patient-2", "condition": "second", "plan": ["reference"]})
        second = deepcopy(self.rows[0])
        second["id"] = "patient-2"
        second["evaluation"][0]["condition"] = score(0)
        second["evaluation"][1]["condition"] = score(0.9)
        second["evaluation"][2]["plan"] = score(None, "skipped_no_recorded_plan")
        self.rows.append(second)
        self.write()
        summary = {(row["rank"], row["field"]): row for row in self.load()["summary"]}
        self.assertEqual(len(summary), 6)
        self.assertEqual(summary[(1, "condition")]["means"]["BLEU"], 0.25)
        self.assertEqual(summary[(2, "condition")]["means"]["BLEU"], 0.7)
        self.assertEqual(summary[(1, "plan")]["means"]["BLEU"], 0.25)
        self.assertEqual(summary[(3, "plan")]["counts"], {"scored": 1, "skipped_no_recorded_plan": 1})

    def test_paired_differences_use_matching_patient_model_rank_and_field(self):
        self.labels = [{"id": f"p{number}", "condition": "reference", "plan": ["reference"]} for number in (1, 2, 3)]
        self.metadata["evaluation"]["selected_patient_ids"] = ["p1", "p2", "p3"]
        self.rows = []
        for patient_id, variant, value in (("p1", "llm_only", 0.2), ("p1", "guidelines", 0.8),
                                          ("p2", "llm_only", 0.4), ("p2", "guidelines", 0.5),
                                          ("p3", "guidelines", 1.0)):
            self.rows.append({"id": patient_id, "evidence_variant": variant,
                              "assessment_plans": {"model-a": output(value), "model-b": output(value / 2)}})
        self.write()
        data = self.load()
        comparisons = {(r["model_id"], r["rank"], r["field"]): r for r in data["comparisons"]}
        self.assertEqual(len(comparisons), 12)
        condition = comparisons[("model-a", 1, "condition")]
        self.assertEqual(condition["paired_patients"], 2)
        self.assertAlmostEqual(condition["differences"]["BLEU"], 0.35)
        self.assertAlmostEqual(comparisons[("model-a", 1, "plan")]["differences"]["BLEU"], 0.175)
        self.assertAlmostEqual(comparisons[("model-b", 1, "condition")]["differences"]["BLEU"], 0.175)
        self.assertEqual(len(data["patients"]), 3)

    def test_skipped_failed_and_missing_fields_never_enter_paired_comparisons(self):
        baseline, evidence = deepcopy(self.rows[0]), deepcopy(self.rows[0])
        baseline["evidence_variant"], evidence["evidence_variant"] = "llm_only", "patient_retrieval"
        baseline["evaluation"][0]["plan"] = score(None, "error")
        evidence["evaluation"][1]["condition"] = score(None, "missing_generation")
        baseline["evaluation"][2]["plan"] = score(None, "skipped_no_recorded_plan")
        self.rows = [baseline, evidence]
        self.write()
        comparisons = {(r["rank"], r["field"]): r for r in self.load()["comparisons"]}
        for key in ((1, "plan"), (2, "condition"), (3, "plan")):
            self.assertEqual(comparisons[key]["paired_patients"], 0)
            self.assertTrue(all(value is None for value in comparisons[key]["differences"].values()))
        self.assertEqual(comparisons[(1, "condition")]["paired_patients"], 1)

    def test_every_evidence_variant_is_supported_without_creating_fake_baselines(self):
        for variant in ("llm_only", "patient_retrieval", "guidelines", "patient_retrieval_guidelines"):
            with self.subTest(variant=variant):
                self.rows[0]["evidence_variant"] = variant
                self.write()
                data = self.load()
                self.assertTrue(all(row["paired_patients"] == 0 for row in data["comparisons"]))

    def test_unscored_statuses_preserve_explanations_and_null_all_metric_values(self):
        statuses = ("skipped_no_recorded_plan", "missing_reference", "missing_generation", "error",
                    "interrupted", "pending", "other_future_status")
        for status in statuses:
            with self.subTest(status=status):
                self.rows[0]["evaluation"][0]["plan"] = score(0.5)
                self.rows[0]["evaluation"][0]["plan"]["status"] = status
                self.write()
                data = self.load()
                entry = data["patients"][0]["outputs"][0]["predictions"][0]["scores"]["plan"]
                self.assertEqual(entry["status"], status)
                self.assertTrue(all(value is None for value in entry["metrics"].values()))
                self.assertTrue(entry["reason"])
                summary = next(row for row in data["summary"] if row["rank"] == 1 and row["field"] == "plan")
                self.assertTrue(all(value is None for value in summary["means"].values()))

    def test_incomplete_boolean_string_and_overflowing_metrics_are_unavailable(self):
        for invalid in ({"BLEU": 0.2}, dict.fromkeys(METRICS, True), dict.fromkeys(METRICS, "0.5"),
                        dict.fromkeys(METRICS, None), dict.fromkeys(METRICS, 10 ** 400)):
            with self.subTest(invalid=invalid):
                self.rows[0]["evaluation"][0]["condition"]["metrics"] = invalid
                self.write()
                data = self.load()
                entry = data["patients"][0]["outputs"][0]["predictions"][0]["scores"]["condition"]
                self.assertEqual(entry["status"], "invalid_metrics")
                self.assertTrue(all(value is None for value in entry["metrics"].values()))

    def test_evaluation_rank_is_used_even_when_saved_entries_are_out_of_order(self):
        self.rows[0]["evaluation"][0]["condition"] = score(0.1)
        self.rows[0]["evaluation"].reverse()
        self.write()
        pair = self.load()["patients"][0]["outputs"][0]["predictions"][0]
        self.assertEqual(pair["scores"]["condition"]["metrics"]["BLEU"], 0.1)

    def test_invalid_and_duplicate_evaluation_ranks_are_rejected(self):
        for rank in (True, 0, 4, "1", 2):
            with self.subTest(rank=rank):
                self.rows[0]["evaluation"][0]["rank"] = rank
                self.write()
                with self.assertRaisesRegex(ValueError, "invalid or duplicate score ranks"):
                    self.load()

    def test_invalid_generation_status_and_model_slots_are_rejected(self):
        original = deepcopy(self.rows)
        for slot in (None, [], 1, "invalid"):
            with self.subTest(slot=slot):
                self.rows = [{"id": "patient-1", "assessment_plans": slot}]
                self.write()
                with self.assertRaisesRegex(ValueError, "assessment_plans must be an object"):
                    self.load()
        for slot in ({"status": "unknown"}, {"status": None}, "invalid"):
            with self.subTest(model_output=slot):
                self.rows = [{"id": "patient-1", "assessment_plans": {"model": slot}}]
                self.write()
                with self.assertRaises(ValueError):
                    self.load()
        self.rows = original

    def test_invalid_recorded_selection_and_config_structures_fail_cleanly(self):
        original = deepcopy(self.metadata)
        cases = [
            {"experiment": "not an object"},
            {"assessment_plan_models": ["model"]},
            {"assessment_plan_models": {"models": ["model"]}},
            {"assessment_plan_models": {"selected_models": "model"}},
            {"assessment_plan_models": {"selected_models": ["model", "model"]}},
            {"assessment_plan_models": {"selected_models": [1]}},
        ]
        for update in cases:
            with self.subTest(update=update):
                self.metadata = {**deepcopy(original), **update}
                self.write()
                with self.assertRaises(ValueError):
                    self.load()
        self.metadata = deepcopy(original)
        for selected in ("patient-1", ["patient-1", "patient-1"], [None]):
            with self.subTest(selected=selected):
                self.metadata["evaluation"]["selected_patient_ids"] = selected
                self.write()
                with self.assertRaisesRegex(ValueError, "selected_patient_ids"):
                    self.load()

    def test_empty_missing_and_invalid_plan_shapes_are_shown_without_inventing_actions(self):
        for plan in ([], None, "not a list", [""], ["one", "two", "three", "four"]):
            with self.subTest(plan=plan):
                self.rows[0]["predictions"][0]["plan"] = plan
                self.write()
                data = self.load()
                generated = data["patients"][0]["outputs"][0]["predictions"][0]["plan"]
                self.assertEqual(generated, [] if plan == [] else None)
                html = render_report(data)
                self.assertIn("Results need attention", html)
                self.assertIn("No plan steps recorded" if plan == [] else "Plan unavailable", html)

    def test_missing_predictions_evaluations_and_partial_failures_remain_visible(self):
        self.rows[0].update(status="failed_incomplete", predictions=None, evaluation=None,
                            error={"type": "AuthenticationError", "message": "PRIVATE_API_KEY"})
        self.write()
        data = self.load()
        model = data["patients"][0]["outputs"][0]
        self.assertEqual(len(model["predictions"]), 3)
        self.assertEqual(model["generation_note"], "Generation failed (AuthenticationError)")
        self.assertTrue(all(pair["condition"] is None for pair in model["predictions"]))
        self.assertTrue(all(pair["scores"]["plan"]["status"] == "not_processed" for pair in model["predictions"]))
        self.assertIn("Results need attention", render_report(data))

    def test_completed_generation_without_evaluation_is_flagged_for_attention(self):
        self.rows[0]["evaluation"] = None
        self.write()
        data = self.load()
        self.assertIn("Results need attention", render_report(data))
        entry = data["patients"][0]["outputs"][0]["predictions"][0]["scores"]["plan"]
        self.assertEqual(entry["status"], "not_scored")

    def test_missing_selected_patient_and_model_slots_are_counted_as_unprocessed(self):
        self.labels.append({"id": "patient-2", "condition": "second", "plan": []})
        self.metadata["evaluation"]["selected_patient_ids"].append("patient-2")
        self.metadata["assessment_plan_models"] = {"selected_models": ["a", "b"], "models": {}}
        self.rows = [{"id": "patient-1", "assessment_plans": {"a": output()}}]
        self.write()
        data = self.load()
        self.assertEqual(len(data["patients"]), 2)
        self.assertEqual(data["patients"][0]["outputs"][1]["status"], "not_processed")
        self.assertTrue(all(item["status"] == "not_processed" for item in data["patients"][1]["outputs"]))
        a = next(row for row in data["summary"] if row["model_id"] == "a" and row["rank"] == 1 and row["field"] == "condition")
        self.assertEqual(a["counts"], {"scored": 1, "not_processed": 1})
        self.assertEqual(a["total"], 2)

    def test_unprocessed_legacy_patient_uses_known_model_when_possible(self):
        self.labels.append({"id": "patient-2", "condition": "second", "plan": []})
        self.metadata["evaluation"]["selected_patient_ids"].append("patient-2")
        self.write()
        data = self.load()
        self.assertEqual(data["patients"][1]["outputs"][0]["model_id"], "legacy-final-version")

    def test_empty_run_generates_an_explicit_empty_report(self):
        self.metadata["evaluation"]["selected_patient_ids"] = []
        self.rows = []
        self.write()
        data = self.load()
        self.assertEqual(data["patients"], [])
        self.assertEqual(data["summary"], [])
        self.assertIn("No results available", render_report(data))
        self.assertIn("No patient results", ' '.join(data["notices"]))

    def test_html_contains_labels_three_pairs_and_all_twelve_scores_per_pair(self):
        self.assertEqual(METRICS, ("BLEU", "METEOR", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore_F1"))
        data = self.load()
        html = render_report(data)
        document = Document(html)
        text = ' '.join(document.text)
        for phrase in ("reference_condition", "Reference action one", "Reference action two", "patient-1",
                       "legacy-final-version", "Condition 1", "Condition 2", "Condition 3", "Action 3.3"):
            self.assertIn(phrase, text)
        self.assertEqual(sum(tag == "tbody" and attrs.get("class") == "prediction" for tag, attrs in document.tags), 3)
        self.assertEqual(sum(tag == "table" and attrs.get("class") == "score-table" for tag, attrs in document.tags), 1)
        self.assertIn('title="0.5">0.5000', html)
        self.assertIn('class="prediction" data-rank="1"', html)
        for metric in METRICS:
            self.assertGreaterEqual(text.count(metric), 5)
        payload = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S)[1]
        self.assertEqual(json.loads(payload), data)

    def test_reference_plan_sentinel_and_real_zero_scores_are_distinct_from_na(self):
        self.labels[0]["plan"] = ["none recorded"]
        for entry in self.rows[0]["evaluation"]:
            entry["condition"] = score(0)
            entry["plan"] = score(None, "skipped_no_recorded_plan")
        self.write()
        html = render_report(self.load())
        self.assertIn("No plan recorded in label", html)
        self.assertIn("No reference plan recorded; scoring skipped", html)
        self.assertIn('title="0">0.0000', html)
        self.assertIn('class="na">N/A', html)
        self.assertIn("All patients evaluated", html)
        self.assertIn("3 condition/plan fields scored; 3 skipped; 0 unavailable", html)

    def test_header_keeps_only_patient_and_model_cards(self):
        html = render_report(self.load())
        stats = re.search(r'<div class="stats">(.*?)</header>', html, re.S)[1]
        document = Document(stats)
        self.assertEqual([text for text in document.text if text.strip()], ["Patients", "1", "Models", "1"])
        self.assertNotIn("Model outputs completed", html)
        self.assertNotIn("Condition / plan fields scored", html)
        self.assertNotIn("model outputs", html)
        self.assertIn("1/1 patients completed generation for all selected models and evidence variants", html)

    def test_patient_completion_counts_once_across_multiple_models_and_variants(self):
        self.rows = [{"id": "patient-1", "evidence_variant": variant,
                      "assessment_plans": {"a": output(), "b": output()}}
                     for variant in ("llm_only", "guidelines")]
        self.write()
        html = render_report(self.load())
        self.assertIn("1/1 patients completed generation", html)
        self.assertNotIn("4/4 patients", html)
        self.rows[1]["assessment_plans"]["b"].update(status="failed_incomplete", evaluation=None, predictions=None)
        self.write()
        html = render_report(self.load())
        self.assertIn("0/1 patients completed generation", html)
        self.assertIn("Results need attention", html)

    def test_removed_sections_are_replaced_by_complete_variant_statistics(self):
        self.rows=[{'id':'patient-1','evidence_variant':variant,'assessment_plans':{'a':output(),'b':output()}}
                   for variant in ('llm_only','guidelines')]
        self.write()
        data=self.load(); data['summary'].reverse()
        html=render_report(data)
        for marker in ('id="summary-table"','id="comparison-table"','id="overview-heading"','id="comparison-heading"',
                       '01 / Aggregate results','02 / Evidence comparison','03 / Individual results'):
            self.assertNotIn(marker,html)
        document=Document(html).root
        panels=document.find_all('details', **{'class':'variant-statistics'})
        self.assertEqual(len(panels),4)
        for panel in panels:
            table=panel.find_one('table', **{'class':'variant-statistics-table'})
            groups=table.find_all('tbody')
            self.assertEqual([group.attrs['data-rank'] for group in groups],['1','2','3'])
            for group in groups:
                rows=group.find_all('tr')
                self.assertEqual(len(rows),12)
                for field in ('condition','plan'):
                    self.assertEqual([row.attrs['data-metric'] for row in rows if row.attrs['data-field']==field],
                        ['BERTScore_F1','ROUGE-L','BLEU','METEOR','ROUGE-1','ROUGE-2'])

    def test_patient_model_variants_are_compact_collapsible_with_all_predictions(self):
        document=Document(render_report(self.load())).root
        patient=document.find_one('details', **{'class':'patient'})
        model=patient.find_one('details', **{'class':'model-output'})
        self.assertNotIn('open',model.attrs)
        self.assertIn('legacy-final-version',model.find_one('summary').text)
        self.assertIn('Reported model: legacy-final-version',model.text)
        table=model.find_one('table', **{'class':'score-table'})
        self.assertEqual([cell.text for cell in table.find_one('thead').find_all('th')][:2],['Rank','Prediction / scoring'])
        groups=table.find_all('tbody', **{'class':'prediction'})
        self.assertEqual([group.attrs['data-rank'] for group in groups],['1','2','3'])
        for rank,group in enumerate(groups,1):
            self.assertEqual([row.attrs['data-field'] for row in group.find_all('tr')],['condition','plan'])
            self.assertIn(f'Condition {rank}',group.text)
            for action in (1,2,3): self.assertIn(f'Action {rank}.{action}',group.text)
            self.assertEqual(len(group.find_all('td')),12)

    def test_patient_score_tables_put_bert_first_then_rouge_l_without_misaligning_values(self):
        values={'BLEU':0.11,'METEOR':0.22,'ROUGE-1':0.33,'ROUGE-2':0.44,'ROUGE-L':0.55,'BERTScore_F1':0.66}
        baseline,evidence=deepcopy(self.rows[0]),deepcopy(self.rows[0])
        baseline['evidence_variant'],evidence['evidence_variant']='llm_only','guidelines'
        for entry in baseline['evaluation']: entry['condition']=score(0)
        for entry in evidence['evaluation']: entry['condition']['metrics']=values.copy()
        self.rows=[baseline,evidence]; self.write()
        tables=Document(render_report(self.load())).root.find_all('table', **{'class':'score-table'})
        self.assertEqual(len(tables),2)
        for table in tables:
            self.assertEqual([cell.text for cell in table.find_one('thead').find_all('th')][-6:],
                             ['BERTScore_F1','ROUGE-L','BLEU','METEOR','ROUGE-1','ROUGE-2'])
        row=tables[1].find_one('tbody').find_all('tr')[0]
        self.assertEqual([cell.text for cell in row.find_all('td')],['0.6600','0.5500','0.1100','0.2200','0.3300','0.4400'])

    def test_patient_rows_preserve_full_ids_in_compact_headings_with_more_controls(self):
        html = render_report(self.load())
        self.assertIn('class="patient-heading"', html)
        self.assertIn('class="patient-id" title="patient-1">patient-1', html)
        self.assertIn('id="show-more"', html)
        self.assertIn('id="show-less"', html)
        self.assertIn("The first 10 matching patients are shown initially", html)
        self.assertIn("Search checks all patients", html)
        self.assertIn("Expand shown patients", html)

    def test_html_and_embedded_json_escape_untrusted_patient_model_and_label_text(self):
        attack = '</script><img src=x onerror="alert(1)"> & \u2028'
        self.labels[0]["condition"] = attack
        self.labels[0]["plan"] = [attack]
        self.rows[0]["predictions"][0] = {"condition": attack, "plan": [attack]}
        self.rows[0]["model_responses"]["assessment_plan"][0]["reported_model"] = attack
        self.write()
        html = render_report(self.load())
        self.assertNotIn(attack, html)
        self.assertNotIn('<img src=x', html)
        self.assertIn('&lt;img src=x', html)
        self.assertEqual(html.count('</script>'), 2)
        payload = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', html, re.S)[1]
        self.assertEqual(json.loads(payload)["patients"][0]["reference_condition"], attack)

    def test_run_metadata_raw_responses_prompts_tokens_and_secret_messages_are_excluded(self):
        self.rows[0].update(assessment_plan_raw="PRIVATE_RAW_REPLY", assessment_plan_prompts="PRIVATE_PROMPT")
        self.rows[0]["evaluation"][0]["plan"].update(status="error", error={"type": "RuntimeError", "message": "PRIVATE_ERROR_MESSAGE PRIVATE_API_KEY"})
        self.write()
        html = render_report(self.load())
        for text in ("PRIVATE_RUN_NAME", "PRIVATE_COMMIT", "PRIVATE_TIMESTAMP", "PRIVATE_API_KEY",
                     "PRIVATE_RESPONSE_ID", "PRIVATE_RAW_REPLY", "PRIVATE_PROMPT", "PRIVATE_ERROR_MESSAGE",
                     "token_usage", "labels_sha256", "started_at", "api_key"):
            self.assertNotIn(text, html)
        self.assertIn("Scoring failed (RuntimeError)", html)

    def test_page_is_self_contained_searchable_and_accessible_without_javascript(self):
        html = render_report(self.load())
        document = Document(html)
        self.assertFalse(any(tag == "script" and "src" in attrs for tag, attrs in document.tags))
        self.assertFalse(any(tag == "link" or tag == "iframe" for tag, _ in document.tags))
        self.assertIn('<noscript>', html)
        for control in ("model-filter", "variant-filter", "patient-search", "expand-all", "reset", "show-more", "show-less"):
            self.assertTrue(any(attrs.get("id") == control for _, attrs in document.tags))
        self.assertFalse(any(attrs.get("id") in ("rank-filter", "field-filter") for _, attrs in document.tags))
        self.assertEqual(sum(tag == "details" and attrs.get("class") == "patient" for tag, attrs in document.tags), 1)
        self.assertIn('lang="en"', html)

    @unittest.skipUnless(shutil.which("node"), "Node.js is required for the offline JavaScript interaction check")
    def test_javascript_filters_all_ranks_paginates_searches_all_patients_and_expands_shown(self):
        self.labels.append({"id": "patient-2", "condition": "second", "plan": ["reference"]})
        self.metadata["evaluation"]["selected_patient_ids"].append("patient-2")
        self.rows = [
            {"id": "patient-1", "evidence_variant": "llm_only", "assessment_plans": {"a": output()}},
            {"id": "patient-2", "evidence_variant": "guidelines", "assessment_plans": {"b": output()}},
        ]
        for number in range(3, 24):
            patient_id = f"patient-{number}"
            self.labels.append({"id": patient_id, "condition": "reference", "plan": ["reference"]})
            self.metadata["evaluation"]["selected_patient_ids"].append(patient_id)
            self.rows.append({"id": patient_id, "evidence_variant": "llm_only", "assessment_plans": {"a": output()}})
        self.write()
        data = self.load()
        html = render_report(data)
        javascript = re.findall(r'<script>(.*?)</script>', html, re.S)[0]
        # Exercise the actual report script against a minimal offline DOM. The
        # separate HTML test verifies these control IDs and data attributes exist.
        harness = r'''
const assert = require('node:assert/strict');
const vm = require('node:vm');
class Element {
  constructor(dataset={},children=[]) {
    this.dataset=dataset; this.children=children; this.hidden=false; this.open=false;
    this.value=''; this.textContent=''; this.listeners={}; this.bounds={top:0,bottom:100};
  }
  querySelectorAll() { return this.children; }
  addEventListener(name,callback) { this.listeners[name]=callback; }
  fire(name) { this.listeners[name](); }
  getBoundingClientRect() { return this.bounds; }
  scrollIntoView() { this.scrolled=true; }
}
const elements={};
for (const id of ['model-filter','variant-filter','patient-search','show-more','show-less','patients-empty',
                  'patient-count','expand-all','reset','collapse-controls','collapse-current','collapse-all'])
  elements[id]=new Element();
const cards=DATA.patients.map(patient=>new Element({search:patient.id+' '+patient.reference_condition},
  patient.outputs.map(output=>new Element({model:output.model_id,variant:output.variant}))));
const document={getElementById:id=>elements[id],querySelectorAll:selector=>selector==='details.patient'?cards:[]};
vm.runInNewContext(SCRIPT,{document,window:{innerHeight:900}});
const visible=()=>cards.filter(card=>!card.hidden);
assert.equal(visible().length,10);
assert.equal(elements['patient-count'].textContent,'Showing 10 of 23 matching patients');
assert.equal(elements['show-more'].textContent,'Show 10 more patients (13 remaining)');
assert.equal(elements['collapse-controls'].hidden,true);
elements['expand-all'].fire('click');
assert.ok(cards.slice(0,10).every(card=>card.open)); assert.equal(cards[10].open,false);
assert.ok(cards.every(card=>card.children.every(output=>!output.open))); // Predictions stay compact.
assert.equal(visible().length,10); assert.equal(elements['collapse-controls'].hidden,false);
elements['expand-all'].fire('click'); assert.ok(cards.every(card=>!card.open));
elements['show-more'].fire('click'); assert.equal(visible().length,20);
assert.equal(elements['show-more'].textContent,'Show 3 more patients (3 remaining)');
elements['show-more'].fire('click'); assert.equal(visible().length,23);
assert.equal(elements['show-more'].hidden,true);
elements['show-less'].fire('click'); assert.equal(visible().length,10);
elements['model-filter'].value='a'; elements['model-filter'].fire('change');
assert.equal(cards[0].hidden,false); assert.equal(cards[1].hidden,true);
assert.equal(elements['patient-count'].textContent,'Showing 10 of 22 matching patients');
elements['variant-filter'].value='guidelines'; elements['variant-filter'].fire('change');
assert.equal(elements['patients-empty'].hidden,false); assert.equal(elements['show-more'].hidden,true);
elements['collapse-current'].fire('click'); // Empty filtered views are harmless.
elements['reset'].fire('click'); assert.equal(visible().length,10);
elements['patient-search'].value='PATIENT-23'; elements['patient-search'].fire('input');
assert.equal(cards[22].hidden,false); assert.equal(visible().length,1);
assert.equal(elements['patient-count'].textContent,'Showing 1 of 1 matching patients');
elements['expand-all'].fire('click'); cards[22].children[0].open=true;
assert.equal(elements['expand-all'].textContent,'Collapse shown patients');
elements['collapse-current'].fire('click');
assert.equal(cards[22].open,false); assert.equal(cards[22].children[0].open,false);
assert.equal(cards[22].scrolled,true); assert.equal(elements['collapse-controls'].hidden,true);
elements['reset'].fire('click');
cards[0].open=true; cards[0].bounds={top:-500,bottom:-1};
cards[1].open=true; cards[1].children[0].open=true; cards[1].bounds={top:-100,bottom:700};
cards[1].fire('toggle'); elements['collapse-current'].fire('click');
assert.equal(cards[0].open,true); assert.equal(cards[1].open,false);
assert.equal(cards[1].children[0].open,false); assert.equal(cards[1].scrolled,true);
cards[22].open=true; cards[22].children[0].open=true; // Also close off-screen/hidden state.
elements['collapse-all'].fire('click');
assert.ok(cards.every(card=>!card.open && card.children.every(output=>!output.open)));
assert.equal(elements['collapse-controls'].hidden,true);
elements['show-more'].fire('click'); elements['patient-search'].value='patient';
elements['patient-search'].fire('input'); assert.equal(visible().length,10);
'''
        program = 'const DATA = ' + json.dumps(data) + ';\nconst SCRIPT = ' + json.dumps(javascript) + ';\n' + harness
        result = subprocess.run([shutil.which("node"), "-e", program], text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_generate_writes_only_html_and_preserves_inputs_byte_for_byte(self):
        snapshots = {path: path.read_bytes() for path in (self.run / "run.json", self.run / "results.jsonl", self.labels_path)}
        path = generate_report(self.run, root=self.root)
        self.assertEqual(path, self.run / "results_summary.html")
        self.assertTrue(path.read_text().startswith('<!doctype html>'))
        self.assertEqual({p: p.read_bytes() for p in snapshots}, snapshots)
        self.assertEqual({p.name for p in self.run.iterdir()}, {"run.json", "results.jsonl", "results_summary.html"})
        first = path.read_bytes()
        self.assertEqual(generate_report(self.run, root=self.root).read_bytes(), first)

    def test_validation_failure_preserves_old_report_and_leaves_no_temporary_file(self):
        path = generate_report(self.run, root=self.root)
        original = path.read_bytes()
        self.labels_path.write_text('changed labels')
        with self.assertRaises(ValueError):
            generate_report(self.run, root=self.root)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.run.glob('.results_summary-*.tmp')), [])

    def test_replace_failure_preserves_old_report_and_cleans_up_temporary_file(self):
        path = generate_report(self.run, root=self.root)
        original = path.read_bytes()
        with patch("scripts.collate_med_copilot_results.os.replace", side_effect=OSError("disk unavailable")):
            with self.assertRaisesRegex(OSError, "disk unavailable"):
                generate_report(self.run, root=self.root)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.run.glob('.results_summary-*.tmp')), [])

    def test_main_reports_its_destination(self):
        self.metadata["evaluation"]["labels_path"] = str(self.labels_path)
        self.write()
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main([str(self.run)]), 0)
        self.assertIn(str(self.run / "results_summary.html"), stdout.getvalue())
        self.assertTrue((self.run / "results_summary.html").exists())

    def test_main_reports_actionable_failure_and_nonzero_exit_without_creating_report(self):
        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            main([str(self.root / "missing")])
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("Cannot generate results summary", stderr.getvalue())
        self.assertFalse((self.run / "results_summary.html").exists())

    def test_script_runs_from_another_working_directory_without_model_packages_or_network(self):
        self.metadata["evaluation"]["labels_path"] = str(self.labels_path)
        self.write()
        program = (
            "import builtins, runpy, socket, sys\n"
            "def blocked(*args, **kwargs): raise AssertionError('Network access attempted')\n"
            "socket.socket.connect = blocked\n"
            "socket.getaddrinfo = blocked\n"
            "original_import = builtins.__import__\n"
            "def guarded_import(name, *args, **kwargs):\n"
            "    if name.split('.')[0] in ('torch', 'pandas', 'openai', 'litellm', 'transformers', 'graphrag'):\n"
            "        raise AssertionError('Model package imported: ' + name)\n"
            "    return original_import(name, *args, **kwargs)\n"
            "builtins.__import__ = guarded_import\n"
            f"sys.argv = [{str(SCRIPT)!r}, {str(self.run)!r}]\n"
            f"runpy.run_path({str(SCRIPT)!r}, run_name='__main__')\n"
        )
        result = subprocess.run([sys.executable, "-I", "-c", program], cwd=self.root,
                                text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Results summary written", result.stdout)
        self.assertTrue((self.run / "results_summary.html").exists())


if __name__ == "__main__":
    unittest.main()
