"""Offline tests for scripts/med_copilot_question_screen.py.

Run directly: python tests/test_med_copilot_question_screen.py
"""

from pathlib import Path
import runpy
import unittest

ROOT = Path(__file__).resolve().parents[1]
S = runpy.run_path(str(ROOT / "scripts/med_copilot_question_screen.py"))
find, remove = S["find_diagnosis_terms"], S["remove_diagnosis_terms"]


class FindDiagnosisTermsTest(unittest.TestCase):
    def test_findings_only_reply_is_clean(self):
        self.assertEqual(find("Which causes fit dyspnea, orthopnea, and BP 150/107?", "dyspnea"), [])

    def test_unrecorded_names_are_found_in_group_order(self):
        reply = "Stage by NYHA class? Treat hypertension or heart failure?"
        self.assertEqual(find(reply, "Subjective: dyspnea"), ["heart failure", "nyha", "hypertension"])

    def test_recorded_condition_allows_its_synonyms(self):
        so = "Active problems include essential hypertension."
        self.assertEqual(find("Manage high blood pressure and hypertension?", so), [])

    def test_recorded_group_does_not_allow_other_groups(self):
        so = "History of acute viral pharyngitis."
        self.assertEqual(find("Is this pharyngitis or strep?", so), ["strep"])

    def test_whole_words_and_case(self):
        self.assertEqual(find("Prehypertension? Streptomycin? COPD?", ""), ["copd"])

    def test_missing_reply_or_so(self):
        self.assertEqual(find(None, None), [])


class RemoveDiagnosisTermsTest(unittest.TestCase):
    def test_terms_are_replaced_with_neutral_phrase(self):
        self.assertEqual(remove("Manage Hypertension now?", ["hypertension"]), "Manage these findings now?")

    def test_following_symptoms_word_is_absorbed(self):
        self.assertEqual(remove("Treat heart failure symptoms?", ["heart failure"]), "Treat these findings?")

    def test_longer_terms_go_first_and_json_stays_valid(self):
        import json
        reply = '{"Key Questions": ["Chronic obstructive disease or COPD?"]}'
        cleaned = remove(reply, ["copd", "chronic obstructive"])
        self.assertEqual(json.loads(cleaned)["Key Questions"], ["these findings disease or these findings?"])


if __name__ == "__main__":
    unittest.main()
