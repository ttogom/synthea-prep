# Plan: Guideline Questions for Concrete Plan Actions

**Branch:** `plan-focused-guideline-questions`
**Status:** steps 0–2 and 5 done (2026-11-04); steps 3–4 waiting on a personal OpenAI API key

## Task

Adjust `KEY_QUESTIONS_TEMPLATE` in
[med_copilot/prompts/diagnosis_plan.py](../med_copilot/prompts/diagnosis_plan.py)
so the generated GraphRAG questions also retrieve guidance on the concrete
actions that appear in recorded Plans: investigations, specific treatments,
classifications, and patient education. Keep the existing maximum of four
questions and the `"Key Questions"` JSON format.

Important: Describe these actions as **general categories**,
not as our exact label items, so the prompt is not tuned to the answer key.

## Where this sits in the pipeline

The runner ([scripts/run_med_copilot.py](../scripts/run_med_copilot.py)) runs
these steps for each patient. Only step 6 is the model being evaluated. Steps
1, 4 and 5c are helper calls that gather evidence.

```text
INPUT: patient S/O
│
├── PATIENT RETRIEVAL
│   1. Keyword extraction       LLM: gpt-4o-mini
│   2. Similar-patient search   BioClinicalBERT + FAISS + keyword overlap
│   3. Rerank                   cross-encoder (ms-marco-MiniLM)
│      → best reference SOAP → {example}
│
├── GUIDELINES
│   4. Question generation      LLM: gpt-4o-mini, KEY_QUESTIONS_TEMPLATE   ★ this task
│      in:  S/O   out: {"Key Questions": [≤4 questions]}
│   5. GraphRAG local search    GraphRAG library
│      5a. match questions to entities   text-embedding-3-small + LanceDB
│      5b. collect linked guideline chunks, relationships, summaries
│      5c. write an answer               LLM: gpt-4o-mini
│      → {Key_info}
│
6. A/P prediction               models under test, EVALUATE_TEMPLATE_KEYINFO
   in: S/O + {example} + {Key_info}   out: 3 × {condition, plan[≤3]}
7. Scoring                      BLEU / ROUGE / METEOR / BERTScore vs labels
```

The change only affects the `guidelines` and `patient_retrieval_guidelines`
variants. `llm_only` and `patient_retrieval` skip steps 4–5, so `llm_only`
serves as a baseline.

## Why the current prompt falls short

The original MED-COPILOT prompt
([upstream/templates/SOA_P_TEMPLATE.py](../med_copilot/upstream/templates/SOA_P_TEMPLATE.py))
received S/O/A and asked about symptom relief, medication, and management of a
known condition. Our version receives only S/O, so diagnostic bullets were
added. With at most four questions, diagnostic questions tend to use up the
budget.

Recorded Synthea Plans are mostly action types that neither version asks about
directly. Plan items from `data/release-v2/soap/corpus-soap.jsonl`:

| Category | Plan items in the data |
|---|---|
| Diagnostic tests and procedures | throat culture, electrocardiography, spirometry, cardiovascular stress testing |
| Drug choice and strength | penicillin V 250/500 mg, amoxicillin-clavulanate, ciprofloxacin, cephalexin, TMP-SMX, nitrofurantoin, lisinopril, hydrochlorothiazide, amlodipine, losartan |
| Severity or functional classification | New York Heart Association classification |
| Patient education | hypertension lifestyle education |
| Management plans for chronic conditions | COPD clinical management plan (a Synthea record artifact) |

Every recorded item fits one of these five categories. Many patients have no
Plan recorded (e.g. 905 of 1,397 viral pharyngitis corpus records). These are
excluded from Plan scoring, so "when not to treat" guidance is out of scope.

The guideline index includes a matching NICE guideline for every target
condition: NG84 (sore throat), NG79 (sinusitis), NG109 (lower UTI), NG106
(chronic heart failure), NG136 (hypertension), NG115 (COPD).

## Approach: add, don't reword

Keep every existing instruction unchanged and add new ones. This preserves
diagnostic behavior, keeps the existing test passing, and makes the diff easy
to compare.

Draft additions to the "Focus on questions about points like:" list:

```text
- Specific diagnostic tests or procedures recommended for the most likely diagnostic possibilities,
- First-line and alternative medication choices, including strength or dose, for each likely possibility,
- Severity or functional classification used to stage a suspected condition,
- Patient education and lifestyle advice for a suspected condition,
- Long-term management or care plans for suspected chronic conditions.
```

And one sentence after the list:

```text
Use at least one question to ask about concrete plan actions (tests, medications,
classification, education, or management plans) for the most likely possibilities.
```

Wording constraints:

- Do not use the word "assessment" anywhere in the prompt; the test rejects it.
  Write "NYHA functional classification", not "NYHA assessment".
- Do not name label items (penicillin V, throat culture, NYHA).
- Keep "at most 4", the `{"Key Questions": [...]}` example with four entries,
  and the `{conditions}` placeholder unchanged.

## Files

| File | Change |
|---|---|
| `med_copilot/prompts/diagnosis_plan.py` | Add bullets and the sentence to `KEY_QUESTIONS_TEMPLATE` |
| `med_copilot/docs/MED_COPILOT_diagnosis_plan_handoff.md` (line 32) | Mention concrete Plan actions in the description of the question step |
| `scripts/run_med_copilot.py` | None; it fills in the prompt and passes the raw reply to GraphRAG |
| `scripts/med_copilot_helpers.py` | None; `parse_guideline_questions` expects the `"Key Questions"` key |
| `tests/test_run_med_copilot.py` | None expected; `test_question_prompt_uses_so_for_diagnostic_and_management_evidence` should still pass |
| `med_copilot/upstream/` | None; frozen upstream source |

## Steps

### 0. Environment (one time)

```bash
brew install python@3.11 git-lfs
git lfs install && git lfs pull       # real guideline index and retrieval artifacts
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab wordnet omw-1.4
cp -n .env.example .env               # add your own OPENAI_API_KEY (no team key)
```

Done: Python 3.11 and git-lfs installed, LFS files pulled, `.venv/` created
with all requirements, and `.env` copied from the template (key still a
placeholder). The old `venv/` (Python 3.13, no packages) is unused. Steps 1–2 need no
API key. Steps 3–4 need an OpenAI API key: the guideline index was built with
`text-embedding-3-small`, and the question step uses the OpenAI client
directly, so it cannot be swapped for Claude without code changes.

### 1. Edit the prompt

Apply the additions above to `KEY_QUESTIONS_TEMPLATE`.

### 2. Offline tests (no API key)

```bash
python tests/test_run_med_copilot.py -k question_prompt
python tests/test_run_med_copilot.py
python tests/test_med_copilot_helpers.py
```

### 3. Compare old and new questions (cents of gpt-4o-mini)

For a handful of patients, send S/O through the old and the new template and
compare the questions side by side. Repeat 2–3 times, because the question
call uses the default temperature. Prefer **dev** records
(`data/release-v2/soap/dev-soap.jsonl`) over test records, so prompt
iteration does not look at test data.

Check:

- heart failure → tests and functional classification
- strep / viral pharyngitis → tests and first-line antibiotic
- hypertension → lifestyle education and drug choice
- one or two diagnostic questions remain
- at most four questions, valid JSON

If the questions barely change, make the added sentence stronger and repeat.

### 4. End-to-end check

Temporarily edit `experiments/configs/experiment.yaml`:

```yaml
patient_input: experiments/input/test_so.jsonl
patient_labels: experiments/input/test_labels.jsonl
selected_variants: [llm_only, guidelines]
selected_models: [gpt_4o_mini]
max_patients: 10
```

Run `python scripts/run_med_copilot.py` once with the new prompt and once with
the old prompt (`git stash` / `git stash pop`), under the same configuration.
Compare `guideline_questions`, `guideline_evidence`, Plan scores for
`guidelines`, and condition scores in the two `results.jsonl` files. With ten
eligible patients and temperature 1, treat this as a sanity check, not
evidence of improvement. Revert the YAML afterwards.

### 5. Documentation

Update the question-step description in
`med_copilot/docs/MED_COPILOT_diagnosis_plan_handoff.md`.

### 6. Commit and PR

Commit only the prompt file, the handoff doc, and this plan. Do not commit the
YAML or `experiments/output/`. Include 2–3 before/after question examples and
the score comparison in the PR description.

## Open PRs to watch (as of 2026-11-04)

None of these touches `diagnosis_plan.py`, so the prompt change itself does
not depend on them.

| PR | Effect on this plan |
|---|---|
| #33 Capture runner failures in run metadata | None. Wraps the question step in `pipeline_stage(...)`; prompt use is unchanged. |
| #34 Add full test input and labels | Changes the default input to the full test set with `max_patients: 10`. The file is sorted by condition, so those 10 are all COPD. If merged first, set the sample input paths explicitly in step 4. |
| #35 Reference patient corpus and retrieval index | Replaces the reference corpus for `patient_retrieval`. The 2026-10-26 run is no longer a valid baseline for retrieval variants; step 4 reruns both prompts and excludes retrieval variants. |
