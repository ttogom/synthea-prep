# SOAP Conversion

Guide for Sumin: how to convert release-v1 records into SOAP format using a
language model. These instructions apply to both corpus records (which include a
DIAGNOSIS and TREATMENT section) and test/dev records (which do not).

---

## Corpus prompt (subjective + objective + assessment + plan)

Use this prompt for `corpus.jsonl` records. The record ends with `DIAGNOSIS` and
`TREATMENT` sections; ask the model to lift them into the SOAP fields.

```
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
```

Expected output shape:

```json
{
  "subjective": "...",
  "objective": "...",
  "assessment": "Acute viral pharyngitis (disorder)",
  "plan": ["Acetaminophen 325 MG Oral Tablet", "Rest"]
}
```

---

## Test/dev prompt (subjective + objective only)

Use this prompt for `test.jsonl` and `dev.jsonl` records. These records have
**no** DIAGNOSIS or TREATMENT section. The model must never name, suggest, or
rank a diagnosis.

```
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
```

Expected output shape:

```json
{
  "subjective": "...",
  "objective": "..."
}
```

---

## Settings

| Setting | Value |
|---------|-------|
| Model | `claude-sonnet-4-6` (pin this exact version; do not use an alias) |
| Temperature | 0 |
| Output caching | Cache by `id` field from the JSONL; skip re-running if a cached output already exists for that id |
| Max output tokens | 1024 (enough for the longest records) |

---

## After conversion: verification steps

1. **Rerun the text leak check** on the test/dev SOAP output:

   ```bash
   # Write the SOAP "objective" fields to a JSONL with the same id and label fields
   # as the original, then run check_text_leaks.py on each condition subset.
   # The "assessment" field in corpus output is expected to contain the diagnosis name
   # and will FAIL the check — that is correct and expected for corpus records.
   # Only run the check on test/dev output, not corpus.
   python3 serialization/check_text_leaks.py \
       <soap_test_output.jsonl> \
       data/pop10000-seed20260916__train-<sha> \
       --report data/release-v1/soap_leak_<condition>.md
   ```

2. **Compare ~20 conversions against their sources.** For each, verify:
   - Every sentence in `subjective` traces to a line in the source record.
   - Every value in `objective` matches a value in the source record (no rounded
     or inferred numbers).
   - No diagnosis word appears in `subjective` or `objective` for test/dev records.
   - `assessment` in corpus records matches the DIAGNOSIS line verbatim.
   - `plan` items in corpus records match the TREATMENT lines.

   A convenient way to do this comparison:

   ```bash
   python3 - <<'PY'
   import json, random
   recs  = [json.loads(l) for l in open("soap_test_output.jsonl")]
   src   = {r["id"]: r["text"] for r in (json.loads(l) for l in open("data/release-v1/test.jsonl"))}
   sample = random.sample(recs, min(20, len(recs)))
   for r in sample:
       print("=" * 60)
       print("SOURCE:\n", src[r["id"]][:800])
       print("SOAP:\n", json.dumps(r, indent=2)[:800])
   PY
   ```
