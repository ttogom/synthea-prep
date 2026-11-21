# Plan judge v2 pilot cases

> **Status: DRAFT. None of the 48 gold labels is approved.** Every case has
> `review.status: "draft"`. Labels are under human review: 11 flagged cases go to a clinical
> spot-check. `judge_plans_v2.py validate` refuses this file without `--allow-draft`, and
> reports made with that flag are marked `draft`.

`plan_judge_v2_pilot_cases.jsonl` holds 48 rank-1 cases. They are balanced across 8
categories and the 6 conditions whose reference Plans can be judged; COPD references are
record artifacts. The set is for **preliminary model selection only**, a pilot rather than a
definitive validation.

| Category | Cases | What it tests |
| --- | --- | --- |
| `exact` | 7 | Verbatim or near-verbatim reference actions |
| `valid_substitute` | 5 | Interchangeable alternatives (e.g. amoxicillin for penicillin V in strep, TMP-SMX for nitrofurantoin) |
| `more_specific` | 6 | Same action with dose, frequency, duration or purpose |
| `less_specific_or_strength` | 6 | Missing or different strength, "consider" with options, a partial substitute |
| `conditional` | 6 | "Do X if Y" where the reference has no prerequisite |
| `off_target` | 6 | No reference coverage: a different target, or appropriate care that omits the reference (must not be flagged harmful) |
| `placeholder` | 6 | Content-free plans |
| `correct_plus_harmful` | 6 | Coverage plus a harmful item (aspirin in children, amoxicillin with penicillin anaphylaxis, NSAIDs in heart failure) |

Sources:

- **33 cases are real rank-1 outputs** from `2026-11-12_022701PST__test_so__main`.
  Their `generated_actions` are the saved `predictions[0].plan` lists from that run's
  `results.jsonl` (`source.results_sha256`). They were checked against the judge's own reader,
  which strips items and drops blank ones. `source.original_plan` keeps the unmodified list
  when the two differ: `specific-strep` has an empty middle item. Do not rebuild them from
  `experiment_results.csv`: its `"; "` join merges or splits items in 5 of these cases.
- **15 cases are constructed:** 6 placeholders, 4 harmful cases, and 5 more where no clean
  real output existed.

S/O text comes from `experiments/input/test_so.jsonl`, and the references from
`test_labels.jsonl`.

## Review workflow

All gold labels start as `review.status: "draft"`, written by Claude for human review.
`review.notes` explains each label and marks contested calls with **REVIEW**, for example
whether a different-class antihypertensive is `full` or `partial`.

1. Alan reviews every case and edits `gold.matches` (one `full`/`partial`/`none` per
   reference action) and `gold.harmful` (generated item numbers). Then set
   `review.status: "reviewed"` and `review.reviewer`.
2. Rasika spot-checks the reviewed cases and sets `review.status: "spot_checked"` on the
   ones she checked.
3. `judge_plans_v2.py validate` runs without `--allow-draft` only when every case is
   `reviewed` or `spot_checked`.

Keep the file's case IDs stable. Pilot reports record the file's sha256, and `compare`
refuses reports made from different versions of the file.
