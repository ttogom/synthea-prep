# Plan judge v2 pilot cases

> **Status: PRELIMINARY. 47 of the 48 gold labels are preliminary pilot labels; 1 case is
> disputed.** The 47 uncontested labels were approved by Alan for preliminary pilot use, not
> independently clinically validated. Rasika spot-checked the 11 flagged cases and disputed one.
> They are marked `review.status: "preliminary"` (approved 2026-11-21) and are not final. The
> disputed case is `less-chf` (spot-check card 9): Rasika considers the plan's stress ECG
> equivalent to the reference Electrocardiographic procedure, so R1 would be `full`, not
> `partial`. Its gold label is **unchanged**. It is marked `review.status: "disputed"` and
> `validate` leaves it out of every agreement statistic.

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

## Review statuses

| `review.status` | Meaning | `validate` |
| --- | --- | --- |
| `draft` | Written by Claude, not yet approved | Refused unless `--allow-draft`; report marked `draft` |
| `preliminary` | Approved for preliminary pilot use; not independently clinically validated | Used; report marked `preliminary` |
| `reviewed` / `spot_checked` | Final human review | Used; report marked `reviewed` if every used case is final |
| `disputed` | Reviewers disagree; gold label kept as is | Not run; excluded from all statistics and listed under `excluded_cases` |

A `disputed` case must carry `review.dispute` (who raised it, their position, and the
resolution). Preliminary labels record `review.approved_by`, `review.approved_on` and `review.approved_for`.

## Review workflow

All gold labels started as `review.status: "draft"`, written by Claude for human review.
`review.notes` explains each label and marks contested calls with **REVIEW**, for example
whether a different-class antihypertensive is `full` or `partial`.

1. Alan reviews every case and edits `gold.matches` (one `full`/`partial`/`none` per
   reference action) and `gold.harmful` (generated item numbers). Labels approved for a
   preliminary run are set to `preliminary`; final ones to `reviewed`, with `review.reviewer`.
2. Rasika spot-checks the flagged cases and sets `review.status: "spot_checked"` on the
   ones she checked. A case where the reviewers disagree is set to `disputed` with
   `review.dispute`, and its gold label is left unchanged until the dispute is resolved.
3. `judge_plans_v2.py validate` runs without `--allow-draft` when no evaluated case is
   `draft`.

Keep the file's case IDs stable. Pilot reports record the file's sha256, and `compare`
refuses reports made from different versions of the file.
