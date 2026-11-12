# Eight-patient four-worker preparation comparison

Eight matched patients, two per complexity group, drawn without reference to their timing. Four CPU workers with two CPU threads each. Same copied pipeline and numeric GraphRAG settings: 12 RPM / 200,000 TPM per worker. No branch or repository changes.

| Measure | Existing two-worker baseline | New four-worker run |
|---|---:|---:|
| Mean preparation duration per patient | 15.17 s | 14.85 s |
| Processing wall, excluding initialization | 61.73 s (reconstructed) | 32.12 s (measured) |
| Total wall including initialization | 91.98 s (reconstructed) | 53.53 s (measured) |

Processing-wall speedup: 1.922x. Concurrency-normalized service-throughput speedup: 2.043x. Startup-inclusive speedup: 1.719x.

| Preparation step | Two-worker seconds/patient | Four-worker seconds/patient |
|---|---:|---:|
| keyword_extraction | 2.829 | 3.294 |
| patient_retrieval_reranking_attribution | 1.681 | 0.969 |
| guideline_question_generation | 2.180 | 1.903 |
| guideline_retrieval | 8.472 | 8.681 |
| prompt_assembly_saving_other_overhead | 0.012 | 0.006 |

Peak sampled sum of worker RSS: 15.45 GiB (excludes other applications).

All eight distinct cases completed both evidence components, saved prompts/ranking metadata, and ran no final generation/scoring/judging. All recorded HTTP attempts returned 200.

## Limitations

- Baseline eight-case wall time is reconstructed from the existing two-worker per-patient durations and original assignments, excluding the two unselected cases; no new baseline run was performed.
- Four-worker measured processing wall includes tiny protocol and shutdown overhead; baseline reconstruction sums recorded service durations.
- Only two cases per new worker; fresh limiter buckets cannot establish long-run throttling behavior.
- API-generated keywords/questions/answers are stochastic. Different screening retry counts and provider latency can affect timing.
- This is preparation-only. Concurrent final generation/scoring interference was not reproduced; existing parent work was left untouched.
- Two-worker and four-worker runs occurred at different times, so throughput ratios are estimates rather than a controlled simultaneous comparison.
