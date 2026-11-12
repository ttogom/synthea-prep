# GraphRAG RPM/TPM tuning results

No repository files were changed. All code and results are in this temporary directory. Only preparation and isolated guideline replay calls were run; no final diagnosis/Plan generation or scoring/judging.

Comparison and validation completed in 379.4 seconds, excluding the initial quota probe and harness setup.

Observed GPT-4o-mini response-header quota: 5,000 RPM and 2,000,000 TPM. Embedding quota: 5,000 RPM and 1,000,000 TPM.

Recommended tested setting: requests_per_minute: 12; tokens_per_minute: 200000 per preparation worker. Two workers therefore have combined local ceilings of 24 RPM / 400,000 TPM. Other clients do not share these local limiters, so maintain quota headroom.

## Comparison

Four same saved queries for the 2x2 factorial; two persistent sequential workers per condition. Initial limiter capacity was deliberately consumed. Timing excludes imports but includes query/context setup, streaming and initial artificial limiter debt. Treat these rates as stress-comparison results, not full-preparation throughput.

| RPM/worker | TPM/worker | Initial valid answers/min | Confirmation valid answers/min |
|---:|---:|---:|---:|
| 6 | 100,000 | 5.61 | not repeated |
| 12 | 100,000 | 8.18 | 7.91 |
| 6 | 200,000 | 7.82 | not repeated |
| 12 | 200,000 | 8.72 | 8.61 |

All 48 comparison HTTP attempts succeeded (200); all 24 streamed answers ended with stop. No retries or truncations were observed in this comparison. Model, temperature, prompts, token settings and retrieval settings were retained. The only config changes were the two limiter fields.

## Ten-patient full-preparation validation

| Measure | Previous auto/auto | Selected numeric limits |
|---|---:|---:|
| Wall time including model loading | 296.7 s | 110.9 s |
| Mean preparation service time/patient | 51.44 s | 14.96 s |
| Mean preparation, warm eight | 59.88 s | 14.68 s |

| Preparation step | Old seconds/patient | New seconds/patient |
|---|---:|---:|
| keyword_extraction | 3.147 | 2.690 |
| patient_retrieval_reranking_attribution | 1.741 | 1.685 |
| guideline_question_generation | 2.167 | 2.355 |
| guideline_retrieval | 44.366 | 8.214 |
| prompt_assembly_saving_other_overhead | 0.015 | 0.012 |

Mean RPM wait: 0.000151 s; TPM wait: 0.000015 s.

10/10 distinct patients prepared, both evidence components completed, ranking metadata and selected reference present, combined prompts saved, and no error artifacts. Final generation/scoring/judging counts remained zero.

A simple two-worker projection is 4.94 hours of preparation for 2,378 patients. This is a small-sample estimate, not an end-to-end deadline guarantee; generation, scoring, overlap and contention are not estimated here.

## Limitations and adversarial checks

- Initial comparison sampled three strata; confirmation corrected to one patient per stratum and full validation used all ten.
- Initial exhausted-bucket stress preconditions RPM then TPM when each is first acquired, adding conservative one-time debt. Use full-preparation timings for operational projections.
- Small sample and stochastic API latency do not establish a unique global optimum or clinical equivalence.
- One low-complexity confirmation case selected a different tenth context entity between conditions, with unchanged input token count. The initial factorial requests were byte-identical; do not interpret confirmation as a perfectly controlled clinical comparison.
- Quota headers are observed at test time; two workers have independent local buckets. Rebudget if workers or other shared API traffic change.
- The existing library warns about sequential reuse of limiters across asyncio loops. No failures were observed; no library code was changed.

The initial 16 factorial requests were byte-identical across conditions. The single variable confirmation context is recorded in audit.json; no limit-related change in evidence selection was demonstrated. The library warning and retrieval variability predate any production fix, since no source files were modified.

Full data: results.json; validation/summary.json; audit.json; per-condition worker JSON files. API credentials and authorization headers were not saved.
