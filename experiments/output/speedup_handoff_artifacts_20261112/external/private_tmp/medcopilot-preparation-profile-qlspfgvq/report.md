# Ten-patient preparation-only profile

Variant: patient_retrieval_guidelines. Two isolated CPU workers, two Torch threads each. No generation or scoring.

| Preparation step | Mean seconds/patient | Share |
|---|---:|---:|
| keyword_extraction | 3.147 | 6.12% |
| patient_retrieval_reranking_attribution | 1.741 | 3.38% |
| guideline_question_generation | 2.167 | 4.21% |
| guideline_retrieval | 44.366 | 86.26% |
| prompt_assembly_saving_other_overhead | 0.015 | 0.03% |

Total wall time including loading: 296.672 seconds.
Worker initialization times: [35.77948416599975, 35.779193499998655].
Mean local RPM wait: 34.737 seconds/patient.
Warm eight: {'mean_preparation_seconds': 59.87806965112486, 'mean_rpm_wait_seconds': 43.420775312624755}.

## Limitations

- Two workers and two CPU threads per worker match the main pilot; other parent-session work can still affect CPU/network timing.
- First request in each process incurs no steady-state RPM wait, making the overall ten-case average optimistic for a large cohort.
- HTTP streaming timings cover headers only; the remainder includes streamed generation and uninstrumented local work.
- Substep timings are observational, not a test of removing the limiter.

## Additional streaming diagnostic

One guideline-only replay from the same ten patients; excluded from the averages above.

- {"step": "load_config", "seconds": 0.00286995799979195}
- {"step": "load_tables", "seconds": 0.19624733299860964}
- {"step": "read_indexer_entities", "seconds": 0.18186016700201435}
- {"step": "read_indexer_reports", "seconds": 0.02675575000466779}
- {"step": "read_indexer_text_units", "seconds": 0.12626791699585738}
- {"step": "read_indexer_relationships", "seconds": 0.14977400000498164}
- {"step": "build_search_engine", "seconds": 0.150954290998925}
- {"step": "limiter", "seconds": 7.167000148911029e-06}
- {"step": "http_full", "path": "/v1/embeddings", "seconds": 0.31619970799511066, "headers_seconds": 0.3161976249975851, "stream": false, "status": 200}
- {"step": "limiter", "seconds": 2.4084001779556274e-05}
- {"step": "http_full", "path": "/v1/chat/completions", "seconds": 5.643935332998808, "headers_seconds": 0.5923858329988434, "stream": true, "status": 200}

GraphRAG total: 6.992 seconds.
