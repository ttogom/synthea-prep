# Ten-patient preparation-only profile

Variant: patient_retrieval_guidelines. Two isolated CPU workers, two Torch threads each. No generation or scoring.

| Preparation step | Mean seconds/patient | Share |
|---|---:|---:|
| keyword_extraction | 2.690 | 17.98% |
| patient_retrieval_reranking_attribution | 1.685 | 11.27% |
| guideline_question_generation | 2.355 | 15.75% |
| guideline_retrieval | 8.214 | 54.92% |
| prompt_assembly_saving_other_overhead | 0.012 | 0.08% |

Total wall time including loading: 110.898 seconds.
Worker initialization times: [30.25273337499675, 30.253796208002314].
Mean local RPM wait: 0.000 seconds/patient.
Warm eight: {'mean_preparation_seconds': 14.67543870324971, 'mean_rpm_wait_seconds': 0.00018496374923415715}.

## Limitations

- Two workers and two CPU threads per worker match the main pilot; other parent-session work can still affect CPU/network timing.
- First request in each process incurs no steady-state RPM wait, making the overall ten-case average optimistic for a large cohort.
- HTTP streaming timings cover headers only; the remainder includes streamed generation and uninstrumented local work.
- Substep timings are observational, not a test of removing the limiter.
