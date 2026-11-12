# Four-worker preparation / Gemma overlap test

Completed in 58.5 seconds. Repository and parent-run artifacts were not modified; the private server used 127.0.0.1:11437 and was stopped afterward.

- Eight longest S/O cases from the previous ten-case set, two per persistent CPU worker.
- Four largest saved guideline-containing prompts from the 64-patient pilot, unchanged.
- Gemma3:4b, four simultaneous requests, 16,384 context tokens per slot, 768 maximum output tokens, production JSON schema and temperature.
- Preparation limits: 12 RPM / 200,000 TPM per worker. No diagnosis scoring or Plan judging.

## Outcomes

All eight preparations completed both evidence components and saved prompts/ranking metadata. All four Gemma responses parsed into three predictions. All recorded preparation HTTP requests succeeded. No worker crash, model failure, or prompt truncation was observed.

The server log confirmed four slots, 65,536 total context capacity and 16,384 per slot. All four requests were simultaneously in flight; active generation lasted 5.31 seconds. Gemma stayed resident throughout the preparation work.

Physical compressed-memory storage increased from 1.15 GiB to a peak of 10.92 GiB. It was roughly stable near 10.6–10.9 GiB during the later work samples, then decreased on cleanup.
There were 0 new swap-out pages. Swap used changed by -8.0 MiB. There were 231 swap-in pages and 49 generic page-outs; these are distinct counters.

Neither existing Ollama endpoint had a resident model at test start. The isolated test did not unload, reconfigure or terminate either existing service.

## Interpretation

The tested configuration completed successfully without disk swap growth. This supports using four preparation workers with the production Gemma allocation. Compression is substantial, so do not describe the result as proof of ample uncompressed headroom or guaranteed overnight completion.

## Limitations

- Short snapshot, not a long-run memory leak test.
- Four generation requests overlapped the beginning of preparation for approximately five seconds. The loaded model, four configured context slots, and associated allocations remained resident throughout preparation. Continuous generation for the full preparation period was not tested.
- System memory readings include unrelated applications and background work; the experiment does not attribute all compression to the tested workers.
- Large compression increase means passing cannot be interpreted as abundant uncompressed RAM headroom.
- Existing FNLLM sequential event-loop reuse warnings were observed; no failures occurred.
