# Smoke test results

Validated in the build environment:

- Python compilation: PASS
- Unit tests: 2/2 PASS
- Synthetic video generation: PASS
- OpenCV visual evidence extraction: PASS
- ReactionContext parsing/query construction: PASS
- Local RAG retrieval: PASS
- ORD JSON ingestion: PASS
- Causal ORD filter: PASS (`outcomes` and a deliberately inserted future product were absent from indexed text)
- Semantic reasoner: PASS
- Optional TCPT training: PASS (1-epoch smoke checkpoint)
- Optional TCPT checkpoint inference: PASS
- End-to-end `RUN_DEMO.py`: PASS

Example demo output separates the two layers:

- visual state: `VISUAL_PLATEAU_OR_STABLE`
- chemical interpretation: `crystallization_or_nucleation`

The demo is an engineering smoke test, not a scientific accuracy benchmark.
