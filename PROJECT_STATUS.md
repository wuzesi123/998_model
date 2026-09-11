# V1 status

Implemented and wired end-to-end:

- [x] context_builder
- [x] typed ReactionContext
- [x] causal video feature extraction
- [x] short/medium/global TCPT architecture
- [x] optional TCPT checkpoint inference
- [x] optional TCPT observable-state trainer
- [x] lightweight TF-IDF RAG index
- [x] ORD protobuf/parquet/JSON ingestion adapter
- [x] official ORD Hugging Face fetch helper by dataset id
- [x] causal ORD field filter (outcomes excluded)
- [x] semantic_reasoner
- [x] CLI
- [x] Streamlit UI
- [x] self-contained synthetic demo
- [x] unit/smoke tests

Next research work should focus on training data and evaluation, not adding more heuristic labels.
