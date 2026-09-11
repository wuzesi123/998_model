Put official Open Reaction Database files here (.parquet, .pb.gz, .pb, .pbtxt, or protobuf JSON).
Use `python scripts/FETCH_ORD_DATASET.py ord_dataset-...` for a specific published dataset when `ord-schema[huggingface]` is installed.
The semantic RAG index deliberately excludes `outcomes` by default to avoid using future result/yield/product information as inference-time evidence.
