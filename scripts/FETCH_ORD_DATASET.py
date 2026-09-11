"""Download one published ORD dataset by official ord_dataset-* id.
Usage: python scripts/FETCH_ORD_DATASET.py ord_dataset-xxxxxxxx
Requires: pip install "ord-schema[huggingface]"
"""
import argparse, shutil
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("dataset_id"); p.add_argument("--out",default="data/ord"); a=p.parse_args()
try:
    from ord_schema import huggingface
except Exception as exc:
    raise SystemExit('Install ORD support first: pip install "ord-schema[huggingface]"') from exc
src=Path(huggingface.fetch_dataset(a.dataset_id)); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); dst=out/src.name
if src.resolve()!=dst.resolve(): shutil.copy2(src,dst)
print(dst.resolve())
