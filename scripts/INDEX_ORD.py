import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from chem_process_rag.rag.ord_rag import ORDRAG
r=ORDRAG("data/ord","data/knowledge/seed_chemistry.jsonl")
print("Indexed documents:",r.stats["documents"])
if r.stats["errors"]:
    print("Errors:")
    for e in r.stats["errors"]: print(" -",e)
else: print("ORD/local knowledge index is ready.")
