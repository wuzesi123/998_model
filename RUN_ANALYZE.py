import argparse, json
from pathlib import Path
from chem_process_rag import ChemProcessPipeline, save_result

p=argparse.ArgumentParser(description="Video + reaction context + ORD-RAG chemical process interpretation")
p.add_argument("--video",required=True)
p.add_argument("--context",required=True)
p.add_argument("--ord-dir",default="data/ord")
p.add_argument("--knowledge",default="data/knowledge/seed_chemistry.jsonl")
p.add_argument("--checkpoint",default=None,help="Optional trained TCPT checkpoint. No random neural predictions are used when omitted.")
p.add_argument("--sample-sec",type=float,default=1.0)
p.add_argument("--top-k",type=int,default=5)
p.add_argument("--out",default="outputs/analysis_result.json")
a=p.parse_args()
pipe=ChemProcessPipeline(a.ord_dir,a.knowledge,a.sample_sec,a.checkpoint)
res=pipe.run(a.video,a.context,a.top_k); save_result(res,a.out)
print(json.dumps(res["report"],ensure_ascii=False,indent=2)); print(f"\nSaved: {Path(a.out).resolve()}")
