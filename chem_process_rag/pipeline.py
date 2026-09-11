from __future__ import annotations
import json
from pathlib import Path
from .context.context_builder import load_context
from .visual.analyzer import VisualProcessAnalyzer
from .rag.ord_rag import ORDRAG
from .reasoning.semantic_reasoner import SemanticReasoner

class ChemProcessPipeline:
    def __init__(self, ord_dir="data/ord", seed_knowledge="data/knowledge/seed_chemistry.jsonl", sample_every_sec=1.0, tcpt_checkpoint=None):
        self.visual=VisualProcessAnalyzer(sample_every_sec=sample_every_sec,checkpoint=tcpt_checkpoint)
        self.rag=ORDRAG(ord_dir=ord_dir,seed_knowledge=seed_knowledge)
        self.reasoner=SemanticReasoner()
    def run(self, video, context, top_k=5):
        ctx=load_context(context); vis=self.visual.analyze_video(video); docs=self.rag.search(ctx,vis["summary"],top_k=top_k); rep=self.reasoner.infer(ctx,vis,docs)
        return {"context":ctx.to_dict(),"visual":vis,"rag_stats":self.rag.stats,"report":rep.to_dict()}

def save_result(result,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
