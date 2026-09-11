from __future__ import annotations
import gzip, json, hashlib
from pathlib import Path
from typing import Any, Iterable
from google.protobuf.json_format import MessageToDict
from .text_index import KnowledgeDoc, TfidfIndex
from ..context.schema import RetrievedDocument, ReactionContext
from ..context.context_builder import context_to_query


def _flatten(obj: Any, prefix: str = "") -> list[str]:
    out=[]
    if isinstance(obj, dict):
        for k,v in obj.items():
            p=f"{prefix}.{k}" if prefix else str(k)
            out.extend(_flatten(v,p))
    elif isinstance(obj, list):
        for v in obj: out.extend(_flatten(v,prefix))
    elif obj is not None:
        out.append(f"{prefix}={obj}")
    return out


def _causal_filter_reaction_dict(r: dict) -> dict:
    # Keep only information that could reasonably be known before/during inference.
    allowed = {"reaction_id","identifiers","inputs","setup","conditions","notes","observations","provenance"}
    return {k:v for k,v in r.items() if k in allowed}


def reaction_dict_to_doc(r: dict, source: str) -> KnowledgeDoc:
    rr = _causal_filter_reaction_dict(r)
    rid = str(rr.get("reaction_id") or rr.get("reactionId") or hashlib.sha1(json.dumps(rr,sort_keys=True,default=str).encode()).hexdigest()[:12])
    identifiers = rr.get("identifiers", [])
    title_parts=[]
    for x in identifiers[:4]:
        if isinstance(x,dict) and x.get("value"): title_parts.append(str(x["value"]))
    title = " | ".join(title_parts) or rid
    text = "; ".join(_flatten(rr))
    return KnowledgeDoc(rid, source, title, text, {"reaction_id":rid,"causal_fields_only":True})


class ORDRAG:
    def __init__(self, ord_dir: str | Path | None = None, seed_knowledge: str | Path | None = None):
        self.docs: list[KnowledgeDoc] = []
        self.index = TfidfIndex([])
        self.load_errors: list[str] = []
        if seed_knowledge:
            self.load_jsonl(seed_knowledge, source="seed_knowledge")
        if ord_dir:
            self.load_ord_directory(ord_dir)

    def load_jsonl(self, path: str | Path, source: str = "jsonl"):
        p=Path(path)
        if not p.exists(): return
        docs=[]
        for i,line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            if not line.strip(): continue
            x=json.loads(line)
            docs.append(KnowledgeDoc(str(x.get("id",i)),x.get("source",source),x.get("title",f"doc-{i}"),x.get("text",""),x.get("metadata",{})))
        self.docs.extend(docs); self.index.add(docs)

    def _load_ord_json(self, path: Path):
        raw=json.loads(path.read_text(encoding="utf-8"))
        reactions = raw.get("reactions",[]) if isinstance(raw,dict) else raw
        if not isinstance(reactions,list): return
        docs=[reaction_dict_to_doc(r, f"ORD:{path.name}") for r in reactions if isinstance(r,dict)]
        self.docs.extend(docs); self.index.add(docs)

    def _load_ord_schema_file(self, path: Path):
        try:
            from ord_schema import datasets
        except Exception as exc:
            self.load_errors.append(f"{path.name}: ord-schema unavailable ({exc})")
            return
        try:
            ds = datasets.load_dataset(path)
            docs=[]
            for reaction in ds.reactions if hasattr(ds,"reactions") else ds:
                rd = MessageToDict(reaction, preserving_proto_field_name=True)
                docs.append(reaction_dict_to_doc(rd, f"ORD:{path.name}"))
            self.docs.extend(docs); self.index.add(docs)
        except Exception as exc:
            self.load_errors.append(f"{path.name}: {exc}")

    def load_ord_directory(self, ord_dir: str | Path):
        root=Path(ord_dir); root.mkdir(parents=True,exist_ok=True)
        for p in sorted(root.rglob("*")):
            if not p.is_file(): continue
            name=p.name.lower()
            if name.endswith(".json"):
                try:self._load_ord_json(p)
                except Exception as exc:self.load_errors.append(f"{p.name}: {exc}")
            elif name.endswith((".pb.gz",".parquet",".pb",".pbtxt",".txtpb")):
                self._load_ord_schema_file(p)

    def search(self, ctx: ReactionContext, visual_summary: dict | None = None, top_k: int = 5) -> list[RetrievedDocument]:
        return self.index.search(context_to_query(ctx,visual_summary),top_k=top_k)

    @property
    def stats(self):
        return {"documents":len(self.docs),"errors":self.load_errors}
