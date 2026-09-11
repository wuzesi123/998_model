from __future__ import annotations
import math, re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable
from ..context.schema import RetrievedDocument

TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-\.]+")

def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "") if len(t) > 1]

@dataclass
class KnowledgeDoc:
    doc_id: str
    source: str
    title: str
    text: str
    metadata: dict

class TfidfIndex:
    def __init__(self, docs: Iterable[KnowledgeDoc] = ()): 
        self.docs = list(docs)
        self._rebuild()

    def _rebuild(self):
        n = max(len(self.docs), 1)
        self.df = defaultdict(int)
        self.doc_tf = []
        for d in self.docs:
            tf = Counter(tokenize(d.title + " " + d.text))
            self.doc_tf.append(tf)
            for t in tf: self.df[t] += 1
        self.idf = {t: math.log((1+n)/(1+df)) + 1.0 for t, df in self.df.items()}

    def add(self, docs: Iterable[KnowledgeDoc]):
        self.docs.extend(docs)
        self._rebuild()

    def search(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        qtf = Counter(tokenize(query))
        qv = {t: c*self.idf.get(t, math.log(1+len(self.docs))+1) for t,c in qtf.items()}
        qn = math.sqrt(sum(v*v for v in qv.values())) or 1.0
        scored = []
        for d, tf in zip(self.docs, self.doc_tf):
            dv = {t: c*self.idf.get(t, 1.0) for t,c in tf.items()}
            dn = math.sqrt(sum(v*v for v in dv.values())) or 1.0
            dot = sum(qv.get(t,0.0)*v for t,v in dv.items())
            score = dot/(qn*dn)
            if score > 0:
                scored.append(RetrievedDocument(d.doc_id,d.source,score,d.title,d.text,d.metadata))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]
