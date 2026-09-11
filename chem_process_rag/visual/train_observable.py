from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np

STATE_NAMES=["QUIESCENT","CHANGE_ONSET","HIGH_ACTIVITY","ACTIVITY_DECAY","PLATEAU","HETEROGENEOUS"]

def _sample_indices(n,end,k,mode):
    end=max(0,min(n-1,end))
    if mode=="short": return np.arange(max(0,end-k+1),end+1)
    if mode=="medium":
        d=np.geomspace(1,max(end,1),min(k,end+1)); return np.unique(np.clip(np.round(end-d+1).astype(int),0,end))
    return np.unique(np.linspace(0,end,min(k,end+1)).round().astype(int))

class NPZObservableDataset:
    def __init__(self,files):
        self.exps=[]; self.items=[]
        for f in files:
            z=np.load(f,allow_pickle=True); x=z["features"].astype(np.float32); t=z["timestamps"].astype(np.float32); y=z["labels"].astype(np.int64)
            if not (len(x)==len(t)==len(y)): raise ValueError(f"length mismatch: {f}")
            self.exps.append((x,t,y,str(f)))
        for ei,(x,t,y,_) in enumerate(self.exps):
            for end in range(2,len(t)): self.items.append((ei,end))
    def __len__(self): return len(self.items)
    def item(self,i):
        ei,end=self.items[i]; x,t,y,_=self.exps[ei]
        ids=[_sample_indices(len(t),end,8,"short"),_sample_indices(len(t),end,12,"medium"),_sample_indices(len(t),end,16,"global")]
        out=[]
        for idx in ids:
            xx=x[idx]; tt=t[idx]-t[idx][0]; out.extend([xx,tt])
        return out,y[end]

def train(files,out="checkpoints/tcpt_observable.pt",epochs=20,lr=2e-4,device=None):
    import torch
    from torch import nn
    from .tcpt import MultiScaleTCPT
    ds=NPZObservableDataset(files)
    if not ds.exps: raise RuntimeError("No training NPZ files")
    dev=device or ("cuda" if torch.cuda.is_available() else "cpu")
    model=MultiScaleTCPT(ds.exps[0][0].shape[1]).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-3); ce=nn.CrossEntropyLoss()
    idx=list(range(len(ds)))
    for ep in range(1,epochs+1):
        random.shuffle(idx); model.train(); losses=[]
        for i in idx:
            vals,y=ds.item(i); tensors=[]
            for j,v in enumerate(vals):
                dtype=torch.float32; tensors.append(torch.tensor(v,dtype=dtype,device=dev)[None])
            yy=torch.tensor([y],dtype=torch.long,device=dev); pred=model(*tensors)["state_logits"]; loss=ce(pred,yy)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        print(f"ep={ep:02d} loss={np.mean(losses):.4f}")
    Path(out).parent.mkdir(parents=True,exist_ok=True); torch.save({"model":model.state_dict(),"state_names":STATE_NAMES,"feature_dim":ds.exps[0][0].shape[1]},out); print("saved",out)
