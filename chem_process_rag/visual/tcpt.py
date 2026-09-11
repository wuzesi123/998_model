from __future__ import annotations
import torch
from torch import nn

class TimeEncoding(nn.Module):
    def __init__(self,d_model:int):
        super().__init__(); self.net=nn.Sequential(nn.Linear(3,d_model),nn.SiLU(),nn.Linear(d_model,d_model))
    def forward(self,t):
        dt=torch.zeros_like(t); dt[:,1:]=t[:,1:]-t[:,:-1]
        x=torch.stack([t,torch.log1p(torch.clamp(t,min=0)),torch.log1p(torch.clamp(dt,min=0))],dim=-1)
        return self.net(x)

class Branch(nn.Module):
    def __init__(self,d_in,d_model=96,nhead=4,layers=2):
        super().__init__(); self.inp=nn.Linear(d_in,d_model); self.te=TimeEncoding(d_model)
        enc=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward=d_model*3,batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(enc,layers); self.norm=nn.LayerNorm(d_model)
    def forward(self,x,t):
        z=self.inp(x)+self.te(t); z=self.enc(z); return self.norm(z[:,-1])

class MultiScaleTCPT(nn.Module):
    def __init__(self,d_in,d_model=96,embedding_dim=128):
        super().__init__(); self.short=Branch(d_in,d_model); self.medium=Branch(d_in,d_model); self.global_=Branch(d_in,d_model)
        self.fuse=nn.Sequential(nn.Linear(d_model*3,256),nn.SiLU(),nn.Dropout(.1),nn.Linear(256,embedding_dim),nn.LayerNorm(embedding_dim))
        self.state_head=nn.Linear(embedding_dim,6) # optional learned observable-state head
    def forward(self,short_x,short_t,med_x,med_t,glob_x,glob_t):
        e=torch.cat([self.short(short_x,short_t),self.medium(med_x,med_t),self.global_(glob_x,glob_t)],-1)
        emb=self.fuse(e); return {"embedding":emb,"state_logits":self.state_head(emb)}
