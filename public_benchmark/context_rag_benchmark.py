from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chem_process_rag.context.schema import ReactionContext, Procedure, Conditions
from chem_process_rag.rag.ord_rag import ORDRAG

IMAGE_EXTS={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}


def _norm_path(s: str) -> str:
    return str(s).replace('\\','/').lstrip('./').lower()


def discover_context_map(root: str|Path) -> tuple[dict[str,str], dict]:
    """Find an explicit image->context mapping.

    Accepted machine-readable patterns:
    - CSV with image/file/path column and context/prompt/protocol/task/description column
    - JSON/JSONL objects with those field pairs
    We intentionally do NOT derive context from class-label directory names, because that can leak labels.
    """
    root=Path(root)
    image_keys=['image','image_path','file','filename','path','img']
    text_keys=['context','prompt','protocol','task','description','experiment_context','text']
    mapping={}; sources=[]

    def add_obj(obj: dict, source: str):
        nonlocal mapping
        low={str(k).lower():v for k,v in obj.items()}
        ik=next((k for k in image_keys if k in low),None)
        tk=next((k for k in text_keys if k in low),None)
        if ik and tk and low.get(ik) and low.get(tk):
            mapping[_norm_path(str(low[ik]))]=str(low[tk])
            return True
        return False

    for p in root.rglob('*'):
        if not p.is_file(): continue
        try:
            if p.suffix.lower()=='.csv' and p.stat().st_size<50_000_000:
                with p.open('r',encoding='utf-8-sig',errors='ignore',newline='') as f:
                    r=csv.DictReader(f)
                    hit=0
                    for row in r:
                        if add_obj(row,str(p)): hit+=1
                    if hit: sources.append({'file':str(p),'rows':hit})
            elif p.suffix.lower()=='.json' and p.stat().st_size<50_000_000:
                obj=json.loads(p.read_text(encoding='utf-8',errors='ignore'))
                rows=obj if isinstance(obj,list) else obj.get('items',[]) if isinstance(obj,dict) and isinstance(obj.get('items'),list) else [obj] if isinstance(obj,dict) else []
                hit=sum(1 for x in rows if isinstance(x,dict) and add_obj(x,str(p)))
                if hit: sources.append({'file':str(p),'rows':hit})
            elif p.suffix.lower()=='.jsonl' and p.stat().st_size<50_000_000:
                hit=0
                for line in p.read_text(encoding='utf-8',errors='ignore').splitlines():
                    try: x=json.loads(line)
                    except Exception: continue
                    if isinstance(x,dict) and add_obj(x,str(p)): hit+=1
                if hit: sources.append({'file':str(p),'rows':hit})
        except Exception:
            continue
    return mapping, {'n_mappings':len(mapping),'sources':sources,'provenance':'explicit_machine_readable_only'}


def _match_context(image: Path, root: Path, mapping: dict[str,str]) -> str|None:
    rel=_norm_path(str(image.relative_to(root))) if image.is_relative_to(root) else _norm_path(str(image))
    name=_norm_path(image.name)
    stem=_norm_path(image.stem)
    for k in [rel,name,stem]:
        if k in mapping: return mapping[k]
    # Suffix match only if unambiguous.
    candidates=[v for k,v in mapping.items() if k.endswith('/'+name) or k==name]
    if len(set(candidates))==1: return candidates[0]
    return None


def audit_context_dataset(root: str|Path, seed_knowledge: str|Path, ord_dir: str|Path|None, out_dir: str|Path) -> dict:
    root=Path(root); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    mapping,meta=discover_context_map(root)
    images=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    matched=[]
    for p in images:
        c=_match_context(p,root,mapping)
        if c: matched.append((p,c))
    rag=ORDRAG(ord_dir=ord_dir,seed_knowledge=seed_knowledge)
    result={
        'dataset_root':str(root),
        'n_images':len(images),
        'context_map_audit':meta,
        'n_images_with_explicit_context':len(matched),
        'coverage':float(len(matched)/len(images)) if images else 0.0,
        'rag_stats':rag.stats,
        'quantitative_context_benchmark_ready':bool(len(matched)>=100 and len(matched)>=0.5*max(len(images),1)),
        'scientific_policy':'No directory/class-name-derived context is treated as explicit experimental context.',
    }
    if not result['quantitative_context_benchmark_ready']:
        result['status']='AUDIT_ONLY'
        result['reason']='Public archive did not expose enough explicit per-image machine-readable context for a leakage-safe quantitative +Context/+RAG benchmark.'
    else:
        result['status']='READY_FOR_FUSION_BENCHMARK'
        # Save a leakage-safe explicit-context manifest for a later/follow-up classifier run.
        manifest=[{'image':str(p),'context':c} for p,c in matched]
        (out/'EXPLICIT_CONTEXT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (out/'CONTEXT_RAG_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def _hash_text(text: str, dim: int=256) -> np.ndarray:
    v=np.zeros(dim,np.float32)
    toks=re.findall(r'[A-Za-z0-9_+\-.]+',text.lower())
    for tok in toks:
        h=int(hashlib.blake2b(tok.encode(),digest_size=8).hexdigest(),16)
        idx=h%dim; sign=1.0 if ((h>>8)&1)==0 else -1.0
        v[idx]+=sign
    n=float(np.linalg.norm(v))
    return v/(n if n>1e-8 else 1.0)


def _label_path(image: Path) -> Path|None:
    parts=list(image.parts)
    # Replace nearest 'images' component with 'labels'.
    for i in range(len(parts)-1,-1,-1):
        if parts[i].lower()=='images':
            q=Path(*parts[:i],'labels',*parts[i+1:]).with_suffix('.txt')
            if q.exists(): return q
    q=image.with_suffix('.txt')
    return q if q.exists() else None


def _read_yolo_boxes(image: Path):
    lab=_label_path(image)
    if not lab: return []
    out=[]
    for line in lab.read_text(encoding='utf-8',errors='ignore').splitlines():
        vals=line.split()
        if len(vals)<5: continue
        try:
            c=int(float(vals[0])); x,y,w,h=map(float,vals[1:5]); out.append((c,x,y,w,h))
        except Exception: pass
    return out


def _crop_box(img, box, pad=0.03):
    from PIL import Image
    c,x,y,w,h=box; W,H=img.size
    x0=max(0,(x-w/2-pad*w)*W); y0=max(0,(y-h/2-pad*h)*H)
    x1=min(W,(x+w/2+pad*w)*W); y1=min(H,(y+h/2+pad*h)*H)
    if x1-x0<4 or y1-y0<4: return None
    return img.crop((int(x0),int(y0),int(x1),int(y1)))


def _official_image_splits(root: Path):
    from public_benchmark.heinsight_yolo import discover_yolo_dataset, _images, _group_id
    info=discover_yolo_dataset(root)
    train_all=_images(info['train_images']); test=_images(info['heldout_images'])
    groups={}
    for p in train_all: groups.setdefault(_group_id(p),[]).append(p)
    keys=sorted(groups)
    random.Random(7).shuffle(keys)
    if len(keys)>=3:
        nv=max(1,int(round(.12*len(keys)))); vk=set(keys[:nv])
        train=[p for k in keys if k not in vk for p in groups[k]]; val=[p for k in keys if k in vk for p in groups[k]]
    else:
        rng=random.Random(7); rng.shuffle(train_all); nv=max(1,int(.12*len(train_all))); val=train_all[:nv]; train=train_all[nv:]
    return train,val,test,info.get('names')


def _extract_crop_embeddings(samples, cache_path: Path):
    import torch
    from PIL import Image
    from torchvision.models import resnet18, ResNet18_Weights
    if cache_path.exists():
        z=np.load(cache_path,allow_pickle=True)
        return z['x'].astype(np.float32),z['y'].astype(np.int64),z['contexts'].tolist(),z['paths'].tolist()
    weights=ResNet18_Weights.DEFAULT; model=resnet18(weights=weights); model.fc=torch.nn.Identity(); model.eval()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(dev)
    prep=weights.transforms(); xs=[];ys=[];cs=[];ps=[]; batch=[]; meta=[]
    def flush():
        nonlocal batch,meta
        if not batch:return
        ten=torch.stack(batch).to(dev)
        with torch.no_grad(), torch.amp.autocast(device_type='cuda',dtype=torch.float16,enabled=dev.type=='cuda'):
            emb=model(ten).float().cpu().numpy()
        for e,m in zip(emb,meta):
            xs.append(e);ys.append(m[0]);cs.append(m[1]);ps.append(m[2])
        batch=[];meta=[]
    for image,label,ctx in samples:
        try:
            im=Image.open(image).convert('RGB')
            # One sample tuple may carry a specific box as 4th element.
            box=label[1] if isinstance(label,tuple) else None
            cls=label[0] if isinstance(label,tuple) else int(label)
            crop=_crop_box(im,(cls,*box)) if box is not None else im
            if crop is None: continue
            batch.append(prep(crop));meta.append((cls,ctx,str(image)))
            if len(batch)>=128:flush()
        except Exception:continue
    flush()
    x=np.asarray(xs,np.float32);y=np.asarray(ys,np.int64)
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache_path,x=x,y=y,contexts=np.asarray(cs,dtype=object),paths=np.asarray(ps,dtype=object))
    return x,y,cs,ps


def _make_samples(images, root, mapping):
    out=[]
    for p in images:
        ctx=_match_context(p,root,mapping)
        if not ctx: continue
        for c,x,y,w,h in _read_yolo_boxes(p): out.append((p,(c,(x,y,w,h)),ctx))
    return out


def _accuracy_f1(y,p,n_classes):
    y=np.asarray(y);p=np.asarray(p);acc=float(np.mean(y==p)) if len(y) else 0.0
    f1s=[]
    for c in range(n_classes):
        tp=int(np.sum((y==c)&(p==c)));fp=int(np.sum((y!=c)&(p==c)));fn=int(np.sum((y==c)&(p!=c)))
        prec=tp/max(tp+fp,1);rec=tp/max(tp+fn,1);f1s.append(2*prec*rec/max(prec+rec,1e-12))
    return {'accuracy':acc,'macro_f1':float(np.mean(f1s)),'per_class_f1':[float(x) for x in f1s]}


def _train_fusion_head(xtr,ytr,xv,yv,xt,yt,n_classes,epochs=40,seed=7):
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset,DataLoader
    torch.manual_seed(seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean=xtr.mean(0,keepdims=True);std=xtr.std(0,keepdims=True);std=np.where(std>1e-6,std,1.0)
    norm=lambda x:((x-mean)/std).astype(np.float32)
    tr=TensorDataset(torch.from_numpy(norm(xtr)),torch.from_numpy(ytr));loader=DataLoader(tr,batch_size=256,shuffle=True)
    model=nn.Sequential(nn.Linear(xtr.shape[1],256),nn.GELU(),nn.LayerNorm(256),nn.Dropout(.15),nn.Linear(256,n_classes)).to(dev)
    counts=np.bincount(ytr,minlength=n_classes).astype(np.float32);w=np.sqrt(max(len(ytr),1)/np.maximum(counts,1));w=w/max(w.mean(),1e-6);w=np.clip(w,.6,2.0)
    lossfn=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=dev));opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    best=None;bestf=-1;stale=0
    def pred(x):
        model.eval();out=[]
        with torch.no_grad():
            for i in range(0,len(x),1024):out.append(model(torch.from_numpy(norm(x[i:i+1024])).to(dev)).argmax(-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0,np.int64)
    for ep in range(1,epochs+1):
        model.train()
        for xb,yb in loader:
            xb,yb=xb.to(dev),yb.to(dev);loss=lossfn(model(xb),yb);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        vm=_accuracy_f1(yv,pred(xv),n_classes)
        if vm['macro_f1']>bestf+1e-4:
            bestf=vm['macro_f1'];best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:
            stale+=1
            if stale>=7:break
    if best:model.load_state_dict(best)
    return {'val':_accuracy_f1(yv,pred(xv),n_classes),'test':_accuracy_f1(yt,pred(xt),n_classes)}


def run_context_fusion_if_ready(root: str|Path, seed_knowledge: str|Path, ord_dir: str|Path|None, out_dir: str|Path, min_coverage=.5) -> dict:
    root=Path(root);out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    mapping,meta=discover_context_map(root)
    try: train_imgs,val_imgs,test_imgs,names=_official_image_splits(root)
    except Exception as e:
        res={'status':'SKIPPED','reason':f'YOLO dataset discovery failed: {e}','context_map_audit':meta};(out/'CONTEXT_FUSION_RESULT.json').write_text(json.dumps(res,indent=2),encoding='utf-8');return res
    all_imgs=train_imgs+val_imgs+test_imgs;coverage=sum(_match_context(p,root,mapping) is not None for p in all_imgs)/max(len(all_imgs),1)
    if len(mapping)<10 or coverage<min_coverage:
        res={'status':'SKIPPED','reason':'Insufficient explicit image-level context; refusing to derive context from class/folder labels.','coverage':coverage,'context_map_audit':meta}
        (out/'CONTEXT_FUSION_RESULT.json').write_text(json.dumps(res,indent=2),encoding='utf-8');return res
    splits=[]
    for name_,imgs in [('train',train_imgs),('val',val_imgs),('test',test_imgs)]:
        samples=_make_samples(imgs,root,mapping);x,y,ctx,paths=_extract_crop_embeddings(samples,out/f'{name_}_crop_embeddings.npz');splits.append((x,y,ctx,paths))
    (xtr,ytr,ctr,_),(xv,yv,cv,_),(xt,yt,ct,_)=splits
    if min(len(ytr),len(yv),len(yt))<20:
        res={'status':'SKIPPED','reason':'Too few context-matched labeled crops after filtering.','counts':[len(ytr),len(yv),len(yt)]};(out/'CONTEXT_FUSION_RESULT.json').write_text(json.dumps(res,indent=2),encoding='utf-8');return res
    n_classes=int(max(ytr.max(),yv.max(),yt.max())+1)
    rag=ORDRAG(ord_dir=ord_dir,seed_knowledge=seed_knowledge)
    def text_feats(contexts,with_rag):
        rows=[]
        for i,text in enumerate(contexts):
            c=_hash_text(text,256)
            if with_rag:
                ctx=ReactionContext(experiment_id=str(i),procedure=Procedure(experiment_type='unknown',current_step=text,objective=text),conditions=Conditions())
                docs=rag.search(ctx,top_k=3);rtext=' '.join(d.title+' '+d.text for d in docs);r=_hash_text(rtext,256);rows.append(np.concatenate([c,r]))
            else:rows.append(c)
        return np.asarray(rows,np.float32)
    results={}
    results['VisualOnly']=_train_fusion_head(xtr,ytr,xv,yv,xt,yt,n_classes)
    c_tr,c_v,c_t=text_feats(ctr,False),text_feats(cv,False),text_feats(ct,False)
    results['VisualPlusContext']=_train_fusion_head(np.concatenate([xtr,c_tr],1),ytr,np.concatenate([xv,c_v],1),yv,np.concatenate([xt,c_t],1),yt,n_classes)
    r_tr,r_v,r_t=text_feats(ctr,True),text_feats(cv,True),text_feats(ct,True)
    results['VisualPlusContextPlusRAG']=_train_fusion_head(np.concatenate([xtr,r_tr],1),ytr,np.concatenate([xv,r_v],1),yv,np.concatenate([xt,r_t],1),yt,n_classes)
    res={'status':'COMPLETED','coverage':coverage,'counts':{'train_crops':len(ytr),'val_crops':len(yv),'test_crops':len(yt)},'class_names':names,'rag_stats':rag.stats,'results':results,'note':'Context uses only explicit machine-readable image->context mappings; RAG query never includes ground-truth class label.'}
    (out/'CONTEXT_FUSION_RESULT.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
    return res
