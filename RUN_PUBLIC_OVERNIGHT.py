from __future__ import annotations

"""One-click public-data overnight benchmark for ChemProcessRAG.

Default run:
    python RUN_PUBLIC_OVERNIGHT.py

It is intentionally fault-tolerant: if one public source changes format or a download
fails, the other benchmarks continue and the failure is recorded in the final JSON.
"""

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

PUBLIC_REQ=ROOT/'requirements-public.txt'


def _missing_modules():
    mods=['requests','yaml','pandas','openpyxl','torch','torchvision','ultralytics','cv2']
    return [m for m in mods if importlib.util.find_spec(m) is None]


def _auto_install():
    miss=_missing_modules()
    if not miss:return
    print('[deps] missing:',miss)
    print('[deps] installing requirements-public.txt into the current Python environment...')
    subprocess.check_call([sys.executable,'-m','pip','install','-r',str(PUBLIC_REQ)])


class Tee:
    def __init__(self,path):
        self.term=sys.__stdout__;self.f=open(path,'a',encoding='utf-8',buffering=1)
    def write(self,s):self.term.write(s);self.f.write(s)
    def flush(self):self.term.flush();self.f.flush()


def _result_or_run(path: Path, force: bool, fn):
    if path.exists() and not force:
        print('[resume] reuse',path)
        return json.loads(path.read_text(encoding='utf-8'))
    return fn()


def _stage(name, summary, fn):
    print('\n'+'='*88);print('STAGE:',name);print('='*88)
    try:
        r=fn();summary['stages'][name]={'status':'COMPLETED','result':r};return r
    except Exception as e:
        traceback.print_exc();err={'status':'FAILED','error':f'{type(e).__name__}: {e}'};summary['stages'][name]=err;return None


def _extract_roots(manifest):
    roots=[]
    for f in manifest.get('files',[]):
        q=f.get('extracted_to') or f.get('path')
        if q:roots.append(Path(q))
    return roots


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--force',action='store_true',help='rerun completed stages')
    ap.add_argument('--no-auto-install',action='store_true')
    ap.add_argument('--skip-context',action='store_true')
    ap.add_argument('--yolo-epochs',type=int,default=60)
    ap.add_argument('--tcpt-epochs',type=int,default=30)
    ap.add_argument('--seed',type=int,default=7)
    args=ap.parse_args()

    if not args.no_auto_install:_auto_install()
    from public_benchmark.downloads import download_zenodo_record, recursively_extract_archives
    from public_benchmark.heinsight_yolo import train_and_test_heinsight
    from public_benchmark.crystalcv_prepare import prepare_crystalcv
    from public_benchmark.crystalcv_benchmark import run_crystalcv_benchmark
    from public_benchmark.context_rag_benchmark import audit_context_dataset, run_context_fusion_if_ready

    result_root=ROOT/'results'/'public_nightly';result_root.mkdir(parents=True,exist_ok=True)
    sys.stdout=Tee(result_root/'RUN_PUBLIC_OVERNIGHT.log');sys.stderr=sys.stdout
    data_root=ROOT/'data'/'public';data_root.mkdir(parents=True,exist_ok=True)
    summary={'version':'ChemProcessRAG_PUBLIC_NIGHTLY_V2','started_at':datetime.now().isoformat(),'scientific_policy':[
        'HeinSight4 uses manual phase bounding-box labels.',
        'CrystalCV temporal benchmark uses observed tracking values, not adapter-derived stage/progress labels.',
        'CrystalCV duplicate processed/raw representations are collapsed to one physical experiment before splitting.',
        'Context/RAG quantitative evaluation uses only explicit machine-readable image->context mappings; otherwise it is skipped rather than inventing context.',
        'RAG queries never receive the ground-truth visual class label.',
    ],'stages':{}}

    print('ChemProcessRAG PUBLIC NIGHTLY V2')
    print('Python:',sys.version)
    try:
        import torch
        print('Torch:',torch.__version__,'CUDA:',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
    except Exception as e:print('Torch check failed',e)

    # --------------------- HeinSight4 manual visual labels ---------------------
    hs_raw=data_root/'heinsight4_v2'
    hs_manifest=_stage('download_heinsight4',summary,lambda:_result_or_run(hs_raw/'ZENODO_DOWNLOAD_MANIFEST.json',args.force,lambda:download_zenodo_record(
        15605098,hs_raw,include_substrings=['chemical_dataset_v2'],exclude_substrings=['model'],extract=True)))
    if hs_manifest:
        for r in _extract_roots(hs_manifest):
            if r.is_dir():recursively_extract_archives(r)
        _stage('train_test_heinsight4_visual',summary,lambda:_result_or_run(result_root/'heinsight4'/'HEINSIGHT4_RESULT.json',args.force,lambda:train_and_test_heinsight(
            hs_raw,result_root/'heinsight4',epochs=args.yolo_epochs,imgsz=640,batch=24,workers=4,seed=args.seed,device='0' if __import__('torch').cuda.is_available() else 'cpu')))

    # --------------------- CrystalCV V2 leakage-free temporal ---------------------
    # The paper reports 129 bulk crystallization experiments. The source-data Zenodo
    # contains the raw case-study data, while the archived software/repository contains
    # author-provided Pre_Processed trajectories. We scan both and select one validated
    # representation per physical experiment.
    cv_raw=data_root/'crystalcv'
    cv_manifest=_stage('download_crystalcv_source',summary,lambda:_result_or_run(cv_raw/'ZENODO_DOWNLOAD_MANIFEST.json',args.force,lambda:download_zenodo_record(
        19140131,cv_raw,extract=True)))
    cv_software=data_root/'crystalcv_software'
    cv_sw_manifest=_stage('download_crystalcv_software',summary,lambda:_result_or_run(cv_software/'ZENODO_DOWNLOAD_MANIFEST.json',args.force,lambda:download_zenodo_record(
        20655010,cv_software,extract=True)))
    prepared=result_root/'crystalcv'/'prepared_v2'
    cv_roots=[]
    for man in [cv_manifest,cv_sw_manifest]:
        if man:
            for r in _extract_roots(man):
                if r.is_dir():
                    recursively_extract_archives(r)
                    cv_roots.append(r)
    if cv_roots:
        _stage('prepare_crystalcv_physical_experiments_v2',summary,lambda:_result_or_run(prepared/'CRYSTALCV_PREPARE_MANIFEST_V2.json',args.force,lambda:prepare_crystalcv(
            cv_roots,prepared,min_rows=48,min_physical_experiments=30)))
        _stage('train_test_crystalcv_temporal_v2',summary,lambda:_result_or_run(result_root/'crystalcv'/'benchmark_v2'/'CRYSTALCV_TEMPORAL_RESULT_V2.json',args.force,lambda:run_crystalcv_benchmark(
            prepared,result_root/'crystalcv'/'benchmark_v2',epochs=args.tcpt_epochs,batch=512,workers=4,seed=args.seed,d_model=192,layers=3,heads=6)))

    # --------------------- 2026 context-aware external dataset ---------------------
    if not args.skip_context:
        ca_raw=data_root/'context_aware_2026'
        ca_manifest=_stage('download_context_aware_2026',summary,lambda:_result_or_run(ca_raw/'ZENODO_DOWNLOAD_MANIFEST.json',args.force,lambda:download_zenodo_record(
            17436705,ca_raw,extract=True)))
        if ca_manifest:
            for r in _extract_roots(ca_manifest):
                if r.is_dir():recursively_extract_archives(r)
            seed_knowledge=ROOT/'data'/'knowledge'/'seed_chemistry.jsonl';ord_dir=ROOT/'data'/'ord'
            _stage('audit_context_and_rag_support',summary,lambda:audit_context_dataset(ca_raw,seed_knowledge,ord_dir,result_root/'context_aware'))
            _stage('train_test_context_rag_fusion_if_supported',summary,lambda:_result_or_run(result_root/'context_aware'/'CONTEXT_FUSION_RESULT.json',args.force,lambda:run_context_fusion_if_ready(
                ca_raw,seed_knowledge,ord_dir,result_root/'context_aware')))

    summary['finished_at']=datetime.now().isoformat()
    (result_root/'PUBLIC_NIGHTLY_SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')

    # Compact CSV/console summary.
    rows=[]
    hs=summary['stages'].get('train_test_heinsight4_visual',{}).get('result') or {}
    if hs:
        m=hs.get('heldout_metrics',{});rows.append({'benchmark':'HeinSight4 visual','metric':'mAP50-95','value':m.get('map50_95',m.get('metrics/mAP50-95(B)'))})
        rows.append({'benchmark':'HeinSight4 visual','metric':'mAP50','value':m.get('map50',m.get('metrics/mAP50(B)'))})
    cv=summary['stages'].get('train_test_crystalcv_temporal_v2',{}).get('result') or {}
    for h,block in cv.get('results',{}).items():
        for model,m in block.get('models',{}).items():
            rows.append({'benchmark':f'CrystalCV {h}','metric':f'{model} macro-exp NMAE','value':m.get('macro_experiment_nmae')})
            if model=='TCPTResidual':
                rows.append({'benchmark':f'CrystalCV {h}','metric':'TCPT skill vs LastValue','value':m.get('skill_vs_last_value')})
                rows.append({'benchmark':f'CrystalCV {h}','metric':'TCPT experiment win-rate vs LastValue','value':m.get('experiment_win_rate_vs_last_value')})
    ca=summary['stages'].get('train_test_context_rag_fusion_if_supported',{}).get('result') or {}
    for model,m in ca.get('results',{}).items():
        rows.append({'benchmark':'Context-aware 2026','metric':f'{model} test macro-F1','value':m.get('test',{}).get('macro_f1')})
    with (result_root/'PUBLIC_NIGHTLY_SUMMARY.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['benchmark','metric','value']);w.writeheader();w.writerows(rows)

    print('\n'+'#'*88);print('NIGHTLY FINISHED');print('#'*88)
    for r in rows:print(f"{r['benchmark']:35s} | {r['metric']:35s} | {r['value']}")
    print('\nResults:',result_root/'PUBLIC_NIGHTLY_SUMMARY.json')
    print('Compact :',result_root/'PUBLIC_NIGHTLY_SUMMARY.csv')

if __name__=='__main__':main()
