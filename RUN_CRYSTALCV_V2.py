from __future__ import annotations

"""Run only the corrected CrystalCV V2 pipeline.

Use this after the previous overnight run if HeinSight is already complete:
    python RUN_CRYSTALCV_V2.py

It reuses existing downloads when present, additionally fetches the archived CrystalCV
software/repository (for author-provided Pre_Processed trajectories), prepares a clean
physical-experiment dataset, audits time units, and reruns the temporal benchmark.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from public_benchmark.downloads import download_zenodo_record, recursively_extract_archives
from public_benchmark.crystalcv_prepare import prepare_crystalcv
from public_benchmark.crystalcv_benchmark import run_crystalcv_benchmark


def _roots(manifest):
    out=[]
    for f in manifest.get('files',[]):
        q=f.get('extracted_to') or f.get('path')
        if q and Path(q).is_dir():
            recursively_extract_archives(q)
            out.append(Path(q))
    return out


def _load_or_download(record_id, folder, force=False):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    man=folder/'ZENODO_DOWNLOAD_MANIFEST.json'
    if man.exists() and not force:
        print('[resume] reuse',man)
        return json.loads(man.read_text(encoding='utf-8'))
    return download_zenodo_record(record_id,folder,extract=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--force-download',action='store_true')
    ap.add_argument('--epochs',type=int,default=30)
    ap.add_argument('--seed',type=int,default=7)
    ap.add_argument('--min-experiments',type=int,default=30)
    args=ap.parse_args()

    data=ROOT/'data'/'public'
    result=ROOT/'results'/'public_nightly'/'crystalcv'
    source=_load_or_download(19140131,data/'crystalcv',args.force_download)
    software=_load_or_download(20655010,data/'crystalcv_software',args.force_download)
    roots=_roots(source)+_roots(software)
    if not roots:
        raise RuntimeError('No extracted CrystalCV roots found')

    prepared=result/'prepared_v2'
    manifest=prepare_crystalcv(roots,prepared,min_rows=48,min_physical_experiments=args.min_experiments)
    print('\nPrepared physical experiments:',manifest['n_physical_experiments'])
    print('Time-unit audit:',manifest['time_unit_audit'])
    print('Materials:',manifest['materials'])

    report=run_crystalcv_benchmark(
        prepared,result/'benchmark_v2',epochs=args.epochs,batch=512,workers=4,
        seed=args.seed,d_model=192,layers=3,heads=6,
    )
    print('\n=== V2 SUMMARY ===')
    for h,b in report['results'].items():
        tc=b['models']['TCPTResidual']; lv=b['models']['LastValue']
        print(
            f"{h:22s} | LastValue={lv['macro_experiment_nmae']:.4f} "
            f"TCPT={tc['macro_experiment_nmae']:.4f} "
            f"skill={tc['skill_vs_last_value']:+.1%} "
            f"win-rate={tc['experiment_win_rate_vs_last_value']:.1%}"
        )
    print('\nResult:',result/'benchmark_v2'/'CRYSTALCV_TEMPORAL_RESULT_V2.json')


if __name__=='__main__':
    main()
