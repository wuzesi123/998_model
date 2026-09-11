from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable

import requests


def _md5(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _checksum_md5(entry: dict) -> str | None:
    c = str(entry.get('checksum') or '')
    if c.startswith('md5:'):
        return c.split(':', 1)[1].strip().lower()
    if len(c) == 32:
        return c.lower()
    return None


def zenodo_metadata(record_id: int | str, timeout: int = 90) -> dict:
    url = f'https://zenodo.org/api/records/{record_id}'
    r = requests.get(url, timeout=timeout, headers={'User-Agent': 'ChemProcessRAG-PublicBenchmark/1.0'})
    r.raise_for_status()
    return r.json()


def download_zenodo_record(
    record_id: int | str,
    out_dir: str | Path,
    include_substrings: Iterable[str] | None = None,
    exclude_substrings: Iterable[str] | None = None,
    extract: bool = True,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = zenodo_metadata(record_id)
    files = list(meta.get('files', []))
    inc = [s.lower() for s in (include_substrings or []) if s]
    exc = [s.lower() for s in (exclude_substrings or []) if s]

    def keep(f: dict) -> bool:
        name = str(f.get('key') or f.get('filename') or '').lower()
        if inc and not any(s in name for s in inc):
            return False
        if exc and any(s in name for s in exc):
            return False
        return True

    files = [f for f in files if keep(f)]
    if not files:
        avail = [str(f.get('key') or f.get('filename')) for f in meta.get('files', [])]
        raise RuntimeError(f'Zenodo {record_id}: no matching files. Available: {avail}')

    manifest = {
        'record_id': str(record_id),
        'title': meta.get('metadata', {}).get('title'),
        'files': [],
    }

    for f in files:
        name = str(f.get('key') or f.get('filename'))
        url = (f.get('links') or {}).get('self') or (f.get('links') or {}).get('download')
        if not url:
            raise RuntimeError(f'Zenodo file has no download URL: {name}')
        dst = out / name
        expected = _checksum_md5(f)
        size = int(f.get('size') or 0)
        valid = dst.exists() and (size <= 0 or dst.stat().st_size == size)
        if valid and expected:
            valid = _md5(dst) == expected
        if not valid:
            print(f'[download] {name} ({size/1e6:.1f} MB)')
            tmp = dst.with_suffix(dst.suffix + '.part')
            if tmp.exists():
                tmp.unlink()
            with requests.get(url, stream=True, timeout=(60, 300), headers={'User-Agent': 'ChemProcessRAG-PublicBenchmark/1.0'}) as r:
                r.raise_for_status()
                with tmp.open('wb') as w:
                    for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                        if chunk:
                            w.write(chunk)
            tmp.replace(dst)
            if expected and _md5(dst) != expected:
                raise RuntimeError(f'MD5 mismatch after download: {dst}')
        else:
            print(f'[download] reuse {dst}')

        extracted_to = None
        if extract:
            extracted_to = maybe_extract(dst, out / (dst.stem + '_extracted'))
        manifest['files'].append({
            'name': name,
            'path': str(dst),
            'size': dst.stat().st_size,
            'md5': expected,
            'extracted_to': str(extracted_to) if extracted_to else None,
        })

    (out / 'ZENODO_DOWNLOAD_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def maybe_extract(path: str | Path, out_dir: str | Path) -> Path | None:
    p = Path(path)
    out = Path(out_dir)
    suffix = p.name.lower()
    is_zip = suffix.endswith('.zip')
    is_tar = suffix.endswith('.tar') or suffix.endswith('.tar.gz') or suffix.endswith('.tgz') or suffix.endswith('.tar.bz2') or suffix.endswith('.tar.xz')
    if not (is_zip or is_tar):
        return None
    marker = out / '.complete'
    if marker.exists():
        return out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    print(f'[extract] {p.name} -> {out}')
    if is_zip:
        with zipfile.ZipFile(p) as z:
            z.extractall(out)
    else:
        with tarfile.open(p, 'r:*') as t:
            t.extractall(out)
    marker.write_text('ok\n', encoding='utf-8')
    return out


def recursively_extract_archives(root: str | Path, max_rounds: int = 3) -> None:
    root = Path(root)
    for _ in range(max_rounds):
        changed = False
        archives = [p for p in root.rglob('*') if p.is_file() and (
            p.name.lower().endswith('.zip') or p.name.lower().endswith('.tar.gz') or p.name.lower().endswith('.tgz')
        )]
        for p in archives:
            out = p.parent / (p.stem + '_extracted')
            if not (out / '.complete').exists():
                maybe_extract(p, out)
                changed = True
        if not changed:
            break
