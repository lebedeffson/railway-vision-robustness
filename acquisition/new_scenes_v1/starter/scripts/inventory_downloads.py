#!/usr/bin/env python3
"""Create a non-destructive inventory of downloaded archives/directories.

This script does not extract or parse restricted data. It records sizes and SHA-256
for files so the project can preserve provenance before conversion.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('--output', type=Path, default=Path('DOWNLOAD_INVENTORY.csv'))
    args = ap.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f'Not a directory: {root}')
    rows=[]
    for p in sorted(root.rglob('*')):
        if p.is_file() and not p.name.endswith('.sha256'):
            rows.append({
                'relative_path': str(p.relative_to(root)),
                'size_bytes': p.stat().st_size,
                'sha256': sha256(p),
            })
    with args.output.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=['relative_path','size_bytes','sha256'])
        w.writeheader(); w.writerows(rows)
    print(f'Wrote {len(rows)} rows to {args.output}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
