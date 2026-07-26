#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path

REQ_ROLES={'train_support':4,'heldout':2,'confirmation':2}
REQ_STRATA={'person_present','no_person_hard_negative','small_person','occlusion','entry_exit'}

def split_tokens(s: str) -> set[str]:
    return {x.strip() for x in s.replace(';','|').split('|') if x.strip()}

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('manifest', type=Path)
    args=ap.parse_args()
    with args.manifest.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    scenes=defaultdict(list)
    for r in rows: scenes[r['scene_id']].append(r)
    roles=Counter(); cameras=set(); illum=set(); all_strata=set(); errors=[]
    for scene,rs in scenes.items():
        roles[rs[0]['split_role']]+=1
        cameras.add(rs[0].get('capture_point_id') or rs[0].get('camera_id'))
        illum.add(rs[0].get('illumination',''))
        for r in rs: all_strata |= split_tokens(r.get('strata',''))
        if not (100 <= len(rs) <= 300): errors.append(f'{scene}: frame count {len(rs)} outside 100..300')
        if not any(str(r.get('temporal_eval_eligible','')).lower() in {'1','true','yes'} for r in rs):
            errors.append(f'{scene}: no temporal_eval_eligible frames')
    if not (8 <= len(scenes) <= 12): errors.append(f'scenes={len(scenes)} outside 8..12')
    if len(cameras) < 3: errors.append(f'capture points={len(cameras)} < 3')
    if len({x for x in illum if x}) < 2: errors.append(f'illumination conditions={len({x for x in illum if x})} < 2')
    for role,n in REQ_ROLES.items():
        if roles[role] < n: errors.append(f'role {role}={roles[role]} < {n}')
    missing=REQ_STRATA-all_strata
    if missing: errors.append('missing strata: '+','.join(sorted(missing)))
    out={'ready_for_full_cpu_audit':not errors,'scene_count':len(scenes),'role_counts':dict(roles),
         'capture_points':len(cameras),'illumination_conditions':len({x for x in illum if x}),
         'missing_strata':sorted(missing),'errors':errors}
    print(json.dumps(out,ensure_ascii=False,indent=2))
    return 0 if not errors else 1
if __name__=='__main__': raise SystemExit(main())
