from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from scripts.person_canonical_v5.common import PROJECT, assert_test_sealed


ROOT = PROJECT / "outputs/person_canonical_v5/instance_pasting"


def image_tag(path: str, title: str) -> str:
    if not path or path == "nan":
        return f"<div><b>{html.escape(title)}</b><br>not available</div>"
    uri = Path(path).resolve().as_uri()
    return (
        f"<div><b>{html.escape(title)}</b><br>"
        f"<img src=\"{html.escape(uri)}\" loading=\"lazy\"></div>"
    )


def render(fold: int, fraction: float, minimum_examples: int = 200) -> dict[str, object]:
    assert_test_sealed()
    suffix = f"fraction_{int(fraction * 100)}"
    root = ROOT / f"fold_{fold}/{suffix}"
    audit = pd.read_csv(root / "pasting_audit.csv")
    accepted = audit[audit["status"].eq("ACCEPTED")].copy()
    if len(accepted) < minimum_examples:
        raise RuntimeError(
            f"Only {len(accepted)} accepted examples; {minimum_examples} required"
        )
    if accepted["critical_flags"].fillna("").astype(str).str.len().gt(0).any():
        raise RuntimeError("Accepted paste contains a critical audit flag")
    sample = accepted.sort_values(
        ["grouped_scene_id", "source_frame", "tile_id", "instance_id"]
    ).head(minimum_examples)
    cards = []
    for row in sample.itertuples(index=False):
        details = {
            "scene": row.grouped_scene_id,
            "tile": row.tile_id,
            "instance_id": row.instance_id,
            "source_instance_scene": row.instance_source_scene_id,
            "range_source": row.range_source,
            "range_or_scale_group": row.source_range_or_scale_group,
            "estimated_source_range_m": row.estimated_source_range_m,
            "scale": row.scale,
            "surface_rule": row.surface_rule,
            "occlusion_rule": row.occlusion_rule,
            "new_box": [row.new_x1, row.new_y1, row.new_x2, row.new_y2],
        }
        cards.append(
            "<section><div class=\"images\">"
            + image_tag(str(row.original_tile), "original")
            + image_tag(str(row.source_instance), "instance")
            + image_tag(str(row.source_mask), "mask")
            + image_tag(str(row.final_tile), "pasted")
            + "</div><pre>"
            + html.escape(json.dumps(details, indent=2, ensure_ascii=False))
            + "</pre></section>"
        )
    payload = """<!doctype html>
<html><head><meta charset="utf-8"><title>Person V5 pasting audit</title>
<style>
body{font-family:sans-serif;margin:20px;background:#f5f5f5}
section{background:white;padding:12px;margin:0 0 16px;border-radius:8px}
.images{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
img{max-width:100%;max-height:280px;object-fit:contain;background:#222}
pre{white-space:pre-wrap}
</style></head><body>
<h1>Perspective-aware person pasting audit</h1>
<p>Development-only, train-scene sources. Accepted examples have no critical
flags. RGB-only inference is unchanged.</p>
""" + "\n".join(cards) + "\n</body></html>\n"
    destination = root / "INSTANCE_PASTING_AUDIT.html"
    destination.write_text(payload, encoding="utf-8")
    return {
        "status": "PASS",
        "fold": fold,
        "fraction": fraction,
        "examples": len(sample),
        "accepted_total": len(accepted),
        "path": str(destination.resolve()),
        "test_used": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--fraction", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.fold, args.fraction), indent=2))

