from __future__ import annotations

import json

from person_v8b.analyze_loso import analyze
from person_v8b.extract_features import extract
from person_v8b.f0_audit import run_f0


def run() -> dict[str, object]:
    f0 = run_f0()
    if f0["status"] != "PASS":
        return {"status": "F0_FAIL", "f0": f0}
    f1 = extract()
    if f1["status"] != "PASS":
        return {"status": "F1_FAIL", "f0": f0, "f1": f1}
    f2 = analyze()
    return {"status": f2["status"], "f0": f0, "f1": f1, "f2": f2}


def main() -> int:
    result = run()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

