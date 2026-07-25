from __future__ import annotations

import json

from scripts.temporal_safety.common import OUTPUT


def main() -> None:
    gate_path = OUTPUT / "triage/TRIAGE_GATE.json"
    if not gate_path.is_file():
        raise RuntimeError("Triage result is missing")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    report = [
        "# Railway Person Temporal Safety v1",
        "",
        f"Status: `{gate['status']}`",
        "",
        "The frozen frame detector and two causal tracking candidates were compared",
        "on identical low-confidence predictions. Tracker parameters were selected",
        "only on fold-0 training scenes. Railway test data remained sealed.",
        "",
        "## Decision",
        "",
    ]
    if gate["status"] == "TRIAGE_PASS":
        report.append(
            "Two-fold triage passed. Full scene-disjoint development evaluation is required."
        )
    else:
        report.append(
            "No tracker passed every frozen triage condition. Full development, test, "
            "and a positive temporal-safety article remain blocked."
        )
    report.extend(["", "## Candidate results", ""])
    for decision in gate["decisions"]:
        delta = decision["deltas"]
        report.extend(
            [
                f"### {decision['tracker']}",
                "",
                f"- status: `{decision['status']}`",
                f"- delta Recall: `{delta['recall']:.6f}`",
                f"- relative FN reduction: `{delta['relative_FN_reduction']:.2%}`",
                f"- relative false-alarm increase: `{delta['relative_false_alarm_increase']:.2%}`",
                f"- delta F1: `{delta['F1']:.6f}`",
                "",
            ]
        )
    report.extend(
        [
            "## Scope",
            "",
            "- V8b was not modified.",
            "- T-norm features are out of scope.",
            "- Test access count is zero.",
            "- The result is a development engineering result, not test evidence.",
        ]
    )
    destination = OUTPUT / "reports/TEMPORAL_SAFETY_DEVELOPMENT_REPORT.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

