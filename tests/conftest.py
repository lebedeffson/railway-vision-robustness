from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

# These integration suites validate frozen outputs that are intentionally not
# distributed with a clean source checkout. They run on the experiment host and
# are skipped publicly when their exact inputs are absent. The canonical B7
# release is not listed here: its tests replay the tracked public ZIP instead.
PRIVATE_ARTIFACT_REQUIREMENTS = {
    "test_article_evidence_v1.py": "outputs/article_evidence_v1",
    "test_canonical_m4.py": "outputs/canonical_m4",
    "test_canonical_v2.py": "outputs/canonical_v2",
    "test_crop_verifier_v1.py": "outputs/crop_verifier_v1",
    "test_event_engineering_evidence_v1.py": "outputs/article_evidence_v1",
    "test_interim_audit.py": "outputs/final_practice/interim_audit",
    "test_new_scenes_v1.py": "outputs/new_scenes_v1",
    "test_operator_assistant_b6.py": "outputs/operator_assistant_b6",
    "test_operator_assistant_evidence_v2.py": "outputs/operator_assistant_evidence_v2",
    "test_operator_assistant_evidence_v3.py": "outputs/operator_assistant_evidence_v3",
    "test_person_canonical_v5.py": "outputs/person_canonical_v5",
    "test_person_v4_math.py": "outputs/training/yolo11m_baseline_stage2/weights/best.pt",
    "test_person_v4_proxy.py": "outputs/person_v4_proxy",
    "test_person_v4_proxy_runtime.py": "outputs/person_v4_proxy",
    "test_person_v5_execution.py": "outputs/person_canonical_v5",
    "test_person_v7_crop_verifier.py": "outputs/person_v7_crop_verifier",
    "test_person_v7_tracklet_verifier.py": "outputs/person_v7_tracklet_verifier",
    "test_person_v8_active_data.py": "outputs/person_v8",
    "test_project_closure_v1.py": "outputs/project_closure_v1",
    "test_rescue_v1.py": "outputs/rescue",
    "test_rescue_v2.py": "outputs/rescue_v2",
    "test_review_assistant_release.py": "outputs/review_assistant_release_v1",
    "test_revision_q1.py": "outputs/final_practice/revision_q1",
    "test_temporal_safety_v1.py": "outputs/temporal_safety_v1",
    "test_temporal_verifier_v1.py": "outputs/temporal_verifier_v1",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        required = PRIVATE_ARTIFACT_REQUIREMENTS.get(item.path.name)
        if required and not (ROOT / required).exists():
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "private frozen integration artifact is not distributed: "
                        f"{required}"
                    )
                )
            )
