from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.review_assistant.b6_association import TrackFragment


ZERO_PREDICTED_LINK_RULE = (
    "precision=1.0; recall=0.0 when true links exist; f1=0.0"
)


def cluster_membership(clusters: list[list[str]]) -> dict[str, int]:
    return {
        fragment_id: cluster_index
        for cluster_index, cluster in enumerate(clusters)
        for fragment_id in cluster
    }


def evaluate_scene(
    fragments: list[TrackFragment],
    clusters: list[list[str]],
    expected_episode_ids: set[str],
) -> dict[str, float | int | str]:
    valid = [fragment for fragment in fragments if fragment.gt_episode_id]
    membership = cluster_membership(clusters)
    tp = fp = fn = tn = 0
    for left, right in itertools.combinations(valid, 2):
        truth = left.gt_episode_id == right.gt_episode_id
        predicted = membership[left.fragment_id] == membership[right.fragment_id]
        if truth and predicted:
            tp += 1
        elif truth:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    predicted_positive = tp + fp
    true_positive = tp + fn
    if predicted_positive == 0:
        precision = 1.0
        recall = 0.0 if true_positive else 1.0
        f1 = 0.0 if true_positive else 1.0
    else:
        precision = tp / predicted_positive
        recall = tp / max(true_positive, 1)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    gt_values = sorted({fragment.gt_episode_id for fragment in valid})
    gt_index = {value: index for index, value in enumerate(gt_values)}
    predicted_labels = np.asarray(
        [membership[fragment.fragment_id] for fragment in valid], dtype=int
    )
    truth_labels = np.asarray(
        [gt_index[fragment.gt_episode_id] for fragment in valid], dtype=int
    )
    bcubed_precision_values = []
    bcubed_recall_values = []
    for index in range(len(valid)):
        predicted_members = predicted_labels == predicted_labels[index]
        truth_members = truth_labels == truth_labels[index]
        overlap = int(np.logical_and(predicted_members, truth_members).sum())
        bcubed_precision_values.append(
            overlap / max(int(predicted_members.sum()), 1)
        )
        bcubed_recall_values.append(overlap / max(int(truth_members.sum()), 1))
    bcubed_precision = (
        float(np.mean(bcubed_precision_values))
        if bcubed_precision_values
        else 0.0
    )
    bcubed_recall = (
        float(np.mean(bcubed_recall_values)) if bcubed_recall_values else 0.0
    )
    bcubed_f1 = (
        2.0
        * bcubed_precision
        * bcubed_recall
        / (bcubed_precision + bcubed_recall)
        if bcubed_precision + bcubed_recall
        else 0.0
    )
    represented = {fragment.gt_episode_id for fragment in valid}
    return {
        "tp_links": tp,
        "fp_links": fp,
        "fn_links": fn,
        "tn_links": tn,
        "true_positive_pairs": true_positive,
        "predicted_positive_pairs": predicted_positive,
        "association_precision": precision,
        "association_recall": recall,
        "association_f1": f1,
        "cross_person_merge_rate": fp / max(predicted_positive, 1),
        "same_person_split_recovery": recall,
        "bcubed_precision": bcubed_precision,
        "bcubed_recall": bcubed_recall,
        "bcubed_f1": bcubed_f1,
        "adjusted_rand_index": (
            float(adjusted_rand_score(truth_labels, predicted_labels))
            if len(valid)
            else 0.0
        ),
        "person_episode_coverage": len(represented & expected_episode_ids)
        / max(len(expected_episode_ids), 1),
        "clusters": len(clusters),
        "evaluated_fragments": len(valid),
        "zero_predicted_link_rule": ZERO_PREDICTED_LINK_RULE,
    }


def micro_aggregate(rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    tp = sum(int(row["tp_links"]) for row in rows)
    fp = sum(int(row["fp_links"]) for row in rows)
    fn = sum(int(row["fn_links"]) for row in rows)
    tn = sum(int(row["tn_links"]) for row in rows)
    predicted = tp + fp
    true_positive = tp + fn
    if predicted == 0:
        precision = 1.0
        recall = 0.0 if true_positive else 1.0
        f1 = 0.0 if true_positive else 1.0
    else:
        precision = tp / predicted
        recall = tp / max(true_positive, 1)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return {
        "tp_links": tp,
        "fp_links": fp,
        "fn_links": fn,
        "tn_links": tn,
        "true_positive_pairs": true_positive,
        "predicted_positive_pairs": predicted,
        "association_precision": precision,
        "association_recall": recall,
        "association_f1": f1,
        "cross_person_merge_rate": fp / max(predicted, 1),
        "same_person_split_recovery": recall,
        "zero_predicted_link_rule": ZERO_PREDICTED_LINK_RULE,
    }
