"""Pure-Python helpers for GGME-based D3PM candidate selection."""

from typing import Dict, Iterable, List, Tuple


DESCRIPTORS = (
    "degree_distribution",
    "clustering_coefficient",
    "normalised_laplacian_spectrum",
)

DIGRESS_TARGET_MMD = {
    "degree_distribution": 0.0125,
    "clustering_coefficient": 99.7936,
    "normalised_laplacian_spectrum": 1.8530,
}

SUPPORTED_SELECTION_METRICS = (
    "mean_rank",
    "digress_target_ratio",
)

PAPER_TARGET_MMD = {
    "degree_distribution": 0.0033908049082798097,
    "clustering_coefficient": 16.00800904243397,
    "normalised_laplacian_spectrum": 0.5686360149190364,
}


def structural_ratio_score(
    mmd: Dict[str, float],
    targets: Dict[str, float] = PAPER_TARGET_MMD,
    descriptors: Iterable[str] = DESCRIPTORS,
) -> Dict[str, object]:
    """Return Paper-normalized structural ratios and their mean score.

    Lower is better. Missing descriptors or zero targets make a run incomplete
    for structural ranking, so they raise ValueError instead of silently
    producing a misleading aggregate.
    """
    descriptor_names = list(descriptors)
    if not descriptor_names:
        raise ValueError("At least one descriptor is required")

    ratios = {}
    for descriptor in descriptor_names:
        if descriptor not in mmd or mmd[descriptor] is None:
            raise ValueError(f"Missing MMD for {descriptor}")
        if descriptor not in targets or targets[descriptor] == 0:
            raise ValueError(f"Missing non-zero structural target for {descriptor}")
        ratios[descriptor] = float(mmd[descriptor]) / float(targets[descriptor])

    return {
        "structural_ratios": ratios,
        "structural_ratio_score": sum(ratios.values()) / len(ratios),
    }


def mean_rank_scores(
    candidates: Iterable[Dict[str, object]],
    descriptors: Iterable[str] = DESCRIPTORS,
) -> List[Dict[str, object]]:
    """Add mean-rank GGME scores to candidate rows.

    Lower MMD is better. Ties receive the same rank. The returned list is sorted
    by mean rank and then by validation loss.
    """
    rows = [dict(candidate) for candidate in candidates]
    descriptor_names = list(descriptors)
    if not rows:
        raise ValueError("At least one candidate is required")
    if not descriptor_names:
        raise ValueError("At least one descriptor is required")

    ranks_by_candidate: Dict[str, List[int]] = {
        str(row["candidate_id"]): [] for row in rows
    }
    for descriptor in descriptor_names:
        ordered = sorted(
            rows,
            key=lambda row: (
                float(row["mmd"][descriptor]),
                # A generator may legitimately have no validation loss to report;
                # the default alone does not cover an explicit null, and this
                # only breaks ties, so treating it as 0.0 is safe.
                float(row.get("validation_loss") or 0.0),
            ),
        )
        previous_value = None
        previous_rank = None
        for index, row in enumerate(ordered, start=1):
            value = float(row["mmd"][descriptor])
            rank = previous_rank if value == previous_value else index
            ranks_by_candidate[str(row["candidate_id"])].append(rank)
            previous_value = value
            previous_rank = rank

    for row in rows:
        ranks = ranks_by_candidate[str(row["candidate_id"])]
        row["descriptor_ranks"] = {
            descriptor: rank for descriptor, rank in zip(descriptor_names, ranks)
        }
        row["ggme_score"] = sum(ranks) / len(ranks)

    return sorted(
        rows,
        key=lambda row: (
            float(row["ggme_score"]),
            float(row.get("validation_loss") or 0.0),
        ),
    )


def digress_target_ratio_scores(
    candidates: Iterable[Dict[str, object]],
    descriptors: Iterable[str] = DESCRIPTORS,
    targets: Dict[str, float] = DIGRESS_TARGET_MMD,
) -> List[Dict[str, object]]:
    """Add mean candidate/DiGress-target ratios to candidate rows."""
    rows = [dict(candidate) for candidate in candidates]
    descriptor_names = list(descriptors)
    if not rows:
        raise ValueError("At least one candidate is required")
    if not descriptor_names:
        raise ValueError("At least one descriptor is required")

    for descriptor in descriptor_names:
        if descriptor not in targets or targets[descriptor] == 0:
            raise ValueError(f"Missing non-zero DiGress target for {descriptor}")

    for row in rows:
        ratios = {
            descriptor: float(row["mmd"][descriptor]) / float(targets[descriptor])
            for descriptor in descriptor_names
        }
        row["descriptor_target_ratios"] = ratios
        row["ggme_score"] = sum(ratios.values()) / len(ratios)

    return sorted(
        rows,
        key=lambda row: (
            float(row["ggme_score"]),
            float(row.get("validation_loss") or 0.0),
        ),
    )


def select_best_candidate(
    candidates: Iterable[Dict[str, object]],
    descriptors: Iterable[str] = DESCRIPTORS,
    selection_metric: str = "mean_rank",
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    if selection_metric == "mean_rank":
        ranked = mean_rank_scores(candidates, descriptors)
    elif selection_metric == "digress_target_ratio":
        ranked = digress_target_ratio_scores(candidates, descriptors)
    else:
        raise ValueError(
            "Unsupported selection metric '{}'. Supported values: {}".format(
                selection_metric,
                ", ".join(SUPPORTED_SELECTION_METRICS),
            )
        )
    return ranked[0], ranked
