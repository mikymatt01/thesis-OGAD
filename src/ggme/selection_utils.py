"""Pure-Python helpers for GGME-based D3PM candidate selection."""

from typing import Dict, Iterable, List, Tuple


DESCRIPTORS = (
    "degree_distribution",
    "clustering_coefficient",
    "normalised_laplacian_spectrum",
)


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
                float(row.get("validation_loss", 0.0)),
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
            float(row.get("validation_loss", 0.0)),
        ),
    )


def select_best_candidate(
    candidates: Iterable[Dict[str, object]],
    descriptors: Iterable[str] = DESCRIPTORS,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    ranked = mean_rank_scores(candidates, descriptors)
    return ranked[0], ranked
