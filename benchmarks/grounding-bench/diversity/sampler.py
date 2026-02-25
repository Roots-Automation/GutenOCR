import logging

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def k_center_greedy(
    embeddings: np.ndarray,
    k: int = None,
    existing_indices: list[int] = None,
) -> list[dict]:
    """
    Ranks items by visual diversity using the k-center greedy algorithm.

    Args:
        embeddings: (N, D) normalized feature matrix
        k: Number of items to select. Defaults to all remaining items.
        existing_indices: Indices of items already selected (seed set).

    Returns:
        List of dicts containing ranking stats for each selected item.
    """
    n_total = embeddings.shape[0]
    if k is None:
        k = n_total

    # Initialize distances
    # min_dists[i] = distance from point i to the nearest selected point
    min_dists = np.full(n_total, np.inf, dtype=np.float32)
    selected_mask = np.zeros(n_total, dtype=bool)

    # 1. Initialize from existing or centroid
    if existing_indices:
        # Mark existing as selected to exclude from ranking
        existing_arr = np.array(existing_indices, dtype=int)
        selected_mask[existing_arr] = True

        if len(existing_indices) > 0:
            logger.info("Initializing distances from existing indices...")
            existing_emb = embeddings[existing_arr]
            # Calculate min distance to any existing point
            dists = 1 - np.dot(embeddings, existing_emb.T)
            dists = np.maximum(dists, 0.0)
            min_dists = np.min(dists, axis=1)
    else:
        # Initialize by prioritizing items furthest from the dataset centroid
        logger.info(
            "No existing context. Initializing with furthest-from-centroid strategy."
        )
        centroid = np.mean(embeddings, axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-10

        # Initialize min_dists such that the first selection will be the centroid outlier
        dists_to_centroid = 1 - np.dot(embeddings, centroid)
        min_dists = np.maximum(dists_to_centroid, 0.0)

    results = []

    for step in tqdm(range(k), desc="Ranking"):
        if np.all(selected_mask):
            break

        # Candidate distances: exclude already selected items
        candidate_dists = min_dists.copy()
        candidate_dists[selected_mask] = -1.0

        new_idx = np.argmax(candidate_dists)
        radius_before = candidate_dists[new_idx]

        if radius_before == -1.0:
            break

        # Update Selection
        selected_mask[new_idx] = True

        # Update distances based on the newly selected point
        dists_new = 1 - np.dot(embeddings, embeddings[new_idx])
        dists_new = np.maximum(dists_new, 0.0)

        min_dists = np.minimum(min_dists, dists_new)

        # Calculate coverage radius (max distance to nearest selected)
        remaining_dists = min_dists.copy()
        remaining_dists[selected_mask] = -1.0
        if np.all(selected_mask):
            radius_after = 0.0
        else:
            radius_after = np.max(remaining_dists)

        results.append(
            {
                "rank": step + 1,
                "index": int(new_idx),
                "radius_before": float(radius_before),
                "radius_after": float(radius_after),
                "radius_reduction": float(radius_before - radius_after),
            }
        )

    return results
