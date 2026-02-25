import argparse
import math

import numpy as np
import pandas as pd


def find_elbow(data: np.ndarray) -> int:
    """
    Finds the elbow of a curve using the point with maximum distance
    from the line connecting the start and end points.
    Expects data to be a sorted descending numpy array (like radius_before).
    """
    n = len(data)
    if n < 2:
        return 0

    # Coordinates of all points
    coords = np.vstack((np.arange(n), data)).T

    # Vector from start to end
    p1 = coords[0]
    p2 = coords[-1]
    vec = p2 - p1

    # Vector from start to each point
    vec_p = coords - p1

    # Project vec_p onto the normal of the line segment (p1->p2)
    # The normal vector is (-dy, dx) relative to vec (dx, dy)
    normal = np.array([-vec[1], vec[0]])
    normal = normal / np.linalg.norm(normal)

    # Calculate perpendicular distances
    distances = np.abs(np.dot(vec_p, normal))

    # The elbow is the point with max distance from the line
    elbow_idx = np.argmax(distances)

    return elbow_idx


def analyze(csv_path: str, recommend: bool = False):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        # If we are in recommend mode, we should perhaps output something safe or re-raise
        if not recommend:
            print(f"Error reading CSV: {e}")
        return

    # Use radius_before as the metric for "novelty" of the sample
    radii = df["radius_before"].values
    ranks = df["rank"].values

    # 1. Kneedle / Elbow method
    elbow_idx = find_elbow(radii)
    elbow_rank = ranks[elbow_idx]

    # 2. Heuristic: Where does radius drop by X% of the total drop?
    max_r = radii.max()
    min_r = radii.min()
    total_drop = max_r - min_r

    def rank_at_drop(pct: float) -> int:
        target = max_r - (total_drop * pct)
        # find first index where radius < target
        # radii is likely sorted descending already if it comes from rank.py usually,
        # but let's assume it is consistent with find_elbow usage.
        # np.searchsorted expects sorted array. -radii is sorted ascending (since radii is desc).
        idx = np.searchsorted(-radii, -target)
        if idx < len(ranks):
            return ranks[idx]
        return ranks[-1]

    r50 = rank_at_drop(0.5)

    if recommend:
        # Take the max of elbow and 50% coverage, round up to nearest 25
        base_k = max(elbow_rank, r50)
        k = math.ceil(base_k / 25) * 25
        print(f"{k}")
        return

    r80 = rank_at_drop(0.8)
    r90 = rank_at_drop(0.9)

    print(f"Total samples: {len(df)}")
    print(f"Radius range: {max_r:.4f} -> {min_r:.4f}")
    print("-" * 30)
    print("Recommended Thresholds:")
    print(f"1. Elbow Point: {elbow_rank} (radius={radii[elbow_idx]:.4f})")
    print(
        "   - Represents the point of maximum curvature; signifies diminishing returns."
    )
    print(f"2. 50% Coverage: {r50} (midpoint of diversity range)")
    print(f"3. 80% Coverage: {r80}")
    print(f"4. 90% Coverage: {r90}")

    # Output top 5 for context
    print("\nTop 5 Most Diverse Samples:")
    print(df[["rank", "filename", "radius_before"]].head(5).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ranking CSV to recommend sampling thresholds."
    )
    parser.add_argument("csv_path", type=str, help="Path to ranking CSV file")
    parser.add_argument(
        "--recommend-k",
        action="store_true",
        help="Output only the recommended top-k (elbow rounded up to nearest 25)",
    )
    args = parser.parse_args()

    analyze(args.csv_path, recommend=args.recommend_k)


if __name__ == "__main__":
    main()
