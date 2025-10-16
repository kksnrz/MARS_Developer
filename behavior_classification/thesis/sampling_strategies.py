from typing import Optional
import numpy as np


def _get_rng(rng: Optional[int] = None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    else:
        return np.random.default_rng(rng)


def _split_labeled(X, Y, keep_indices):
    """Return (X_labeled, Y_labeled, Y_partial) given kept indices."""
    Y_partial = np.full_like(Y, -1)
    Y_partial[keep_indices] = Y[keep_indices]
    X_labeled = X[keep_indices]
    Y_labeled = Y[keep_indices]
    return X_labeled, Y_labeled, Y_partial


def simple_random_sampling(X, Y, sampling_pct, rng=None):
    """Randomly sample a fraction of frames to keep labeled."""
    rng = _get_rng(rng)
    num_labels = len(Y)
    num_keep = int(round(num_labels * sampling_pct))
    keep_indices = np.sort(rng.choice(num_labels, size=num_keep, replace=False))
    return _split_labeled(X, Y, keep_indices) + (keep_indices,)


def stratified_sampling(X, Y, sampling_pct, rng=None):
    """Sample sampling_pct of frames within each class."""
    rng = _get_rng(rng)
    keep_indices = []
    unique, counts = np.unique(Y, return_counts=True)
    for unique_val, count in zip(unique, counts):
        if unique_val == -1:
            continue
        num_keep = int(round(count * sampling_pct))
        indices = np.where(Y == unique_val)[0]
        if num_keep > 0:
            chosen_indices = rng.choice(indices, size=min(num_keep, len(indices)), replace=False)
            keep_indices.append(chosen_indices)
    keep_indices = np.concatenate(keep_indices) if keep_indices else np.array([], dtype=int)
    keep_indices = np.sort(keep_indices)
    return _split_labeled(X, Y, keep_indices) + (keep_indices,)


def systematic_sampling(X, Y, sampling_pct, rng=None):
    """Sample every Kth frame with wrap-around, starting from a random offset."""
    rng = _get_rng(rng)
    num_labels = len(Y)
    num_keep = int(round(num_labels * sampling_pct))

    K = max(1, int(round(num_labels / max(1, num_keep))))  # prevents div 0
    start = int(rng.integers(0, K))

    steps = np.arange(num_labels, dtype=int)
    candidates = (start + steps * K) % num_labels

    # Preserve order of first occurrence and take first num_keep unique indices
    unique_vals, first_pos = np.unique(candidates, return_index=True)
    order = np.argsort(first_pos)
    keep_indices = unique_vals[order][:num_keep]
    keep_indices = np.sort(keep_indices)

    return _split_labeled(X, Y, keep_indices) + (keep_indices,)


def cluster_sampling(X, Y, sampling_pct, cluster_size_frames, rng=None):
    """Sample contiguous segments (clusters) of frames."""
    rng = _get_rng(rng)
    num_labels = len(Y)

    num_clusters = int(np.ceil(num_labels / cluster_size_frames))
    clusters = np.arange(num_clusters)
    rng.shuffle(clusters)

    num_keep = int(round(num_labels * sampling_pct))
    total = 0
    chosen_indices = []

    for cluster in clusters:
        start = cluster * cluster_size_frames
        end = min((cluster + 1) * cluster_size_frames, num_labels)
        indices = np.arange(start, end)
        chosen_indices.append(indices)
        total += len(indices)
        if total >= num_keep:
            break

    keep_indices = np.concatenate(chosen_indices)
    keep_indices = np.sort(keep_indices)
    return _split_labeled(X, Y, keep_indices) + (keep_indices,)


def purposive_sampling(X, Y, sampling_pct, rng=None, mode='rel_dist_centroid'):
    """Wrapper for purposive sampling strategies."""
    if mode == 'rel_dist_centroid':
        return _sample_mean_rel_dist_centroid(X, Y, sampling_pct, rng)
    else:
        raise ValueError(f"Unknown purposive sampling mode: {mode}")


def _sample_mean_rel_dist_centroid(X, Y, sampling_pct, rng=None):
    """
    Sample frames where the mean of the 21-frame window (mean(21))
        of rel_dist_centroid (X[:, 634]) is minimal.

    rel_dist_centroid is the distance between the centroids of the two mice in cm.

    Keeps lowest values until desired sampling percentage is reached.
    """
    rng = np.random.default_rng(rng)
    num_labels = len(Y)
    num_keep = int(round(num_labels * sampling_pct))

    rel_dist_mean21 = X[:, 634]

    order = np.argsort(rel_dist_mean21) # smallest = closer centroids
    keep_indices = np.sort(order[:num_keep]) # keep until pct

    return _split_labeled(X, Y, keep_indices) + (keep_indices,)


def apply_sampling_strat(X, Y, sampling_strategy='', sampling_pct=1, rng=42,
                         cluster_size_frames=None):
    """
    Wrapper function for applying various sampling strategies.
    
    Extra parameters:
        sampling_strategy: ['simple_random', 'stratified', 'systematic', 'cluster', 'purposive'].
        cluster_size_frames: optional required for cluster sampling.
    """
    strat = sampling_strategy.lower().strip()
    
    if strat == 'simple_random':
        X_labeled, Y_labeled, Y_partial, keep_indices = simple_random_sampling(X,
                                                                               Y,
                                                                               sampling_pct,
                                                                               rng=rng)
    elif strat == 'stratified':
        X_labeled, Y_labeled, Y_partial, keep_indices = stratified_sampling(X,
                                                                            Y,
                                                                            sampling_pct,
                                                                            rng=rng)
    elif strat == 'systematic':
        X_labeled, Y_labeled, Y_partial, keep_indices = systematic_sampling(X,
                                                                            Y,
                                                                            sampling_pct,
                                                                            rng=rng)
    elif strat == 'cluster':
        if cluster_size_frames is None:
            raise ValueError("Parameter 'cluster_size_frames' must be provided for cluster sampling.")
        X_labeled, Y_labeled, Y_partial, keep_indices = cluster_sampling(X,
                                                                         Y,
                                                                         sampling_pct,
                                                                         cluster_size_frames,
                                                                         rng=rng)
    elif strat == 'purposive':
        X_labeled, Y_labeled, Y_partial, keep_indices = purposive_sampling(X,
                                                                           Y,
                                                                           sampling_pct,
                                                                           rng=rng,
                                                                           mode='rel_dist_centroid')
    else:
        raise ValueError(
            f"Unknown sampling strategy '{sampling_strategy}'. "
            "Expected one of: ['simple_random', 'stratified', 'systematic', 'cluster', 'purposive'].")
    
    return X_labeled, Y_labeled, Y_partial, keep_indices


def main():
    num_frames = 10
    num_features = 635  # ensure index 634, for rel_dist_centroid sampling
    rng = np.random.default_rng(42)

    X = rng.random((num_frames, num_features))
    Y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    X[:, 634] = np.array([0.9, 0.2, 0.5, 0.8, 0.1, 0.3, 0.4, 0.05, 0.6, 0.15])

    # test purposive_sampling
    X_ps, Y_ps, Y_partial_ps, keep_indices = purposive_sampling(X, Y, sampling_pct=0.4, rng=rng)
    print("Purposive Sampling:")
    print("Kept indices:", keep_indices)
    print("Kept X[:, 634]:", X_ps[:, 634])
    print("Kept Y:", Y_ps)
    print("Y_partial:", Y_partial_ps)

    # test stratified_sampling
    _, Y_ss, Y_partial_ss, keep_indices = stratified_sampling(X, Y, sampling_pct=0.5, rng=rng)
    print("\nStratified Sampling:")
    print("Kept indices:", keep_indices)
    print("Kept Y:", Y_ss)
    print("Y_partial:", Y_partial_ss)


if __name__ == "__main__":
    main()
