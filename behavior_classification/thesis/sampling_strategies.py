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


def simple_random_sampling(X, Y, sampling_pct, rng=None, target_pos_frac=0.2):
    """Randomly sample a fraction of frames to keep labeled."""
    rng = _get_rng(rng)
    num_labels = len(Y)
    num_keep = int(np.floor(num_labels * sampling_pct))

    pos_index = np.where(Y == 1)[0]
    neg_index = np.where(Y == 0)[0]

    num_pos_keep = int(np.floor(num_keep * target_pos_frac))
    num_neg_keep = num_keep - num_pos_keep

    pos_keep = rng.choice(pos_index,
                          size=min(num_pos_keep, len(pos_index)), replace=False) if len(pos_index) > 0 else np.array([], dtype=int)
    neg_keep = rng.choice(neg_index,
                          size=min(num_neg_keep, len(neg_index)), replace=False) if len(neg_index) > 0 else np.array([], dtype=int)

    keep_indices = np.sort(np.concatenate([pos_keep, neg_keep]))
    return (*_split_labeled(X, Y, keep_indices), keep_indices)


def stratified_sampling(X, Y, sampling_pct, rng=None):
    """Sample sampling_pct of frames within each class."""
    rng = _get_rng(rng)
    keep_indices = []
    unique, counts = np.unique(Y, return_counts=True)
    for unique_val, count in zip(unique, counts):
        if unique_val == -1:
            continue
        num_keep = int(np.floor(count * sampling_pct))
        indices = np.where(Y == unique_val)[0]
        if num_keep > 0:
            chosen_indices = rng.choice(indices, size=min(num_keep, len(indices)), replace=False)
            keep_indices.append(chosen_indices)
    keep_indices = np.concatenate(keep_indices) if keep_indices else np.array([], dtype=int)
    keep_indices = np.sort(keep_indices)
    return (*_split_labeled(X, Y, keep_indices), keep_indices)


def systematic_sampling(X, Y, sampling_pct, rng=None):
    """Sample every Kth frame with wrap-around, starting from a random offset."""
    rng = _get_rng(rng)
    num_labels = len(Y)
    num_keep = int(np.floor(num_labels * sampling_pct))
    K = max(1, int(np.floor(num_labels / max(1, num_keep))))  # prevents div 0
    start = int(rng.integers(0, K))

    steps = np.arange(num_labels, dtype=int)
    candidates = (start + steps * K) % num_labels

    # Preserve order of first occurrence and take first num_keep unique indices
    unique_vals, first_pos = np.unique(candidates, return_index=True)
    order = np.argsort(first_pos)
    keep_indices = unique_vals[order][:num_keep]
    keep_indices = np.sort(keep_indices)

    return (*_split_labeled(X, Y, keep_indices), keep_indices)


def cluster_sampling(X, Y, sampling_pct, cluster_size_frames, rng=None, target_pos_frac=0.2):
    """Sample contiguous segments (clusters) of frames to match target_pos_frac"""
    rng = _get_rng(rng)
    num_labels = len(Y)

    num_keep = int(np.floor(num_labels * sampling_pct))
    num_clusters = int(np.ceil(num_labels / cluster_size_frames))
    clusters_pos = []
    clusters_neg = []

    for cluster in range(num_clusters):
        start = cluster * cluster_size_frames
        end = min((cluster + 1) * cluster_size_frames, num_labels)
        y_seg = Y[start:end]
        pos_count = int(np.sum(y_seg == 1))
        total = end - start

        if total == 0:
            continue

        if pos_count > 0:
            clusters_pos.append((start, end, pos_count))
        else:
            clusters_neg.append((start, end, pos_count))

    # Randomize order within each group
    rng.shuffle(clusters_pos)
    rng.shuffle(clusters_neg)

    chosen = []
    total_pos = 0
    total_frames = 0

    i_pos = 0
    i_neg = 0

    while total_frames < num_keep and (i_pos < len(clusters_pos) or i_neg < len(clusters_neg)):

        # Decide from which pool to draw next
        if total_frames == 0:
            # Start with a positive cluster if possible
            if i_pos < len(clusters_pos):
                use_pos = True
            elif i_neg < len(clusters_neg):
                use_pos = False
            else:
                break
        else:
            curr_frac = total_pos / total_frames
            if curr_frac < target_pos_frac and i_pos < len(clusters_pos):
                # Need more positives -> pick from pos pool
                use_pos = True
            elif curr_frac >= target_pos_frac and i_neg < len(clusters_neg):
                # Too many or enough positives -> pick from neg pool
                use_pos = False
            else:
                # Fallback if preferred pool is exhausted
                if i_pos < len(clusters_pos):
                    use_pos = True
                elif i_neg < len(clusters_neg):
                    use_pos = False
                else:
                    break

        if use_pos:
            start, end, pos_count = clusters_pos[i_pos]
            i_pos += 1
        else:
            start, end, pos_count = clusters_neg[i_neg]
            i_neg += 1

        seg_len = end - start

        # Respect frame budget
        if total_frames + seg_len > num_keep:
            continue

        # Accept cluster
        chosen.extend(range(start, end))
        total_frames += seg_len
        total_pos += pos_count

    keep_indices = np.sort(np.array(chosen))
    return (*_split_labeled(X, Y, keep_indices), keep_indices)


def purposive_sampling(X, Y, sampling_pct, rng=None, mode='rel_dist_centroid', target_pos_frac=0.2, cluster_size_frames=300):
    """Wrapper for purposive sampling strategies."""
    if mode == 'rel_dist_centroid':
        return _sample_mean_rel_dist_centroid(X, Y, sampling_pct, rng, target_pos_frac=target_pos_frac, cluster_size_frames=cluster_size_frames)
    else:
        raise ValueError(f"Unknown purposive sampling mode: {mode}")


def _sample_mean_rel_dist_centroid(X, Y, sampling_pct, rng=None, target_pos_frac=0.2, cluster_size_frames=300):
    """
    Sample non-overlapping clusters of frames centered around frames 
        where the mean of the 21-frame window (mean(21)) 
        of rel_dist_centroid (X[:, 634]) is minimal.

    rel_dist_centroid is the distance between the centroids of the two mice in cm.

    Keeps lowest values until desired sampling percentage is reached.
    """
    def _batch_shuffle(arr, batch_size, rng):
        n = len(arr)
        if n == 0:
            return arr
        arr = arr.copy()
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            rng.shuffle(arr[start:end])
        return arr

    def get_window(anchor):
        start = max(0, anchor - half_window)
        end = min(num_labels, anchor + half_window)
        return start, end

    def window_stats(start, end):
        index = np.arange(start, end)
        return index, len(index), int(Y[index].sum())
    

    rng = _get_rng(rng)
    num_labels = len(Y)

    num_keep_target = int(np.floor(num_labels * sampling_pct))
    pos_target = int(num_keep_target * target_pos_frac)

    half_window = cluster_size_frames // 2
    covered = np.zeros(num_labels, dtype=bool)

    rel_dist = X[:, 634]

    pos_indices = np.where(Y == 1)[0]
    neg_indices = np.where(Y == 0)[0]

    pos_sorted = pos_indices[np.argsort(rel_dist[pos_indices])]
    neg_sorted = neg_indices[np.argsort(rel_dist[neg_indices])]

    pos_sorted = _batch_shuffle(pos_sorted, batch_size=30, rng=rng)
    neg_sorted = _batch_shuffle(neg_sorted, batch_size=30, rng=rng)

    min_len = min(len(pos_sorted), len(neg_sorted))
    interleave = np.empty((min_len * 2,), dtype=int)
    interleave[0::2] = pos_sorted[:min_len]
    interleave[1::2] = neg_sorted[:min_len]

    remaining = np.concatenate([pos_sorted[min_len:], neg_sorted[min_len:]])
    candidate_anchors = np.concatenate([interleave, remaining])

    total_frames = 0
    total_pos_frames = 0
    clusters = []

    for anchor in candidate_anchors:
        if total_frames >= num_keep_target:
            break
        start, end = get_window(anchor)

        if covered[start:end].any():
            continue

        win_index, win_total, win_pos = window_stats(start, end)
        new_total_pos = total_pos_frames + win_pos
        accept = new_total_pos <= pos_target

        if accept:
            clusters.append(win_index)
            covered[start:end] = True
            total_frames += win_total
            total_pos_frames = new_total_pos

    if clusters:
        keep_indices = np.unique(np.concatenate(clusters))
    else:
        keep_indices = np.array([], dtype=int)

    keep_indices = np.sort(keep_indices)

    if len(keep_indices) > num_keep_target:
        keep_indices = keep_indices[:num_keep_target]

    return (*_split_labeled(X, Y, keep_indices), keep_indices)


def natural_sampling(X, Y, sampling_pct, rng=None, target_pos_frac=0.20, mean_delay_frames=6, std_delay_frames=2):
    """
    Simulate human annotation behavior:
    Label onset/offset are shifted because humans react with ~250ms reaction delay.
    Assumed Human reaction delay ~200 ms * (30 frames / 1000 ms)  = 6 frames
    """
    rng = _get_rng(rng)
    Y = np.asarray(Y, dtype=int)
    N = len(Y)
    target_keep = int(np.floor(N * sampling_pct))

    # Detect behavior bouts
    diff = np.diff(Y)
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if Y[0] == 1:
        starts = np.r_[0, starts]
    if Y[-1] == 1:
        ends = np.r_[ends, N - 1]
    bouts = list(zip(starts, ends))

    # Apply delay, collect "human labeled positives"
    pos_indices = []
    for s, e in bouts:
        bout_len = e - s + 1
        delay = max(0, int(rng.normal(mean_delay_frames, std_delay_frames)))
        if bout_len <= delay:
            continue
        start_shift = s + delay
        end_shift = min(N - 1, e + delay)
        pos_indices.extend(range(start_shift, end_shift + 1))
    pos_indices = np.array(sorted(set(pos_indices)), dtype=int)

    # Add negatives until reaching desired positive fraction
    neg_candidates = np.where(Y == 0)[0]
    neg_candidates = np.setdiff1d(neg_candidates, pos_indices, assume_unique=True)

    num_pos = len(pos_indices)
    num_neg_target = int(num_pos * (1 - target_pos_frac) / target_pos_frac)

    if num_neg_target > 0 and len(neg_candidates) > 0:
        if num_neg_target > len(neg_candidates):
            num_neg_target = len(neg_candidates)
        neg_indices = rng.choice(neg_candidates, size=num_neg_target, replace=False)
    else:
        neg_indices = np.array([], dtype=int)

    keep_indices = np.sort(np.concatenate([pos_indices, neg_indices]))

    # Adjust to final sampling_pct
    if len(keep_indices) > target_keep:
        keep_indices = rng.choice(keep_indices, size=target_keep, replace=False)
        keep_indices = np.sort(keep_indices)
    elif len(keep_indices) < target_keep and len(neg_candidates) > 0:
        need = target_keep - len(keep_indices)
        extra_neg = np.setdiff1d(neg_candidates, neg_indices, assume_unique=True)
        if need > len(extra_neg):
            need = len(extra_neg)
        extra_samples = rng.choice(extra_neg, size=need, replace=False)
        keep_indices = np.sort(np.concatenate([keep_indices, extra_samples]))

    return (*_split_labeled(X, Y, keep_indices), keep_indices)


def apply_sampling_strat(X, Y, sampling_strategy='', sampling_pct=1, rng=42,
                         cluster_size_frames=None, target_pos_frac=0.2):
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
                                                                               rng=rng,
                                                                               target_pos_frac=target_pos_frac)
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
                                                                         rng=rng,
                                                                         target_pos_frac=target_pos_frac)
    elif strat == 'purposive':
        X_labeled, Y_labeled, Y_partial, keep_indices = purposive_sampling(X,
                                                                           Y,
                                                                           sampling_pct,
                                                                           rng=rng,
                                                                           mode='rel_dist_centroid',
                                                                           target_pos_frac=target_pos_frac,
                                                                           cluster_size_frames=cluster_size_frames)
    elif strat == 'natural':
        X_labeled, Y_labeled, Y_partial, keep_indices = natural_sampling(X,
                                                                         Y,
                                                                         sampling_pct,
                                                                         rng=rng,
                                                                         target_pos_frac=target_pos_frac)


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
