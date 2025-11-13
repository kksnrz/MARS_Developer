from __future__ import division
import os,sys
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import binarize
from collections import Counter
from sklearn.ensemble import BaggingClassifier
from hmmlearn import hmm
import scipy
from scipy import signal
import pywt
from scipy.signal import medfilt
from numba import njit

flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))

@njit(fastmath=True)
def clean_data(data):
    """
    Eliminate NaN and Inf values by replacing them
    with the last valid value along axis 0.
    If the first element is NaN/Inf, it is set to 0.
    """
    n0, n1, n2 = data.shape
    
    for i in range(n1):
        for j in range(n2):
            last_val = 0.0
            for k in range(n0):
                val = data[k, i, j]
                if np.isnan(val) or np.isinf(val):
                    data[k, i, j] = last_val
                else:
                    last_val = val
    return data


def apply_wavelet_transform(starter_features, scales=[1, 3, 5, 10, 30, 90, 270]):
    wave1 = pywt.ContinuousWavelet('gaus8')
    # wave2 = pywt.ContinuousWavelet('gaus7')
    dims = np.shape(starter_features)
    nfeat = dims[1]

    transformed_features = np.zeros((dims[0], nfeat + nfeat * len(scales)))#*2))
    transformed_features[:, range(0, nfeat)] = starter_features

    for f, feat in enumerate(starter_features.swapaxes(0, 1)):
        for w, wavelet in enumerate([wave1]): #, wave2]):
            for i, s in enumerate(scales):
                a, _ = pywt.cwt(medfilt(feat), s, wavelet)
                a[0][:s] = 0
                a[0][-s:] = 0
                inds = nfeat + f + nfeat*i + w*nfeat*len(scales)
                transformed_features[:, inds] = a

    return transformed_features


@njit(fastmath=True)
def create_toeplitz(col, row):
    """Manually create a Toeplitz matrix (Numba-compatible)."""
    n_rows = col.size
    n_cols = row.size
    mat = np.empty((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            idx = j - i
            if idx < 0:
                mat[i, j] = col[-idx]
            else:
                mat[i, j] = row[idx]
    return mat


@njit(fastmath=True)
def get_JAABA_feats(starter_feature, window_size=3):
    """
    Compute window features using a Toeplitz matrix, Numba-compatible.
    Returns same shape as original (frames x 4 functions).
    """
    number_of_frames = starter_feature.shape[0]
    radius = int(np.ceil((window_size - 1) / 2))
    
    # Prepare column and row for Toeplitz
    row_placeholder = np.zeros(window_size)
    col_placeholder = np.zeros(number_of_frames)
    
    # Fill row
    for i in range(radius):
        row_placeholder[i] = starter_feature[radius - i]
    for i in range(radius, window_size):
        row_placeholder[i] = starter_feature[i - radius]
    
    # Fill column
    for i in range(number_of_frames - radius):
        col_placeholder[i] = starter_feature[i + radius]
    for i in range(number_of_frames - radius, number_of_frames):
        col_placeholder[i] = starter_feature[number_of_frames - (i - (number_of_frames - radius)) - 2]

    # Create Toeplitz matrix
    window_matrix = create_toeplitz(col_placeholder, row_placeholder)
    
    # Compute min, max, mean, std along axis 1
    window_feats = np.zeros((number_of_frames, 4))
    for i in range(number_of_frames):
        # Min
        window_feats[i, 0] = np.min(window_matrix[i, :])
        # Max
        window_feats[i, 1] = np.max(window_matrix[i, :])
        # Mean (special case for window_size <=3)
        if window_size <= 3:
            window_feats[i, 2] = starter_feature[i]
        else:
            window_feats[i, 2] = np.mean(window_matrix[i, :])
        # Std
        window_feats[i, 3] = np.std(window_matrix[i, :])
    
    return window_feats


@njit(fastmath=True)
def compute_win_feat(starter_feature, windows=(3, 11, 21)):
    """
    Compute window features for multiple window sizes.
    Returns same shape as original (frames x num_windows*4).
    """
    number_of_frames = starter_feature.shape[0]
    num_windows = len(windows)
    num_fxns = 4
    num_feats = num_windows * num_fxns
    features = np.zeros((number_of_frames, num_feats))
    
    for window_num in range(num_windows):
        w = windows[window_num]
        left = window_num * num_fxns
        right = left + num_fxns
        window_feats = get_JAABA_feats(starter_feature, w)
        for i in range(number_of_frames):
            for j in range(num_fxns):
                features[i, left + j] = window_feats[i, j]
    
    return features


@njit(fastmath=True)
def apply_windowing(starter_features, windows=(3, 11, 21)):
    """
    Apply windowing to each feature column in starter_features using compute_win_feat.
    Returns a pre-allocated array with the same logic but faster.
    """
    num_frames, total_feat_num = starter_features.shape
    num_windows = len(windows)
    num_fxns = 4
    out_num_cols = total_feat_num * num_windows * num_fxns
    
    # Pre-allocate output array
    window_features = np.zeros((num_frames, out_num_cols))
    
    for i in range(total_feat_num):
        feat_temp = compute_win_feat(starter_features[:, i], windows)
        # Compute column indices for placement
        start_col = i * num_windows * num_fxns
        end_col = start_col + num_windows * num_fxns
        window_features[:, start_col:end_col] = feat_temp
    
    return window_features


def normalize_pixel_data(data,view):
    if view == 'top':fd = [range(40, 49)]
    elif view == 'front': fd = [range(47, 67)]
    elif view == 'top_pcf': fd = [range(40, 57)]
    fd = list(flatten(fd))
    md = np.nanmedian(data[:, :, fd], 1, keepdims=True)
    data[:, :, fd] /= md
    return data


def remove_pixel_data(data, view):
    if view == 'top': fd = [range(40, 49)]
    elif view == 'front': fd = [range(47,67)]
    elif view == 'top_pcf': fd = [range(40,57)]
    fd = list(flatten(fd))
    if type(data) == np.ndarray:
        data = np.delete(data, fd, 1)
    else:
        data = [i for j, i in enumerate(data) if j not in fd]
    return data


def get_transmat(gt, n_states):
    # Count transitions between states
    cs = [Counter() for _ in range(n_states)]
    prev = gt[0]
    for row in gt[1:]:
        cs[prev][row] += 1
        prev = row

    # Convert to probabilities
    transitions = np.zeros((n_states, n_states))
    for x in range(n_states):
        for y in range(n_states):
            transitions[x, y] = float(cs[x][y]) / float(sum(cs[x].values()))
    return transitions


def get_emissionmat(gt, pred, n_states, n_bins):
    # The emissions are the translations from ground truth to predicted
    # Count emissions
    counts = [Counter() for _ in range(n_states)]
    for s, o in zip(gt, pred):
        counts[s][o] += 1

    emissions = np.zeros((n_states, n_bins))
    for s in range(n_states):
        total = float(sum(counts[s].values()))
        if total > 0:
            for o in range(n_bins):
                emissions[s, o] = counts[s][o] / total
        else:
            # fallback if no samples for that state
            emissions[s, :] = 1.0 / n_bins

    return emissions


def do_fbs(y_pred_class, kn, blur, blur_steps, shift):
    """Does forward-backward smoothing."""
    len_y = len(y_pred_class)

    # fbs with classes
    z = np.zeros((3, len_y))  # Make a matrix to hold the shifted predictions --one row for each shift.

    # Create mirrored start and end indices for extending the length of our prediction vector.
    mirrored_start = range(shift, -1, -1)  # Creates indices that go (shift, shift-1, ..., 0)
    mirrored_end = range(len_y - 1, len_y - 1 - shift, -1)  # Creates indices that go (-1, -2, ..., -shift)

    # Now we extend the predictions to have a mirrored portion on the front and back.
    extended_predictions = np.r_[ y_pred_class[mirrored_start], y_pred_class, y_pred_class[mirrored_end] ]

    # Do our blurring.
    for s in range(blur_steps):
        extended_predictions = signal.convolve(np.r_[extended_predictions[0], extended_predictions, extended_predictions[-1]],
                                               kn / kn.sum(),  # The kernel we are convolving.
                                               'valid')  # Only use valid conformations of the filter.
        # Note: this will leave us with 2 fewer items in our signal each iteration, so we append on both sides.

    z[0, :] = extended_predictions[2 * shift + 1:]
    z[1, :] = extended_predictions[:-2 * shift - 1]
    z[2, :] = extended_predictions[shift + 1:-shift]

    z_mean = np.mean(z, axis=0)  # Average the blurred and shifted signals together.
    y_pred_fbs = binarize(z_mean.reshape((-1, 1)), threshold=0.5).astype(int).reshape((1, -1))[0]  # Anything that has a signal strength over 0.5, is taken to be positive.
    return y_pred_fbs


def do_hmm(gt, pd):
    hmm_bin = hmm.MultinomialHMM(n_components=2, algorithm="viterbi", random_state=42, params="", init_params="")
    hmm_bin.startprob_ = np.array([np.sum(gt == i) / float(len(gt)) for i in range(2)])
    hmm_bin.transmat_ = get_transmat(gt, 2)
    hmm_bin.emissionprob_ = get_emissionmat(gt, pd, 2)
    return hmm_bin
