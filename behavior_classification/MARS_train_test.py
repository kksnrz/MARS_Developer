from __future__ import division
import os,sys,fnmatch
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import binarize
import dill
import json
import yaml
import time
from sklearn.ensemble import BaggingClassifier
from hmmlearn import hmm
from scipy import signal
import copy
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from behavior_classification import annotation_parsers as map
import gc
from behavior_classification import MARS_ts_util as mts
import joblib
from behavior_classification.behavior_helpers import *
from Util.seqIo import *
import scipy.io as sio
import progressbar
import random
import pdb

import orjson as oj
import pickle
import ijson

from scipy.ndimage import gaussian_filter1d

from .thesis import constrained_baum_welch as cbw
from .thesis import sampling_strategies as ss
from .thesis import analysis as ana
# warnings.filterwarnings("ignore")
# plt.ioff()

lcat = lambda L: [i for j in L for i in j]
flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))


def clf_suffix(clf_params):
    if clf_params['clf_type'].lower() == 'mlp':
        suff = '_layers' + '-'.join(clf_params['hidden_layer_sizes']) if 'hidden_layer_sizes' in clf_params.keys() else ''
        suff = suff + '/'

    elif clf_params['clf_type'].lower() == 'xgb':
        suff = '_es' + str(clf_params['early_stopping']) if 'early_stopping' in clf_params.keys() else ''
        suff = suff + '_depth' + str(clf_params['max_depth']) if 'max_depth' in clf_params.keys() else suff
        suff = suff + '_child' + str(
            clf_params['min_child_weight']) if 'min_child_weight' in clf_params.keys() else suff

    else:  # defaults to xgb
        suff = '_es' + str(clf_params['early_stopping']) if 'early_stopping' in clf_params.keys() else ''
        suff = suff + '_depth' + str(clf_params['max_depth']) if 'max_depth' in clf_params.keys() else suff
        suff = suff + '_child' + str(clf_params['min_child_weight']) if 'min_child_weight' in clf_params.keys() else suff

    suff = suff + '_wnd' if clf_params['do_wnd'] else suff
    suff = suff + '_cwt' if clf_params['do_cwt'] else suff
    suff = suff + '_' + str(clf_params['user_suff']) if 'user_suff' in clf_params.keys() else suff
    suff = suff + '/'
    return suff


def unpack_params(clf_params, clf_type):

    with open(os.path.join('behavior_classification','clf_defaults.yaml')) as f:
        defaults = yaml.load(f, Loader=yaml.FullLoader)
    params = {}
    for k in defaults[clf_type].keys():
        if k in clf_params.keys():
            params[k] = clf_params[k]
        else:
            params[k] = defaults[clf_type][k]
    return params


def choose_classifier(clf_params):
    if clf_params['clf_type'].lower() == 'mlp':
        params = unpack_params(clf_params, 'mlp_defaults')
        mlp = MLPClassifier(**params)
        clf = BaggingClassifier(mlp, max_samples=0.1, n_jobs=3, random_state=7, verbose=0)

    elif clf_params['clf_type'].lower() == 'xgb':
        params = unpack_params(clf_params, 'xgb_defaults')
        clf = XGBClassifier(**params)
        print(clf.get_xgb_params())
    else:
        print('Unrecognized classifier type %s, defaulting to XGBoost!' % clf_params['clf_type'])
        params = unpack_params(clf_params, 'xgb_defaults')
        clf = XGBClassifier(**params)
        print(clf.get_xgb_params())
    return clf


def load_data(project, dataset, train_behaviors,
              drop_behaviors=[], drop_empty_trials=False,
              drop_movies=[], do_quicksave=False):
    """
    Stream + preprocess behavior features with ijson, storing features in memmap
    so large datasets don't explode RAM.
    Returns:
        data_stack (np.memmap)  : shape (total_frames, max_feat_dim)
        annot_clean (dict)      : dict of {behavior: binary label list}
        vocabulary (dict)       : mapping of labels
    """
    # --- Load configs ---
    with open(os.path.join(project, 'project_config.yaml'), encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(project, 'behavior', 'config_classifiers.yaml'), encoding='utf-8') as f:
        clf_params = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(project, 'behavior', 'behavior_equivalences.yaml'), encoding='utf-8') as f:
        equivalences = yaml.load(f, Loader=yaml.FullLoader)
        if equivalences is None:
            equivalences = {}

    savestr = os.path.join(project, 'behavior', 'behavior_jsons', dataset + '_features')
    if clf_params['do_wnd']:
        savestr += '_wnd.pkl'
    elif clf_params['do_cwt']:
        savestr += '_cwt.pkl'
    else:
        savestr += '.pkl'

    json_path = os.path.join(project, 'behavior', 'behavior_jsons', dataset + '_features.json')

    if dataset not in ['train', 'test', 'val']:
        print(f"[ERROR] Invalid dataset: {dataset}")
        return [], [], []

    # --- Load vocabulary ---
    with open(json_path, "r", encoding='utf-8') as f:
        first_pass_data = json.load(f)
    vocabulary = first_pass_data['vocabulary']

    # --- Reload quicksave ---
    if do_quicksave and os.path.isfile(savestr):
        print(f"[INFO] Reloading from quicksave: {savestr}")
        with open(savestr, "rb") as f:
            metadata = pickle.load(f)

        annot_raw = metadata['annot_raw']
        vocabulary = metadata['vocabulary']
        shape = metadata['shape']
        memmap_path = metadata['memmap_path']
        data_stack = np.memmap(memmap_path, dtype=np.float32, mode='r', shape=shape)

        # Build clean annotations
        annot_clean = {}
        for label_name in train_behaviors:
            annot_clean[label_name] = []
            if label_name in equivalences:
                hit_list = [vocabulary[i] for i in equivalences[label_name] if i in vocabulary]
            else:
                hit_list = [vocabulary[label_name]]
            for a in annot_raw:
                a_clean = [1 if j in hit_list else 0 for j in a]
                if 1 in a_clean:
                    annot_clean[label_name] += a_clean
                else:
                    annot_clean[label_name] += [-1] * len(a_clean)

        print("done! (reloaded)\n")
        return data_stack, annot_clean, vocabulary

    # --- First pass: compute shapes ---
    print(f"[INFO] First pass: scanning {json_path} for shapes")
    total_frames, max_feat_dim = 0, 0
    with open(json_path, 'rb') as f:
        parser = ijson.kvitems(f, 'sequences.' + cfg['project_name'])
        for seq_key, seq_val in parser:
            if seq_key in drop_movies:
                continue

            feats = np.array(seq_val['features'], dtype=np.float32)
            if feats.ndim == 3:
                feats = np.swapaxes(feats, 0, 1)
                feats = mts.clean_data(feats)
                feats = np.concatenate((feats[:, 0, :], feats[:, 1, :]), axis=1)
            elif feats.ndim == 2:
                feats = mts.clean_data(feats)
            else:
                continue

            if clf_params['do_wnd']:
                windows = [int(np.ceil(w * cfg['framerate']) * 2 + 1) for w in clf_params['windows']]
                feats = mts.apply_windowing(feats, windows)
            elif clf_params['do_cwt']:
                scales = [int(np.ceil(w * cfg['framerate'])) for w in clf_params['wavelets']]
                feats = mts.apply_wavelet_transform(feats, scales)

            total_frames += feats.shape[0]
            max_feat_dim = max(max_feat_dim, feats.shape[1])

    print(f"[INFO] Total frames={total_frames}, max_feat_dim={max_feat_dim}")

    # --- Create memmap file ---
    cache_dir = os.path.join(project, "behavior", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    memmap_path = os.path.join(cache_dir, f"{dataset}_data_stack.dat")

    print(f"[INFO] Creating memmap at {memmap_path}")
    data_stack = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=(total_frames, max_feat_dim))

    # --- Second pass: fill memmap ---
    offset, annot_raw = 0, []
    with open(json_path, 'rb') as f:
        parser = ijson.kvitems(f, 'sequences.' + cfg['project_name'])
        for seq_key, seq_val in parser:
            if seq_key in drop_movies:
                continue

            feats = np.array(seq_val['features'], dtype=np.float32)
            annots = seq_val['annotations']

            if feats.ndim == 3:
                feats = np.swapaxes(feats, 0, 1)
                feats = mts.clean_data(feats)
                feats = np.concatenate((feats[:, 0, :], feats[:, 1, :]), axis=1)
            elif feats.ndim == 2:
                feats = mts.clean_data(feats)
            else:
                continue

            # --- Handle length mismatch ---
            if len(annots) != feats.shape[0]:
                print(f"[WARN] Length mismatch in sequence {seq_key}: "
                      f"annotations={len(annots)}, features={feats.shape[0]}. "
                      f"Trimming to min length.")
                min_len = min(len(annots), feats.shape[0])
                annots = annots[:min_len]
                feats = feats[:min_len, :]

            # --- Drop empty trials ---
            if drop_empty_trials:
                keep_seq = False
                for label_name in train_behaviors:
                    if label_name in equivalences:
                        hit_list = [vocabulary[i] for i in equivalences[label_name] if i in vocabulary]
                    else:
                        hit_list = [vocabulary[label_name]]
                    if any([a in hit_list for a in annots]):
                        keep_seq = True
                        break
                if not keep_seq:
                    # skip this sequence entirely
                    continue

            if clf_params['do_wnd']:
                feats = mts.apply_windowing(feats, windows)
            elif clf_params['do_cwt']:
                feats = mts.apply_wavelet_transform(feats, scales)

            # Drop behaviors
            if drop_behaviors:
                if not isinstance(drop_behaviors, list):
                    drop_behaviors = [drop_behaviors]
                drop_list = []
                for d in drop_behaviors:
                    if d in equivalences:
                        drop_list += [vocabulary[i] for i in equivalences[d]]
                    else:
                        drop_list.append(vocabulary[d])
                mask = ~np.isin(annots, drop_list)
                feats = feats[mask, :]
                annots = np.array(annots)[mask].tolist()

            # Write block into memmap
            n = feats.shape[0]
            padded = np.zeros((n, max_feat_dim), dtype=np.float32)
            padded[:, :feats.shape[1]] = feats
            data_stack[offset:offset+n, :] = padded
            offset += n

            annot_raw.append(annots)

    data_stack.flush()
    print(f"[INFO] Finished filling memmap ({offset} frames written)")

    # --- Save quicksave metadata if requested ---
    metadata = {
        'annot_raw': annot_raw,
        'vocabulary': vocabulary,
        'shape': (total_frames, max_feat_dim),
        'memmap_path': memmap_path,
    }
    if do_quicksave:
        with open(savestr, "wb") as f:
            pickle.dump(metadata, f)

    # --- Build clean annotations ---
    annot_clean = {}
    for label_name in train_behaviors:
        annot_clean[label_name] = []
        if label_name in equivalences:
            hit_list = [vocabulary[i] for i in equivalences[label_name] if i in vocabulary]
        else:
            hit_list = [vocabulary[label_name]]
        for a in annot_raw:
            a_clean = [1 if j in hit_list else 0 for j in a]
            if 1 in a_clean:
                annot_clean[label_name] += a_clean
            else:
                annot_clean[label_name] += [-1] * len(a_clean)

    print("done!\n")
    return data_stack, annot_clean, vocabulary


def assign_labels(all_predicted_probabilities, vocabulary):
    # Assigns labels based on the provided probabilities.
    labels_num = []
    num_frames = all_predicted_probabilities.shape[0]
    # Looping over frames, determine which annotation label to take.
    for i in range(num_frames):
        # Get the [Nx2] matrix of current prediction probabilities.
        current_prediction_probabilities = all_predicted_probabilities[i]
        # Get the positive/negative labels for each behavior, by taking the argmax along the pos/neg axis.
        onehot_class_predictions = np.argmax(current_prediction_probabilities, axis=1)
        # Get the actual probabilities of those predictions.
        predicted_class_probabilities = np.max(current_prediction_probabilities, axis=1)

        # If every behavioral predictor agrees that the current behavior is "other"
        if np.all(onehot_class_predictions == 0):
            # The index here is one past any positive behavior --this is how we code for "other".
            beh_frame = 0
            # How do we get the probability of it being "other?" Since everyone's predicting it, we just take the mean.
            proba_frame = np.mean(predicted_class_probabilities)
            labels_num.append(vocabulary['other'])
        else:
            # If we have positive predictions, we find the probabilities of the positive labels and take the argmax.
            pos = np.where(onehot_class_predictions)[0]
            max_prob = np.argmax(predicted_class_probabilities[pos])

            # This argmax is, by construction, the id for this behavior.
            beh_frame = pos[max_prob]
            proba_frame = predicted_class_probabilities[beh_frame]
            labels_num.append(beh_frame)

    return labels_num


def handle_missing_trials(X, y, drop_empty_trials=False):
    mask = []
    if type(y) is dict:  # if y is a dictionary, all behaviors must be annotated for in a trial
        for k in list(y.keys()):
            mask = min(mask, y[k]) if mask != [] else y[k]
    else:
        mask = y
    if drop_empty_trials:
        X = X[[i != -1 for i in mask]]
        if type(y) is dict:
            for k in list(y.keys()):
                y[k] = np.array(y[k])
                y[k] = np.array(y[k][[i != -1 for i in mask]])
        else:
            y = np.array(y)
            y = np.array(y[[i != -1 for i in mask]])
    else:
        if type(y) is dict:
            for k in list(y.keys()):
                y[k] = np.array(y[k])
                y[k] = np.array([i if i != -1 else 0 for i in y[k]])
        else:
            y = np.array([i if i != -1 else 0 for i in y])  # remove the -1's
    return X, y


def do_train(beh_classifier, X_tr, y_tr_beh, X_ev, y_ev_beh, savedir, verbose=0):
    t = time.time()
    beh_name = beh_classifier['beh_name']
    clf = beh_classifier['clf']
    clf_params = beh_classifier['params']
    print("XGB downsample rate: ", clf_params['downsample_rate'])
    # downsample the data & free up ram
    tmp = X_tr[::clf_params['downsample_rate'], :].copy()
    del X_tr
    X_tr = tmp
    tmp = y_tr_beh[::clf_params['downsample_rate']].copy()
    del y_tr_beh
    y_tr_beh = tmp
    if not X_ev == []:
        tmp = X_ev[::clf_params['downsample_rate'], :].copy()
        del X_ev
        X_ev = tmp
        tmp = y_ev_beh[::clf_params['downsample_rate']].copy()
        del y_ev_beh
        y_ev_beh = tmp

    # scale the data
    gc.collect()
    if(verbose):
        print('fitting preprocessing parameters...')
    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_tr = scaler.transform(X_tr)
    if not X_ev == []:
        X_ev = scaler.transform(X_ev)
    # shuffle data
    X_tr, idx_tr = shuffle_fwd(X_tr)
    y_tr_beh = y_tr_beh[idx_tr]

    # fit the classifier!
    gc.collect()
    if (verbose):
        print('fitting clf for %s' % beh_name)
        print('    training set: %d positive / %d total (%d %%)' % (
            sum(y_tr_beh), len(y_tr_beh), 100*sum(y_tr_beh)/len(y_tr_beh)))
    if not X_ev == []:
        eval_set = [(X_ev, y_ev_beh)]
        print('    eval set: %d positive / %d total (%d %%)' % (
            sum(y_ev_beh), len(y_ev_beh), 100 * sum(y_ev_beh) / len(y_ev_beh)))
        if clf_params['early_stopping']:
            if verbose:
                print('  + early stopping')
            clf.fit(X_tr, y_tr_beh, eval_set=eval_set,
                    early_stopping_rounds=clf_params['early_stopping'], verbose=True)
        else:
            clf.fit(X_tr, y_tr_beh, eval_set=eval_set, eval_metric='aucpr', verbose=True)
        results = clf.evals_result()
    else:
        if verbose:
            print('  no validation set included')
        clf.fit(X_tr, y_tr_beh, eval_metric='aucpr', verbose=True)
        results = []

    beh_classifier.update({'clf': clf,
                           'scaler': scaler})
    dill.dump(beh_classifier, open(os.path.join(savedir, 'classifier_' + beh_name), 'wb'))
    dt = (time.time() - t) / 60.
    print('Runtime of XGB training (do_train) was %.2f mins' % dt)
    return results, beh_classifier


def do_train_smooth(beh_classifier,
                    X_tr_beh, y_tr_beh_partial, keep_indices_tr,
                    X_ev_beh, y_ev_beh_partial, keep_indices_ev, savedir, verbose=False):
    # Optional DEBUGGING: load the trained classifier
    # clf_path = os.path.join(savedir, 'classifier_' + 'investigation')
    # beh_classifier_loaded = dill.load(open(clf_path, 'rb'))
    # clf = beh_classifier_loaded['clf']       # trained model
    # scaler = beh_classifier_loaded['scaler'] # fitted scaler

    beh_name = beh_classifier['beh_name']
    clf = beh_classifier['clf']
    scaler = beh_classifier['scaler']
    clf_params = beh_classifier['params']
    # set some parameters for post-classification smoothing:
    kn = clf_params['smk_kn']
    blur_steps = clf_params['blur'] ** 2
    shift = clf_params['shift']
    # get the labels for the current behavior
    t = time.time()

    # predict labels on training set
    if (verbose):
        print('XGB: predict labels on training set...')
    y_tr_pred_proba = np.zeros((len(y_tr_beh_partial), 2))
    gen = Batch(range(len(y_tr_beh_partial)), lambda x: x % 1e5 == 0, 1e5)
    for i in gen:
        inds = list(i)
        X_tr_s = scaler.transform(X_tr_beh[inds])
        pd_proba_tmp = (clf.predict_proba(X_tr_s))
        y_tr_pred_proba[inds] = pd_proba_tmp
    
    # predict labels on eval set
    if (verbose):
        print('XGB: predict labels on eval set...')
    y_ev_pred_proba = np.zeros((len(y_ev_beh_partial), 2))
    gen = Batch(range(len(y_ev_beh_partial)), lambda x: x % 1e5 == 0, 1e5)
    for i in gen:
        inds = list(i)
        X_ev_s = scaler.transform(X_ev_beh[inds])
        pd_proba_tmp = (clf.predict_proba(X_ev_s))
        y_ev_pred_proba[inds] = pd_proba_tmp

    print("XGB Diags pre-smoothing")
    y_pred = (y_tr_pred_proba[keep_indices_tr, 1] >= 0.5).astype(int)
    n_total = len(y_tr_beh_partial[keep_indices_tr])
    n_misclassified = np.sum(y_pred != y_tr_beh_partial[keep_indices_tr])
    frac_misclassified = n_misclassified / n_total
    print(f"Misclassified frames: {n_misclassified} / {n_total} ({frac_misclassified:.2%})")

    fp = np.sum((y_pred == 1) & (y_tr_beh_partial[keep_indices_tr] == 0))
    fn = np.sum((y_pred == 0) & (y_tr_beh_partial[keep_indices_tr] == 1))
    print(f"False Positives: {fp} ({fp/n_total:.2%})")
    print(f"False Negatives: {fn} ({fn/n_total:.2%})")
    
    # with gaussian smoothing 
    proba_smooth_tr = y_tr_pred_proba[:, 1]
    proba_smooth_ev = y_ev_pred_proba[:, 1]
    # with gaussian smoothing
    # proba_smooth_tr = gaussian_filter1d(y_tr_pred_proba[:, 1], sigma=1.5)
    # proba_smooth_ev = gaussian_filter1d(y_ev_pred_proba[:, 1], sigma=1.5)

    if clf_params['n_bins'] != 2:
        # Create binned observations for hmms
        bin_edges = np.linspace(0.0, 1.0, clf_params['n_bins'] + 1)
        # bin train set
        obs_bin_tr = np.digitize(proba_smooth_tr, bin_edges, right=False) - 1  # bins start from 0
        obs_bin_tr = np.clip(obs_bin_tr, 0, clf_params['n_bins'] - 1).astype(np.int64)
        obs_bin_ev = np.digitize(proba_smooth_ev, bin_edges, right=False) - 1  # bins start from 0
        obs_bin_ev = np.clip(obs_bin_ev, 0, clf_params['n_bins'] - 1).astype(np.int64)
    else:
        obs_bin_tr = np.argmax(y_tr_pred_proba, axis=1)
        obs_bin_ev = np.argmax(y_ev_pred_proba, axis=1)

    train_bin_counts = np.bincount(obs_bin_tr, minlength=clf_params['n_bins'])
    eval_bin_counts  = np.bincount(obs_bin_ev, minlength=clf_params['n_bins'])

    print("XGB Bin sample counts")
    for i in range(clf_params['n_bins']):
        print(f"Bin {i}: train={train_bin_counts[i]}, eval={eval_bin_counts[i]}")

    print("XGB Diags post-smoothing")
    y_pred = (proba_smooth_tr[keep_indices_tr] >= 0.5).astype(int)
    n_total = len(y_tr_beh_partial[keep_indices_tr])
    n_misclassified = np.sum(y_pred != y_tr_beh_partial[keep_indices_tr])
    frac_misclassified = n_misclassified / n_total
    print(f"Misclassified frames: {n_misclassified} / {n_total} ({frac_misclassified:.2%})")

    fp = np.sum((y_pred == 1) & (y_tr_beh_partial[keep_indices_tr] == 0))
    fn = np.sum((y_pred == 0) & (y_tr_beh_partial[keep_indices_tr] == 1))
    print(f"False Positives: {fp} ({fp/n_total:.2%})")
    print(f"False Negatives: {fn} ({fn/n_total:.2%})")




    ana.plot_xgb_proba_diags(proba=y_tr_pred_proba[keep_indices_tr, 1],
                             labels=y_tr_beh_partial[keep_indices_tr],
                             save_path=f"{savedir}xgb_proba_diagnostics_{beh_name}.png",
                             bin_edges=bin_edges,
                             threshold=0.5,
                             log_scale=True)

    ana.plot_calibration_curve(y_tr_pred_proba[keep_indices_tr, 1],
                               y_tr_beh_partial[keep_indices_tr],
                               'XGB',
                               f"{savedir}xgb_calibration_{beh_name}.png",
                               n_bins=clf_params['n_bins'])       

    # ----------------------------------------------------------------------------------------------
    # constrained Baum-Welch training + FBS smoothing (semi supervised branch)
    print(f"Labeled frames count (0, 1): {(y_tr_beh_partial != -1).sum()}, Unlabeled frames count (-1): {(y_tr_beh_partial == -1).sum()}")
    best_model, all_models = cbw.multi_restart_log_cbw([obs_bin_tr],
                                                       [y_tr_beh_partial],
                                                       [obs_bin_ev], 
                                                       [y_ev_beh_partial],
                                                       num_states=2,
                                                       num_symbols=clf_params['n_bins'],
                                                       early_stop=1e-6,
                                                       max_rounds=50,
                                                       n_restarts=10,
                                                       n_jobs=clf_params['nthread'],
                                                       model_selection=True,
                                                       decoder='post-viterbi',
                                                       verbose=True)
    print("CBW: Initial prob matrix:\n", best_model['best_params'][0])
    print("CBW: Transition matrix:\n", best_model['best_params'][1])
    print("CBW: Emission matrix:\n", best_model['best_params'][2])
    cbw_init_prob_mat, cbw_trans_mat, cbw_emission_mat = best_model['best_params']

    if (verbose):
        print('CBW: fitting HMM smoother...')
    hmm_bin_cbw = hmm.MultinomialHMM(n_components=2,
                                     algorithm="viterbi",
                                     random_state=42,
                                     params="",
                                     init_params="")
    
    hmm_bin_cbw.startprob_ = cbw_init_prob_mat
    hmm_bin_cbw.transmat_ = cbw_trans_mat
    hmm_bin_cbw.emissionprob_ = cbw_emission_mat

    y_proba_hmm_cbw = hmm_bin_cbw.predict_proba(obs_bin_tr.reshape((-1, 1)))

    y_pred_hmm_cbw = np.argmax(y_proba_hmm_cbw, axis=1)
    hmm_bin_cbw = _stabilize_hmm(hmm_bin_cbw)

    try:
        ana.plot_calibration_curve(
            y_proba_hmm_cbw[keep_indices_tr, 1],
            y_tr_beh_partial[keep_indices_tr],
            'HMM CBW',
            f"{savedir}hmm_cbw_calibration_{beh_name}.png",
            n_bins=clf_params['n_bins'],
            strategy='uniform'
        )
    except Exception as e:
        print(f"Error: Skipped calibration curve for {beh_name} due to error: {e}")

    print(f"CBW metrics:")
    precision_cbw, recall_cbw, f_measure_cbw = prf_metrics(y_tr_beh_partial[keep_indices_tr],
                                                           y_pred_hmm_cbw[keep_indices_tr],
                                                           beh_name)

    # ----------------------------------------------------------------------------------------------
    # original HMM MARS + FBS smoothing (fully supervised branch)
    # do hmm
    if (verbose):
        print('MARS: fitting HMM smoother...')
    hmm_bin = hmm.MultinomialHMM(n_components=2,
                                 algorithm="viterbi",
                                 random_state=42,
                                 params="",
                                 init_params="")
    hmm_bin.startprob_ = np.array([np.sum(y_tr_beh_partial[keep_indices_tr] == i) / float(len(y_tr_beh_partial[keep_indices_tr])) for i in range(2)])
    hmm_bin.transmat_ = mts.get_transmat(y_tr_beh_partial[keep_indices_tr], 2)

    # 10 bins case
    hmm_bin.emissionprob_ = mts.get_emissionmat(y_tr_beh_partial[keep_indices_tr], obs_bin_tr[keep_indices_tr], 2, clf_params['n_bins'])
    y_proba_hmm = hmm_bin.predict_proba(obs_bin_tr[keep_indices_tr].reshape((-1, 1)))
    
    y_pred_hmm = np.argmax(y_proba_hmm, axis=1)
    hmm_bin = _stabilize_hmm(hmm_bin)

    try:
        ana.plot_calibration_curve(
            y_proba_hmm[:, 1],
            y_tr_beh_partial[keep_indices_tr],
            'HMM MARS',
            f"{savedir}hmm_mars_calibration_{beh_name}.png",
            n_bins=clf_params['n_bins'],
            strategy='uniform'
        )
    except Exception as e:
        print(f"Error: Skipped calibration curve for {beh_name} due to error: {e}")

    # print the results of training
    dt = (time.time() - t) / 60.
    print('training took %.2f mins' % dt)
    print('performance on training set:')
    print("MARS metrics (unsmoothed):")
    precision, recall, f_measure = prf_metrics(y_tr_beh_partial[keep_indices_tr],
                                               y_pred_hmm, beh_name)
    
    beh_classifier.update({'clf': clf,
                           'scaler': scaler,
                           'precision': precision,
                           'recall': recall,
                           'f_measure': f_measure,
                           'hmm_bin': hmm_bin,
                           'precision_cbw': precision_cbw,
                           'recall_cbw': recall_cbw,
                           'f_measure_cbw': f_measure_cbw,
                           'hmm_bin_cbw': hmm_bin_cbw,
                           })
    dill.dump(beh_classifier, open(os.path.join(savedir, 'classifier_' + beh_name), 'wb'))


def _stabilize_hmm(hmm):
    eps = 1e-9
    # start
    sp = np.maximum(hmm.startprob_.astype(float), eps)
    hmm.startprob_ = sp / sp.sum()
    # transition
    A = np.maximum(hmm.transmat_.astype(float), eps)
    hmm.transmat_ = A / A.sum(axis=1, keepdims=True)
    # emission
    B = np.maximum(hmm.emissionprob_.astype(float), eps)
    hmm.emissionprob_ = B / B.sum(axis=1, keepdims=True)
    return hmm


def do_test(name_classifier, X_te_labeled, y_te_beh_labeled, verbose=0, doPRC=0):
    classifier = joblib.load(name_classifier)
    # unpack the classifier
    beh_name = classifier['beh_name']
    clf = classifier['bag_clf'] if 'bag_clf' in classifier.keys() else classifier['clf']

    hmm_bin = classifier['hmm_bin']
    hmm_bin_cbw = classifier['hmm_bin_cbw']

    # unpack the smoothing parameters
    if 'params' in classifier.keys():
        scaler = classifier['scaler']
        clf_params = classifier['params']
        kn = clf_params['smk_kn']
        blur_steps = clf_params['blur'] ** 2
        shift = clf_params['shift']
    else:
        scaler = joblib.load(os.path.join(os.path.dirname(name_classifier),'scaler'))
        kn = classifier['k']
        blur_steps = classifier['blur_steps']
        shift = classifier['shift']

    # scale the data
    X_te_labeled = scaler.transform(X_te_labeled)
    t = time.time()
    len_y = len(y_te_beh_labeled)
    gt = y_te_beh_labeled

    # predict probabilities:
    y_pred_proba = clf.predict_proba(X_te_labeled)
    proba_xgb = y_pred_proba

    y_pred_class = np.argmax(y_pred_proba, axis=1)
    preds_xgb = y_pred_class

    # HMM CBW configuration
    SMOOTH_GAUSSIAN = False

    if clf_params['n_bins'] != 2:
        # multi-bin hmm case
        bin_edges = np.linspace(0.0, 1.0, clf_params['n_bins'] + 1)
        if SMOOTH_GAUSSIAN:
            proba_smooth = gaussian_filter1d(y_pred_proba[:, 1], sigma=1.5) # do_fbs() is equivalent to 3-4 sigma
            obs_seq = np.digitize(proba_smooth, bin_edges, right=False) - 1
        else:
            obs_seq = np.digitize(y_pred_proba[:, 1], bin_edges, right=False) - 1
        obs_seq = np.clip(obs_seq, 0, clf_params['n_bins'] - 1).astype(np.int64)
    else:
        # 2x2 hmm case
        if SMOOTH_GAUSSIAN:
            proba_smooth = gaussian_filter1d(y_pred_proba[:, 1], sigma=1.5)
            obs_seq = (proba_smooth > 0.5).astype(np.int64)
        else:
            obs_seq = y_pred_class

    # HMM MARS
    # y_pred_fbs = mts.do_fbs(y_pred_class=y_pred_class, kn=kn, blur=4, blur_steps=blur_steps, shift=shift)
    # y_proba_fbs_hmm = hmm_bin.predict_proba(y_pred_fbs.reshape((-1, 1)))

    y_proba_fbs_hmm = hmm_bin.predict_proba(obs_seq.reshape((-1, 1)))
    y_pred_fbs_hmm = np.argmax(y_proba_fbs_hmm, axis=1)
    preds_fbs_hmm = y_pred_fbs_hmm
    proba_fbs_hmm = y_proba_fbs_hmm

    # HMM CBW
    y_proba_fbs_hmm_cbw = hmm_bin_cbw.predict_proba(obs_seq.reshape((-1, 1)))
    y_pred_fbs_hmm_cbw = np.argmax(y_proba_fbs_hmm_cbw, axis=1)
    preds_fbs_hmm_cbw = y_pred_fbs_hmm_cbw
    proba_fbs_hmm_cbw = y_proba_fbs_hmm_cbw

    if doPRC:
        # 2x2 hmm with fbs
        # y_pred_fbs = mts.do_fbs(y_pred_class=obs_seq, kn=kn, blur=4, blur_steps=blur_steps, shift=shift)
        # proba_sv = hmm_bin.predict_proba(y_pred_fbs.reshape(-1, 1))[:, 1]

        proba_sv = hmm_bin.predict_proba(obs_seq.reshape(-1, 1))[:, 1]
        proba_cbw = hmm_bin_cbw.predict_proba(obs_seq.reshape(-1, 1))[:, 1]

        sio.savemat(name_classifier + '_results.mat',
                    {'proba': proba_sv,
                     'gt': gt})
        sio.savemat(name_classifier + '_results_cbw.mat',
                    {'proba': proba_cbw,
                     'gt': gt})

    A_hmm_mars = hmm_bin.transmat_
    B_hmm_mars = hmm_bin.emissionprob_
    A_hmm_cbw = hmm_bin_cbw.transmat_
    B_hmm_cbw = hmm_bin_cbw.emissionprob_

    print("Transition Matrix (A) - HMM MARS:\n", A_hmm_mars)
    print("Transition Matrix (A) - HMM CBW:\n", A_hmm_cbw)
    print("Emission Matrix (B) - HMM MARS:\n", B_hmm_mars)
    print("Emission Matrix (B) - HMM CBW:\n", B_hmm_cbw)

    diff = np.sum(preds_fbs_hmm != preds_fbs_hmm_cbw)
    print("Number of differing frames:", diff)

    # mae: 0=identical proba output, 1=maximally different proba
    mae = np.mean(np.abs(proba_fbs_hmm[:,1] - proba_fbs_hmm_cbw[:,1]))
    # corr: 1=aligned proba output, 0=unaliged probas (unrelated patterns)
    corr = np.corrcoef(proba_fbs_hmm[:,1], proba_fbs_hmm_cbw[:,1])[0,1]
    print("Mean absolute error (MAE) probs:", mae, "Correlation:", corr)

    dt = time.time() - t
    print('inference took %.2f sec' % dt)

    return gt, proba_xgb, preds_xgb, preds_fbs_hmm, proba_fbs_hmm, preds_fbs_hmm_cbw, proba_fbs_hmm_cbw


def train_classifier(project, train_behaviors, drop_behaviors=[], drop_empty_trials=False,
                     max_positive=[0], drop_movies=[], do_quicksave=False):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # unpack user-provided classification parameters, and use default values for those not provided.
    config_fid = os.path.join(project, 'behavior', 'config_classifiers.yaml')
    with open(config_fid) as f:
        clf_params = yaml.load(f, Loader=yaml.FullLoader)
        if 'smk_kn' in clf_params.keys():
            clf_params['smk_kn'] = np.array(clf_params['smk_kn'])
    if not (clf_params['downsample_rate'] == int(clf_params['downsample_rate'])):
        print('Training set downsampling rate must be an integer; reverting to default value of 1.')
        clf_params['downsample_rate'] = 1

    print('loading training data...')
    X_tr, y_tr, vocab = load_data(project, 'train', train_behaviors,
                                  drop_behaviors=drop_behaviors,
                                  drop_empty_trials=drop_empty_trials,
                                  drop_movies=drop_movies,
                                  do_quicksave=do_quicksave)
    print('loading validation data...')
    X_ev, y_ev, _ = load_data(project, 'val', train_behaviors,
                              drop_behaviors=drop_behaviors,
                              drop_empty_trials=drop_empty_trials,
                              drop_movies=drop_movies,
                              do_quicksave=do_quicksave)
    print('loaded training data: %d X %d - %s ' % (X_tr.shape[0], X_tr.shape[1], list(y_tr.keys())))

    # now create the classifier and give it an informative name:
    classifier = choose_classifier(clf_params)
    classifier_name = cfg['project_name'] + '_' + clf_params['clf_type'] + clf_suffix(clf_params)
    savedir = os.path.join(project, 'behavior', 'trained_classifiers', classifier_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    print('Training classifier: ' + classifier_name.upper())
    # train each classifier in a loop:
    for beh_name in train_behaviors:
        print('######################### %s #########################' % beh_name)
        # drop trials missing annotations for our behavior of interest
        X_tr_beh, y_tr_beh = handle_missing_trials(X_tr, y_tr[beh_name], drop_empty_trials=drop_empty_trials)
        if X_ev != []:
            X_ev_beh, y_ev_beh = handle_missing_trials(X_ev, y_ev[beh_name], drop_empty_trials=drop_empty_trials)
        else:
            X_ev_beh = []
            y_ev_beh = []

        # # shuffle in blocks of 2000 frames:
        # blocksize = 2000
        # num_blocks_tr = int(np.ceil(len(y_tr_beh) / blocksize))
        # blockorder_tr = list(range(num_blocks_tr))
        # random.shuffle(blockorder_tr)
        # newinds_tr = list([j + blocksize * i for i in blockorder_tr for j in np.arange(blocksize)])
        # X_tr_beh = np.array([X_tr_beh[i, :] for i in newinds_tr if i < len(y_tr_beh)])
        # y_tr_beh = np.array([y_tr_beh[i] for i in newinds_tr if i < len(y_tr_beh)])
        # if X_ev != []:
        #     num_blocks_ev = int(np.ceil(len(y_ev_beh) / blocksize))
        #     blockorder_ev = list(range(num_blocks_ev))
        #     random.shuffle(blockorder_ev)
        #     newinds_ev = list([j + blocksize * i for i in blockorder_ev for j in np.arange(blocksize)])
        #     X_ev_beh = np.array([X_ev_beh[i, :] for i in newinds_ev if i < len(y_ev_beh)])
        #     y_ev_beh = np.array([y_ev_beh[i] for i in newinds_ev if i < len(y_ev_beh)])

        print(f"Keeping {clf_params['sampling_pct'] * 100}% of frames for training")
        X_tr_beh_labeled, \
            y_tr_beh_labeled, \
                y_tr_beh_partial, \
                    keep_indices_tr = ss.apply_sampling_strat(X_tr_beh,
                                                              y_tr_beh,
                                                              sampling_strategy=clf_params['sampling_strategy'],
                                                              sampling_pct=clf_params['sampling_pct'],
                                                              rng=42,
                                                              cluster_size_frames=clf_params['cluster_size_frames'])
        
        if X_ev != []:
            X_ev_beh_labeled, \
                y_ev_beh_labeled, \
                    y_ev_beh_partial, \
                        keep_indices_ev = ss.apply_sampling_strat(X_ev_beh,
                                                                  y_ev_beh,
                                                                  sampling_strategy=clf_params['sampling_strategy'],
                                                                  sampling_pct=clf_params['sampling_pct'],
                                                                  rng=42,
                                                                  cluster_size_frames=clf_params['cluster_size_frames'])
        else:
            X_ev_beh_labeled = []
            y_ev_beh_labeled = []
            y_ev_beh_partial = []
            keep_indices_ev = []

        bouts_tr = sum([(i != 0 and j == 0) for i, j in zip(y_tr_beh_partial[keep_indices_tr][:-1], y_tr_beh_partial[keep_indices_tr][1:])])
        print('training using %d positive (presence) frames (%s bouts)' % (sum(y_tr_beh_partial[keep_indices_tr]!=0), bouts_tr))

        beh_classifier = {'beh_name': beh_name,
                          'beh_id': vocab[beh_name],
                          'clf': classifier,
                          'params': clf_params}

        results = do_train(beh_classifier,
                           X_tr_beh_labeled, y_tr_beh_labeled,
                           X_ev_beh_labeled, y_ev_beh_labeled,
                           savedir, verbose=clf_params['verbose'])
        del X_ev_beh_labeled, y_ev_beh_labeled
        gc.collect()

        do_train_smooth(beh_classifier,
                        X_tr_beh,
                        y_tr_beh_partial,
                        keep_indices_tr,
                        X_ev_beh,
                        y_ev_beh_partial,
                        keep_indices_ev,
                        savedir,
                        verbose=clf_params['verbose'])
    
        print('done training!')
    return results


def test_classifier(project, test_behaviors, drop_behaviors=[], drop_empty_trials=False,
                    do_quicksave=False):
    config_fid = os.path.join(project, 'project_config.yaml')
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    # unpack user-provided classification parameters, and use default values for those not provided.
    config_fid = os.path.join(project, 'behavior', 'config_classifiers.yaml')
    with open(config_fid) as f:
        clf_params = yaml.load(f, Loader=yaml.FullLoader)
        if 'smk_kn' in clf_params.keys():
            clf_params['smk_kn'] = np.array(clf_params['smk_kn'])

    print('loading test data...')
    X_te, y_te, vocab = load_data(project, 'test', test_behaviors,
                                    drop_behaviors=drop_behaviors, do_quicksave=do_quicksave)
    print('loaded test data: %d X %d - %s ' % (X_te.shape[0], X_te.shape[1], list(set(y_te))))
    X_te, y_te = handle_missing_trials(X_te, y_te, drop_empty_trials=drop_empty_trials)


    classifier_name = cfg['project_name'] + '_' + clf_params['clf_type'] + clf_suffix(clf_params)
    savedir = os.path.join(project, 'behavior', 'trained_classifiers', classifier_name)

    # ref_beh = test_behaviors[0]   # reference behavior

    # _, _, _, keep_indices_te = ss.apply_sampling_strat(
    #     X_te,
    #     y_te[ref_beh],
    #     sampling_strategy=clf_params['sampling_strategy'],
    #     sampling_pct=clf_params['sampling_pct'],
    #     rng=42,
    #     cluster_size_frames=clf_params.get('cluster_size_frames', None)
    # )

    # keep_indices_te = np.sort(keep_indices_te)
    # T = len(keep_indices_te)

    T = len(list(y_te.values())[0])
    n_classes = max([vocab[b] for b in list(vocab.keys())])+1
    gt = np.zeros((T, n_classes)).astype(int)
    proba_xgb = np.zeros((T, n_classes, 2))
    preds_xgb = np.zeros((T, n_classes)).astype(int)

    preds_fbs_hmm = np.zeros((T, n_classes)).astype(int)
    proba_fbs_hmm = np.zeros((T, n_classes, 2))

    preds_fbs_hmm_cbw = np.zeros((T, n_classes)).astype(int)
    proba_fbs_hmm_cbw = np.zeros((T, n_classes, 2))

    print('loading classifiers from %s' % savedir)
    for b, beh_name in enumerate(test_behaviors):
        print('predicting %s...' % beh_name)
        # X_te_beh_labeled = X_te[keep_indices_te]
        # y_te_beh_labeled = y_te[beh_name][keep_indices_te]

        # print(f"applying sampling strategy: '{clf_params['sampling_strategy']}' sampling {clf_params['sampling_pct']*100}% of frames")
        # print(f"sampled test data size: {X_te_beh_labeled.shape[0]} X {X_te_beh_labeled.shape[1]}")
        name_classifier = os.path.join(savedir, 'classifier_' + beh_name)

        gt[:, vocab[beh_name]], \
        proba_xgb[:, vocab[beh_name], :], \
        preds_xgb[:, vocab[beh_name]], \
        preds_fbs_hmm[:, vocab[beh_name]], \
        proba_fbs_hmm[:, vocab[beh_name], :], \
        preds_fbs_hmm_cbw[:, vocab[beh_name]], \
        proba_fbs_hmm_cbw[:, vocab[beh_name], :], = do_test(name_classifier,
                                                            X_te, # X_te_beh_labeled,
                                                            y_te[beh_name], # y_te_beh_labeled,
                                                            verbose=clf_params['verbose'],
                                                            doPRC=True)
    all_pred = assign_labels(proba_xgb, vocab)
    
    all_pred_fbs_hmm = assign_labels(proba_fbs_hmm, vocab)
    
    all_pred_fbs_hmm_cbw = assign_labels(proba_fbs_hmm_cbw, vocab)
    gt = np.argmax(gt, axis=1)

    print('Classifier performance:')
    print("HMM MARS:")
    score_info(gt, all_pred_fbs_hmm, vocab)
    print("HMM CBW:")
    score_info(gt, all_pred_fbs_hmm_cbw, vocab)

    P = {'0_G': gt, # ground truth with values from vocab i.e. [0,1,2,3,...]
         '0_Gc': y_te, # ground truth categorical labels i.e. [0, 1]
         '1_pd': preds_xgb,
         '2_pd_fbs_hmm': preds_fbs_hmm,
         '3_proba_pd': proba_xgb,
         '4_proba_pd_hmm_fbs': proba_fbs_hmm,
         '5_pred_ass': all_pred, # xgb preds assigned labels (multiclass)
         '6_pred_fbs_hmm_ass': all_pred_fbs_hmm, # hmm mars preds assigned labels (multiclass)
         '7_pd_fbs_hmm_cbw': preds_fbs_hmm_cbw, 
         '8_proba_pd_hmm_fbs_cbw,': proba_fbs_hmm_cbw,
         '9_pred_fbs_hmm_ass_cbw': all_pred_fbs_hmm_cbw, # hmm cbw preds assigned labels (multiclass)
         }
    dill.dump(P, open(savedir + 'results.dill', 'wb'))
    sio.savemat(savedir + 'results.mat', P)


# def run_classifier(project, test_behaviors):
#     # this code actually saves *.annot files containing the raw predictions of the trained classifier,
#     # instead of just giving you the precision and recall. You can load these *.annot files in Bento
#     # along with the movies to inspect behavior labels by eye.
#     # Unlike test_classifier, this function runs classification on each video separately.
#     config_fid = os.path.join(project, 'project_config.yaml')
#     with open(config_fid) as f:
#         cfg = yaml.load(f, Loader=yaml.FullLoader)
#     # unpack user-provided classification parameters, and use default values for those not provided.
#     config_fid = os.path.join(project, 'behavior', 'config_classifiers.yaml')
#     with open(config_fid) as f:
#         clf_params = yaml.load(f, Loader=yaml.FullLoader)
#         if 'smk_kn' in clf_params.keys():
#             clf_params['smk_kn'] = np.array(clf_params['smk_kn'])
#
#     classifier_name = cfg['project_name'] + '_' + clf_params['clf_type'] + clf_suffix(clf_params)
#     savedir = os.path.join('trained_classifiers', 'mars_v1_8', classifier_name)
#     print('loading test data...')
#     X_te_0, y_te, names = load_data(project, 'test', test_behaviors)
#     print('loaded test data: %d X %d - %s ' % (X_te_0.shape[0], X_te_0.shape[1], list(set(y_te))))
#
#     for vid in test_videos:
#         print('processing %s...' % vid)
#         X_te_0, y_te, _ = load_data(video_path, [vid], test_behaviors)
#         if not y_te:
#             print('skipping this video...\n\n')
#             continue
#
#         T = len(list(y_te.values())[0])
#         n_classes = len(test_behaviors.keys())
#         gt = np.zeros((T, n_classes)).astype(int)
#         proba = np.zeros((T, n_classes, 2))
#         preds = np.zeros((T, n_classes)).astype(int)
#         preds_hmm = np.zeros((T, n_classes)).astype(int)
#         proba_hmm = np.zeros((T, n_classes, 2))
#         preds_fbs_hmm = np.zeros((T, n_classes)).astype(int)
#         proba_fbs_hmm = np.zeros((T, n_classes, 2))
#         beh_list = list()
#         for b, beh_name in enumerate(test_behaviors.keys()):
#             print('predicting behavior %s...' % beh_name)
#             beh_list.append(beh_name)
#             name_classifier = savedir + 'classifier_' + beh_name
#             gt[:, b], proba[:, b, :], preds[:, b], preds_hmm[:, b], proba_hmm[:, b, :], \
#                 preds_fbs_hmm[:, b], proba_fbs_hmm[:, b, :] = do_test(name_classifier, X_te_0, y_te, clf_params['verbose'])
#         all_pred = assign_labels(proba, beh_list)
#         all_pred_hmm = assign_labels(proba_hmm, beh_list)
#         all_pred_fbs_hmm = assign_labels(proba_fbs_hmm, beh_list)
#         all_gt = assign_labels(gt, beh_list) if b>1 else np.squeeze(gt)
#
#         vname,_ = os.path.splitext(os.path.basename(vid))
#         if not save_path:
#             save_path = video_path
#         map.dump_labels_bento(all_pred, os.path.join(save_path, 'predictions_'+vname+'.annot'),
#                               moviename=vid, framerate=30, beh_list=beh_list, gt=all_gt)
#         map.dump_labels_bento(all_pred_hmm, os.path.join(save_path, 'predictions_hmm_' + vname + '.annot'),
#                               moviename=vid, framerate=30, beh_list=beh_list, gt=all_gt)
#         map.dump_labels_bento(all_pred_fbs_hmm, os.path.join(save_path, 'predictions_fbs_hmm_' + vname + '.annot'),
#                               moviename=vid, framerate=30, beh_list=beh_list, gt=all_gt)
#         print('\n\n')
