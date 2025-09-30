"""
Log-domain Constrained Baum-Welch (cBW) implementation from Li et al. 2021:
"A new algorithm to train hidden Markov models for biological sequences with partial labels"

Implements constrained:
    forward/backward, eqs. (1)-(4),
    gamma/xi eqs. (5)-(6),
    M-step eqs. (10)-(12)
    model selection (eq. (13))

Supports topology masks to preserve zeros in A/B.
"""
import time
from typing import List, Optional, Tuple, Dict

import numpy as np

from numba import njit
from joblib import Parallel, delayed
from hmmlearn import hmm
from sklearn.metrics import precision_score, recall_score, f1_score


def to_log(x: np.ndarray) -> np.ndarray:
    """Convert probability array to log, mapping zeros to -inf."""
    with np.errstate(divide='ignore'):
        logx = np.log(x)
    logx[~np.isfinite(logx)] = -np.inf
    return logx


def from_log(logx: np.ndarray) -> np.ndarray:
    """Safe exp of log-array (may underflow to 0 for very small values)."""
    with np.errstate(over='ignore', under='ignore'):
        x = np.exp(logx)
    x = np.maximum(x, 0.0)
    return x


@njit(fastmath=True)
def log_constrained_forward(obs_seq: np.ndarray,
                            log_init_prob: np.ndarray,
                            log_trans_mat: np.ndarray,
                            log_emission_mat: np.ndarray,
                            labels: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    Constrained forward in log-domain, implements eqs. (1), (2)

    Returns:
      logalpha: (num_states, obs_seq_len)
      loglik: log P(obs_seq | theta)
    """
    num_states = log_trans_mat.shape[0]
    obs_seq_len = len(obs_seq)
    logalpha = np.full((num_states, obs_seq_len), -np.inf)  # init with log(0) = -inf

    # normalize labels to -1 for unlabeled
    if labels is None:
        # no labels present -> -1 for every frame
        labels_normalized = np.full(obs_seq_len, -1, dtype=np.int64)
    else:
        labels_normalized = np.empty(obs_seq_len, dtype=np.int64)
        for t in range(obs_seq_len):
            label = labels[t]
            labels_normalized[t] = -1 if (label < 0) else int(label)

    # initialization t = 0, paper eq. (1)
    logalpha[:, 0] = log_init_prob + log_emission_mat[:, obs_seq[0]]

    if labels_normalized[0] != -1:
        # first state is labeled -> positions disallowed by label get log(0) = -inf
        for i in range(num_states):
            if i != labels_normalized[0]:
                logalpha[i, 0] = -np.inf

    # recursion, paper eq. (2)
    # for t from 0..obs_seq_len-2 compute logalpha[:, t+1]
    for t in range(obs_seq_len - 1):
        logalpha_next_j = np.full(num_states, -np.inf)  # temp vector for next-state probs

        # for each possible next-state j, compute logsum over all previous states i
        for j in range(num_states):
            logsum_prev = -np.inf
            for i in range(num_states):
                contrib_ij = logalpha[i, t] + log_trans_mat[i, j] # i -> j transition
                logsum_prev = np.logaddexp(logsum_prev, contrib_ij) # avoid underflow for small log-space numbers
            logalpha_next_j[j] = logsum_prev + log_emission_mat[j, obs_seq[t + 1]]

        logalpha[:, t + 1] = logalpha_next_j

        # enforce constraint at t+1 (partial label)
        if labels_normalized[t + 1] != -1:
            for j in range(num_states):
                if j != labels_normalized[t + 1]:
                    logalpha[j, t + 1] = -np.inf

    # log-likelihood from final column (P(O|theta))
    loglik = -np.inf
    for i in range(num_states):
        loglik = np.logaddexp(loglik, logalpha[i, obs_seq_len - 1])

    return logalpha, loglik


@njit(fastmath=True)
def log_constrained_backward(obs_seq: np.ndarray,
                             log_trans_mat: np.ndarray,
                             log_emission_mat: np.ndarray,
                             labels: Optional[np.ndarray]) -> np.ndarray:
    """
    Constrained backward in log-domain, implements eqs. (3), (4)

    Returns:
        logbeta (num_states, obs_seq_len).
    """
    num_states = log_trans_mat.shape[0]
    obs_seq_len = len(obs_seq)
    logbeta = np.full((num_states, obs_seq_len), -np.inf)

    # normalize labels to -1 for unlabeled
    if labels is None:
        labels_normalized = np.full(obs_seq_len, -1, dtype=np.int64)
    else:
        labels_normalized = np.empty(obs_seq_len, dtype=np.int64)
        for t in range(obs_seq_len):
            label = labels[t]
            labels_normalized[t] = -1 if (label < 0) else int(label)

    # initialization beta(:,obs_seq_len) = 1 -> logbeta = 0 (log(1) = 0) for allowed states eq. (3)
    if labels_normalized[obs_seq_len - 1] == -1:
        # final time step, case unlabeled -> all states allowed
        logbeta[:, obs_seq_len - 1] = 0.0
    else:
        # final time step, case labeled -> constrain to labeled state
        allowed_state = labels_normalized[obs_seq_len - 1]
        logbeta[allowed_state, obs_seq_len - 1] = 0.0

    # recursion (eq. 4)
    for t in range(obs_seq_len - 2, -1, -1):
        # For each state i at time t, compute:
        # logbeta[i,t] = log sum_j[ log_trans_mat[i,j + log_emission_mat[j, obs_seq[t+1]] + logbeta[j, t+1] ]
        for i in range(num_states):
            logsum_prev = -np.inf
            for j in range(num_states):
                contrib_ij = (log_trans_mat[i, j] +
                              log_emission_mat[j, obs_seq[t + 1]] +
                              logbeta[j, t + 1])
                # accumulate contributions in log-domain
                logsum_prev = np.logaddexp(logsum_prev, contrib_ij)
            logbeta[i, t] = logsum_prev

        # enforce constraint at time t: if labeled, disallow all but that state
        if labels_normalized[t] != -1:
            for i in range(num_states):
                if i != labels_normalized[t]:
                    logbeta[i, t] = -np.inf

    return logbeta


@njit(fastmath=True)
def compute_log_gamma_xi(logalpha: np.ndarray,
                         logbeta: np.ndarray,
                         log_trans_mat: np.ndarray,
                         log_emission_mat: np.ndarray,
                         obs_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute loggamma and logxi in log-domain with per-time normalization, implements eqs. (5), (6)

    Returns:
      loggamma: (num_states, obs_seq_len)
      logxi: (num_states, num_states, obs_seq_len-1)
    """
    num_states, obs_seq_len = logalpha.shape
    loggamma = np.full((num_states, obs_seq_len), -np.inf)
    logxi = np.full((num_states, num_states, obs_seq_len - 1), -np.inf)

    # gamma per time step eq. (5)
    for t in range(obs_seq_len):
        numer = logalpha[:, t] + logbeta[:, t]

        denom = -np.inf
        for i in range(num_states):
            denom = np.logaddexp(denom, numer[i])
        for i in range(num_states):
            loggamma[i, t] = numer[i] - denom

    # xi per time step eq. (6) (transitions between time steps)
    for t in range(obs_seq_len - 1):
        # compute numerator log matrix for xi at time t:
        # numer[i,j] = logalpha[i,t] + log_trans_mat[i,j] + log_emission_mat[j,o_{t+1}] + logbeta[j,t+1]
        denom = -np.inf
        for i in range(num_states):
            for j in range(num_states):
                numer = (logalpha[i, t] + log_trans_mat[i, j] +
                       log_emission_mat[j, obs_seq[t + 1]] + logbeta[j, t + 1])
                logxi[i, j, t] = numer
                denom = np.logaddexp(denom, numer)

        # normalize logxi[:, :, t]
        for i in range(num_states):
            for j in range(num_states):
                logxi[i, j, t] -= denom

    return loggamma, logxi


@njit(fastmath=True)
def do_mstep_from_log(all_loggamma: List[np.ndarray],
                      all_logxi: List[np.ndarray],
                      sequences: List[np.ndarray],
                      num_states: int,
                      num_obs: int,
                      trans_mat_mask: Optional[np.ndarray] = None,
                      emission_mat_mask: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    M-step for HMM parameter updates (init_prob, trans_mat, emission_mat)
    using log-domain expectations (loggamma, logxi)
    returns updated model parameters in prob-space.

    Optional masks are used to preserve zeros to remain topology.
    """
    num_sequences = len(all_loggamma)

    if trans_mat_mask is None:
        trans_mat_mask = np.ones((num_states, num_states), dtype=np.uint8)
    if emission_mat_mask is None:
        emission_mat_mask = np.ones((num_states, num_obs), dtype=np.uint8)

    # ----------------------------------------
    # update init_prob
    pi_acc = np.zeros(num_states, dtype=np.float64)
    for seq_idx in range(num_sequences):
        loggamma = all_loggamma[seq_idx]
        loggamma_0 = loggamma[:, 0]

        logsum = -np.inf
        for i in range(num_states):
            logsum = np.logaddexp(logsum, loggamma_0[i])

        # convert log-space to prob-space and accumulate
        # as log(gamma_1)  + log(gamma_2) != log(gamma_1 + gamma_2)
        for i in range(num_states):
            pi_acc[i] += np.exp(loggamma_0[i] - logsum)

    # compute average over sequences
    init_prob_new = pi_acc / num_sequences
    init_prob_new /= np.sum(init_prob_new)

    # ----------------------------------------
    # update trans_mat
    numer_trans_mat = np.zeros((num_states, num_states), dtype=np.float64)
    denom_trans_mat = np.zeros(num_states, dtype=np.float64)

    for seq_idx in range(num_sequences):
        loggamma = all_loggamma[seq_idx]
        logxi = all_logxi[seq_idx]
        obs_seq_len = loggamma.shape[1]

        # compute numerator
        for t in range(obs_seq_len - 1):
            # normalize xi in log-space
            logsum = -np.inf
            for i in range(num_states):
                for j in range(num_states):
                    logsum = np.logaddexp(logsum, logxi[i, j, t])

            # convert to prob-space
            for i in range(num_states):
                for j in range(num_states):
                    numer_trans_mat[i,j] += np.exp(logxi[i, j, t] - logsum)

        # compute denominator: sum over gamma[:, t] for t=0..T-2
        for t in range(obs_seq_len - 1):
            logsum = -np.inf
            for i in range(num_states):
                logsum = np.logaddexp(logsum, loggamma[i, t])

            # convert to prob-space
            for i in range(num_states):
                denom_trans_mat[i] += np.exp(loggamma[i, t] - logsum)

    # consider optional mask
    trans_mat_new = np.zeros((num_states, num_states), dtype=np.float64)
    for i in range(num_states):
        if trans_mat_mask is None:
            # update trans_mat without mask
            for j in range(num_states):
                trans_mat_new[i,j] = numer_trans_mat[i,j] / denom_trans_mat[i]
        else:
            # update trans_mat with mask
            allowed_transitions_sum = 0.0
            for j in range(num_states):
                if trans_mat_mask[i,j]:
                    allowed_transitions_sum += numer_trans_mat[i,j]

            for j in range(num_states):
                if trans_mat_mask[i,j]:
                    if allowed_transitions_sum > 0.0:
                        trans_mat_new[i,j] = numer_trans_mat[i,j] / allowed_transitions_sum
                    else:
                        # zero-count row: temporarily set 0, will fix in normalization
                        trans_mat_new[i,j] = 0.0
                else:
                    trans_mat_new[i,j] = 0.0

    # normalize rows
    for i in range(num_states):
        row_sum = np.sum(trans_mat_new[i,:])
        if row_sum > 0.0:
            trans_mat_new[i,:] /= row_sum
        else:
            # fallback: uniform over allowed transitions
            num_allowed = 0
            for j in range(num_states):
                if trans_mat_mask[i,j]:
                    num_allowed += 1
            for j in range(num_states):
                if trans_mat_mask[i,j]:
                    trans_mat_new[i,j] = 1.0 / num_allowed

    # ----------------------------------------
    # update emission_mat
    numer_emission_mat = np.zeros((num_states, num_obs), dtype=np.float64)
    denom_emission_mat = np.zeros(num_states, dtype=np.float64)

    for seq_idx in range(num_sequences):
        seq = sequences[seq_idx]
        loggamma = all_loggamma[seq_idx]
        obs_seq_len = len(seq)


        for t in range(obs_seq_len):
            obs = seq[t]
            logsum = -np.inf
            for i in range(num_states):
                logsum = np.logaddexp(logsum, loggamma[i, t])

            for i in range(num_states):
                gamma_prob = np.exp(loggamma[i, t] - logsum)
                numer_emission_mat[i, obs] += gamma_prob # indicator func
                denom_emission_mat[i] += gamma_prob

    # consider optional mask
    emission_mat_new = np.zeros((num_states, num_obs), dtype=np.float64)
    for i in range(num_states):
        if emission_mat_mask is None:
            for k in range(num_obs):
                emission_mat_new[i,k] = numer_emission_mat[i,k] / denom_emission_mat[i]
        else:
            allowed_emission_total = 0.0
            for k in range(num_obs):
                if emission_mat_mask[i,k]:
                    allowed_emission_total += numer_emission_mat[i,k]

            for k in range(num_obs):
                if emission_mat_mask[i,k]:
                    if allowed_emission_total > 0.0:
                        emission_mat_new[i,k] = numer_emission_mat[i,k] / allowed_emission_total
                    else:
                        emission_mat_new[i,k] = 0.0
                else:
                    emission_mat_new[i,k] = 0.0

    # normalize rows
    for i in range(num_states):
        row_sum = np.sum(emission_mat_new[i,:])
        if row_sum > 0.0:
            emission_mat_new[i,:] /= row_sum
        else:
            # fallback: uniform over allowed emissions
            num_allowed = 0
            for j in range(num_states):
                if emission_mat_mask[i,j]:
                    num_allowed += 1
            for j in range(num_states):
                if emission_mat_mask[i,j]:
                    emission_mat_new[i,j] = 1.0 / num_allowed

    return init_prob_new, trans_mat_new, emission_mat_new


def log_cbw_train(sequences: List[np.ndarray],
                  labels_list: List[Optional[np.ndarray]],
                  num_states: int,
                  num_symbols: int,
                  init_init_prob_mat: Optional[np.ndarray] = None,
                  init_trans_mat: Optional[np.ndarray] = None,
                  init_emission_mat: Optional[np.ndarray] = None,
                  trans_mat_mask: Optional[np.ndarray] = None,
                  emission_mat_mask: Optional[np.ndarray] = None,
                  early_stop: float = 1e-6,
                  model_selection: bool = True,
                  decoder: str = 'post-viterbi',
                  rng_seed: int = 1234,
                  verbose: bool = False) -> Dict:
    """
    Train constrained Baum-Welch in log-domain with optional model selection
    based on partially labeled sequences.

    Returns:
        dict with final params and histories
    """
    obs_seq_len = len(sequences)
    assert len(labels_list) == obs_seq_len # prevent length mismatch

    rng = np.random.RandomState(rng_seed)

    # initialize params (prob space)
    if init_init_prob_mat is None:
        init_prob_mat = rng.rand(num_states)
        init_prob_mat = init_prob_mat / init_prob_mat.sum()
    else:
        init_prob_mat = np.asarray(init_init_prob_mat, dtype=float).copy()
        init_prob_mat = init_prob_mat / init_prob_mat.sum()

    if init_trans_mat is None:
        trans_mat = rng.rand(num_states, num_states)
        trans_mat = trans_mat / trans_mat.sum(axis=1, keepdims=True)
    else:
        trans_mat = np.asarray(init_trans_mat, dtype=float).copy()
        trans_mat = trans_mat / trans_mat.sum(axis=1, keepdims=True)

    if init_emission_mat is None:
        emission_mat = rng.rand(num_states, num_symbols)
        emission_mat = emission_mat / emission_mat.sum(axis=1, keepdims=True)
    else:
        emission_mat = np.asarray(init_emission_mat, dtype=float).copy()
        emission_mat = emission_mat / emission_mat.sum(axis=1, keepdims=True)

    # apply masks to initial mats
    if trans_mat_mask is not None:
        trans_mat_mask = np.asarray(trans_mat_mask, dtype=bool)
        trans_mat *= trans_mat_mask.astype(float) # apply masking
        trans_mat /= trans_mat.sum(axis=1, keepdims=True)

    if emission_mat_mask is not None:
        emission_mat_mask = np.asarray(emission_mat_mask, dtype=bool)
        emission_mat *= emission_mat_mask.astype(float) # apply masking
        emission_mat /= emission_mat.sum(axis=1, keepdims=True)

    # convert to log
    log_init_prob_mat = to_log(init_prob_mat)
    log_trans_mat = to_log(trans_mat)
    log_emission_mat = to_log(emission_mat)

    params_history = [(init_prob_mat.copy(), trans_mat.copy(), emission_mat.copy())] if model_selection else None
    loglik_history = []
    accuracy_history = [] if model_selection else None
    if model_selection:
        _, _, accs  = do_model_selection([params_history[-1]],
                                         sequences,
                                         labels_list,
                                         decoder=decoder)
        accuracy_history.append(accs[0])

    prev_total_loglik = -np.inf
    overall_start = time.time()
    it = 0

    while True:
        iter_start = time.time()
        all_loggamma = []
        all_logxi = []
        total_loglik = 0.0

        # E-step
        for seq_idx, obs_seq in enumerate(sequences):
            labels = labels_list[seq_idx]
            logalpha, seq_loglik = log_constrained_forward(obs_seq,
                                                           log_init_prob_mat,
                                                           log_trans_mat,
                                                           log_emission_mat,
                                                           labels)
            logbeta = log_constrained_backward(obs_seq, log_trans_mat, log_emission_mat, labels)
            loggamma, logxi = compute_log_gamma_xi(logalpha, logbeta, log_trans_mat, log_emission_mat, obs_seq)

            all_loggamma.append(loggamma)
            all_logxi.append(logxi)
            total_loglik += seq_loglik

        loglik_history.append(total_loglik)

        # M-step with optional masks
        pi_new, trans_mat_new, emission_mat_new = do_mstep_from_log(all_loggamma,
                                                                    all_logxi,
                                                                    sequences,
                                                                    num_states,
                                                                    num_symbols,
                                                                    trans_mat_mask=trans_mat_mask,
                                                                    emission_mat_mask=emission_mat_mask)

        # convert to log for next iteration
        log_init_prob_mat = to_log(pi_new)
        log_trans_mat  = to_log(trans_mat_new)
        log_emission_mat  = to_log(emission_mat_new)

        # model selection by decoding accuracy
        if model_selection:
            params_history.append((pi_new.copy(), trans_mat_new.copy(), emission_mat_new.copy()))
            _, _, accs = do_model_selection(
                [(pi_new, trans_mat_new, emission_mat_new)],
                sequences,
                labels_list, decoder=decoder)

            accuracy_history.append(accs[0])

        # progress logging
        iter_time = time.time() - iter_start
        delta = total_loglik - prev_total_loglik
        if verbose:
            msg = f"Seed {rng_seed} | Iter {it+1:3d} | Log-lik = {total_loglik:.4f} | delta = {delta:.9f} | Time = {iter_time:.2f}s"
            if model_selection:
                msg += f" | Acc = {accuracy_history[-1]:.4f}"
            print(msg)

        # stopping criterion
        if it > 0 and abs(delta) < early_stop:
            break

        prev_total_loglik = total_loglik
        it += 1

    overall_time = time.time() - overall_start
    if verbose:
        print(f"Training finished in {overall_time:.2f} seconds over {it+1} iterations.")

    # final outputs (prob-space)
    final_init_prob_mat = from_log(log_init_prob_mat)
    final_init_prob_mat /= final_init_prob_mat.sum()

    final_trans_mat  = from_log(log_trans_mat)
    final_trans_mat  /= final_trans_mat.sum(axis=1, keepdims=True)

    final_emission_mat  = from_log(log_emission_mat)
    final_emission_mat  /= final_emission_mat.sum(axis=1, keepdims=True)

    # select best model by accuracy if requested
    if model_selection and len(accuracy_history) > 0:
        best_idx = int(np.nanargmax(accuracy_history))
        best_params = params_history[best_idx]
    else:
        best_idx = None
        best_params = (final_init_prob_mat, final_trans_mat, final_emission_mat)

    return {'best_params': best_params,        # selected by partial-label accuracy
            'best_idx': best_idx,              # iteration index of the best model
            'params_history': params_history,  # all models per iteration
            'loglik_history': loglik_history,
            'accuracy_history': accuracy_history}


def train_once(seed: int,
               sequences: List[np.ndarray],
               labels_list: List[Optional[np.ndarray]],
               num_states: int,
               num_symbols: int,
               early_stop: float = 1e-6,
               trans_mat_mask: Optional[np.ndarray] = None,
               emission_mat_mask: Optional[np.ndarray] = None,
               model_selection: bool = True,
               decoder: str = 'post-viterbi',
               verbose: bool = False)-> Tuple[int, Dict]:
    """
    Single training run with a given seed, supports model selection by labels.
    Returns: seed, result_dict
    """
    result = log_cbw_train(
        sequences=sequences,
        labels_list=labels_list,
        num_states=num_states,
        num_symbols=num_symbols,
        early_stop=early_stop,
        trans_mat_mask=trans_mat_mask,
        emission_mat_mask=emission_mat_mask,
        model_selection=model_selection,
        decoder=decoder,
        rng_seed=seed,
        verbose=verbose
    )
    return seed, result


def multi_restart_log_cbw(sequences: List[np.ndarray],
                          labels_list: List[Optional[np.ndarray]],
                          num_states: int,
                          num_symbols: int,
                          early_stop: float = 1e-6,
                          n_restarts: int = 5,
                          n_jobs: int = -1,
                          trans_mat_mask: Optional[np.ndarray] = None,
                          emission_mat_mask: Optional[np.ndarray] = None,
                          model_selection: bool = True,
                          decoder: str = 'post-viterbi',
                          verbose: bool = False) -> Tuple[Dict, List[Tuple[int, Dict]]]:
    """
    Run multiple random restarts of constrained Baum-Welch in parallel,
    and select the best model based on decoding accuracy on partial labels.

    Returns:
        best_result: dict of model parameters (best across all restarts)
        all_results: list of (seed, result) for inspection
    """
    start_time = time.time()
    seeds = list(range(n_restarts))

    results = Parallel(n_jobs=n_jobs)(delayed(log_cbw_train)(sequences=sequences,
                                                             labels_list=labels_list,
                                                             num_states=num_states,
                                                             num_symbols=num_symbols,
                                                             early_stop=early_stop,
                                                             trans_mat_mask=trans_mat_mask,
                                                             emission_mat_mask=emission_mat_mask,
                                                             model_selection=model_selection,
                                                             decoder=decoder,
                                                             rng_seed=seed,
                                                             verbose=verbose) for seed in seeds)

    # pick best restart based on accuracy
    best_seed_idx = np.argmax([
        res['accuracy_history'][res['best_idx']] if res['accuracy_history'] else -1.0
        for res in results])
    best_result = results[best_seed_idx]

    elapsed = time.time() - start_time
    print(f"Finished {n_restarts} restarts in {elapsed:.2f}s.")

    if model_selection:
        print(f"Best restart: seed={seeds[best_seed_idx]}, , final loglik={best_result['loglik_history'][-1]:.3f}, accuracy={best_result['accuracy_history'][best_result['best_idx']]:.4f}")
    else:
        print(f"Best restart: seed={seeds[best_seed_idx]}, final loglik={best_result['loglik_history'][-1]:.3f}")

    # pair seeds with results
    all_results = list(zip(seeds, results))
    return best_result, all_results


def viterbi_decode(init_prob_mat: np.ndarray,
                   trans_mat: np.ndarray,
                   emission_mat: np.ndarray,
                   obs_seq: np.ndarray) -> np.ndarray:
    """Viterbi decoding using hmmlearn"""
    hmm_bin = hmm.MultinomialHMM(n_components=2, algorithm="viterbi", random_state=42, params="", init_params="")
    hmm_bin.startprob_ = init_prob_mat
    hmm_bin.transmat_ = trans_mat
    hmm_bin.emissionprob_ = emission_mat
    proba = hmm_bin.predict_proba(obs_seq.reshape((-1, 1)))
    preds = np.argmax(proba, axis=1)
    return preds


@njit(fastmath=True)
def posterior_viterbi(log_init_prob_mat: np.ndarray,
                      log_trans_mat: np.ndarray,
                      log_emission_mat: np.ndarray,
                      obs_seq: np.ndarray) -> np.ndarray:
    """
    Posterior-Viterbi decoding following the algorithm of Fariselli et al. 2005:
    "A new decoding algorithm for hidden Markov models improves
    the prediction of the topology of all-beta membrane proteins."

    Returns:
        path: decoded state sequence
        posterior: posterior probabilities 
    """
    num_states = log_trans_mat.shape[0]
    obs_seq_len = len(obs_seq)

    # 1. Forward-backward to get posterior
    logalpha, loglik = log_constrained_forward(obs_seq, log_init_prob_mat, log_trans_mat, log_emission_mat, labels=None)
    logbeta = log_constrained_backward(obs_seq, log_trans_mat, log_emission_mat, labels=None)

    # Posterior probabilities in log-space
    log_post = logalpha + logbeta - loglik  # shape (num_states, obs_seq_len)
    posterior = np.exp(log_post)

    # Initialize Viterbi DP arrays (in log-space)
    v = np.full((num_states, obs_seq_len), -np.inf)
    ptr = np.zeros((num_states, obs_seq_len), dtype=np.int64)

    # Initialization
    for k in range(num_states):
        v[k, 0] = log_post[k, 0]  # log(posterior at t=0)

    # Recursion
    for t in range(1, obs_seq_len):
        for k in range(num_states):
            max_val = -np.inf
            arg_max = 0
            for s in range(num_states):
                if log_trans_mat[s, k] > -np.inf:  # allowed transition
                    score = v[s, t-1] + log_trans_mat[s, k]
                    if score > max_val:
                        max_val = score
                        arg_max = s
            v[k, t] = max_val + log_post[k, t]  # multiply posterior in log-space → add
            ptr[k, t] = arg_max

    # Termination
    last_state = 0
    max_val = -np.inf
    for k in range(num_states):
        if v[k, obs_seq_len-1] > max_val:
            max_val = v[k, obs_seq_len-1]
            last_state = k

    # Traceback
    path = np.zeros(obs_seq_len, dtype=np.int64)
    path[obs_seq_len-1] = last_state
    for t in range(obs_seq_len-1, 0, -1):
        path[t-1] = ptr[path[t], t]

    return path, posterior


def do_model_selection(params_history: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                      sequences: List[np.ndarray],
                                      labels_list: List[Optional[np.ndarray]],
                                      decoder: str = 'post-viterbi'
                                      ) -> Tuple[int, Tuple[np.ndarray, np.ndarray, np.ndarray], List[float]]:
    """
    Evaluate each saved model using decoding accuracy at known labeled positions.
    """
    accuracies = []

    for init_prob_mat, trans_mat, emission_mat in params_history:
        total_correct = 0
        total_labeled = 0

        for seq, labels in zip(sequences, labels_list):
            if labels is None:
                continue
            if decoder == 'viterbi':
                preds = viterbi_decode(init_prob_mat, trans_mat, emission_mat, seq)
            elif decoder == 'post-viterbi':
                preds, probs = posterior_viterbi(to_log(init_prob_mat), to_log(trans_mat), to_log(emission_mat), seq)
            else:
                raise ValueError("decoder must be 'viterbi' or 'post-viterbi'")

            # compute accuracy only at known labels
            for t in range(len(seq)):
                label = labels[t]
                if label is None or label < 0:
                    continue
                total_labeled += 1
                if preds[t] == label:
                    total_correct += 1

        acc = total_correct / total_labeled if total_labeled > 0 else float('nan')
        accuracies.append(acc)

    best_idx = int(np.nanargmax(accuracies))
    best_params = params_history[best_idx]

    return best_idx, best_params, accuracies


def _simulate_hmm(init_prob_mat: np.ndarray,
                  trans_mat: np.ndarray,
                  emission_mat: np.ndarray,
                  obs_seq_len: int,
                  rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate one sequence of length T from given HMM
    Returns: states, observations
    """
    num_states = init_prob_mat.shape[0]
    states = np.zeros(obs_seq_len, dtype=int)
    obs = np.zeros(obs_seq_len, dtype=int)
    states[0] = rng.choice(num_states, p=init_prob_mat)
    obs[0] = rng.choice(emission_mat.shape[1], p=emission_mat[states[0]])
    for t in range(1, obs_seq_len):
        states[t] = rng.choice(num_states, p=trans_mat[states[t-1]])
        obs[t] = rng.choice(emission_mat.shape[1], p=emission_mat[states[t]])
    return states, obs


def main():
    """
    Synthetic sanity test:
      - Build a ground-truth HMM
      - Simulate sequences and partial labels
      - Run constrained Baum–Welch.
      - Print intermediate matrices (alpha, beta, gamma, xi) for one sequence
      - Print log-likelihood history and model selection results.
      """
    rng = np.random.RandomState(0)
    partial_label_pct = 0.05
    num_states = 2 
    num_obs_symbols = 2

    # 1. Ground truth parameters
    init_prob_mat_true = np.array([0.3, 0.7])

    trans_mat_true = np.array([[0.7, 0.3],
                               [0.4, 0.6]])

    emission_mat_true = np.array([[0.9, 0.1],
                                  [0.2, 0.8]])
    
    print("\nGround-truth pi:\n", init_prob_mat_true)
    print("Ground-truth A:\n", trans_mat_true)
    print("Ground-truth B:\n", emission_mat_true)

    # 2. Simulate sequences
    sequences, states_list, labels_list = [], [], []
    for _ in range(1):  # number of sequences
        states, obs = _simulate_hmm(init_prob_mat_true,
                                    trans_mat_true,
                                    emission_mat_true,
                                    obs_seq_len=1000000,
                                    rng=rng)
        sequences.append(obs)
        states_list.append(states)
        mask = rng.rand(len(obs)) < partial_label_pct
        labels = np.full(len(obs), -1, dtype=int)
        labels[mask] = states[mask]
        labels_list.append(labels)
        print("\nSimulated hidden states:", states)
        print("Observed symbols:", obs)
        print("Partial labels:", labels)

    # # 3. Debug one sequence
    # print("\nCalculation of: forward/backward/gamma/xi")
    # seq, labels = sequences[0], labels_list[0]

    # # Initialize parameters (simple uniform)
    # init_prob_mat_init = np.ones(num_states) / num_states
    # trans_mat_init  = np.ones((num_states,num_states)) / num_states
    # emission_mat_init  = np.ones((num_states,num_obs_symbols)) / num_obs_symbols

    # log_init_prob_mat = to_log(init_prob_mat_init)
    # log_trans_mat = to_log(trans_mat_init)
    # log_emission_mat = to_log(emission_mat_init)

    # logalpha, loglik = log_constrained_forward(seq, log_init_prob_mat, log_trans_mat, log_emission_mat, labels)
    # logbeta          = log_constrained_backward(seq, log_trans_mat, log_emission_mat, labels)
    # loggamma, logxi  = compute_log_gamma_xi(logalpha, logbeta, log_trans_mat, log_emission_mat, seq)

    # def show(mat, name, exp=False):
    #     print(f"\n{name} shape={mat.shape}")
    #     print(mat if not exp else np.exp(mat))
    # show(logalpha, "logalpha")
    # show(np.exp(logalpha), "alpha (prob)", exp=False)
    # show(logbeta, "logbeta")
    # show(np.exp(logbeta), "beta (prob)")
    # show(loggamma, "loggamma")
    # show(np.exp(loggamma), "gamma (prob)")
    # show(logxi, "logxi")
    # for t in range(seq.shape[0]-1):
    #     print(f"\nxi at time {t} (prob, sums to 1):\n", np.exp(logxi[:,:,t]))

    # print("\nSequence log-likelihood (from alpha):", loglik)

    # 4. Run full training
    print("\nTraining on observations")

    best_model, all_models = multi_restart_log_cbw(sequences,
                                                   labels_list,
                                                   num_states=2,
                                                   num_symbols=2,
                                                   early_stop=1e-8,
                                                   n_restarts=5,
                                                   n_jobs=-1,
                                                   model_selection=False,
                                                #    decoder='viterbi',
                                                   verbose=True)

    # print("Log-likelihood history:\n", best_model['loglik_history'])
    print("Final Initial prob matrix:\n", best_model['best_params'][0])
    print("Final Transition matrix:\n", best_model['best_params'][1])
    print("Final Emission matrix:\n", best_model['best_params'][2])

    init_prob_mat, trans_mat, emission_mat = best_model['best_params']

    for i, seq in enumerate(sequences):
        preds = viterbi_decode(init_prob_mat, trans_mat, emission_mat, seq)
        labels = labels_list[i]

        # only evaluate positions with labels
        mask = labels >= 0
        y_true = labels[mask]
        y_pred = preds[mask]

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"\nSequence {i}:")
        print(f"Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")

if __name__ == "__main__":
    main()
