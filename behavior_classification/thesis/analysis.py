import sys
import os
import time

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from matplotlib.lines import Line2D

from pathlib import Path
from glob import glob

from sklearn.calibration import calibration_curve
from sklearn.metrics import (precision_recall_curve, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, brier_score_loss,
                             PrecisionRecallDisplay)

DIR_PREFIX = 'verify_paper_results_xgb_es50_depth3_child1_wnd'

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from behavior_classification.thesis.parsers import find_parse_annot_files


def read_dill_file(file_path):
    with open(file_path, 'rb') as f:
        return dill.load(f)


def plot_xgb_proba_diags(proba,
                         labels,
                         save_path=None,
                         bin_edges=10,
                         threshold=0.5,
                         log_scale=False,
                         class_names=("Absence (0)", "Presence (1)")):
    """Plot histogram of predicted probas."""
    plt.figure(figsize=(8, 5))

    proba = np.asarray(proba)
    labels = np.asarray(labels)

    proba_0 = proba[labels == 0]
    proba_1 = proba[labels == 1]

    plt.hist(proba_0, bins=bin_edges, density=True, alpha=0.5, label=f"True {class_names[0]}",
             color="#4C72B0", edgecolor="none")
    plt.hist(proba_1, bins=bin_edges, density=True, alpha=0.5, label=f"True {class_names[1]}",
             color="#DD8452", edgecolor="none")
    plt.vlines(bin_edges, ymin=1e-4, ymax=plt.ylim()[1], color='gray', lw=0.5, alpha=0.5)

    # trheshold line
    plt.axvline(threshold, color="black", linestyle="--", lw=1)
    # mean and std lines
    mean0, std0 = np.mean(proba_0), np.std(proba_0)
    mean1, std1 = np.mean(proba_1), np.std(proba_1)

    plt.axvline(mean0, color="#4C72B0", linestyle=":", lw=1)
    plt.axvline(mean1, color="#DD8452", linestyle=":", lw=1)

    plt.text(mean0, plt.ylim()[1]*0.7, f"μ₀={mean0:.2f}\nσ₀={std0:.2f}", color="#4C72B0",
             fontsize=9, ha="center")
    plt.text(mean1, plt.ylim()[1]*0.7, f"μ₁={mean1:.2f}\nσ₁={std1:.2f}", color="#DD8452",
             fontsize=9, ha="center")

    plt.xlabel("Predicted Probability of Presence", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("XGBoost Predicted Probability Distribution by True Class", fontsize=13, pad=12)
    plt.legend(fontsize=10, loc='lower left')
    plt.xticks(np.arange(0, 1.05, 0.05), rotation=45)
    plt.grid(alpha=0.25, axis='y')

    if log_scale:
        plt.yscale("log")
        plt.ylabel("Density (log scale)", fontsize=12)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, facecolor='white', edgecolor='white')
        print(f"Saved probability diagnostic plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_calibration_curve(proba, labels, plot_label, save_path=None, n_bins=10,
                           strategy='uniform'):
    """Plot calibration curve comparing predicted vs. true probability."""
    prob_true, prob_pred = calibration_curve(labels, proba, n_bins=n_bins, strategy=strategy)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', markersize=4, linestyle='-', color="black",
             linewidth=1, label=plot_label)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1, label='Perfect calibration')
    plt.xlabel("Predicted probability", fontsize=12)
    plt.ylabel("Observed frequency", fontsize=12)
    plt.title("Calibration Curve", fontsize=13)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, facecolor='white', edgecolor='white')
        print(f"Saved calibration plot to {save_path}")
    else:
        plt.show()


def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    if len(prob_true) == 0:
        return np.nan
    hist, _ = np.histogram(y_prob, bins=n_bins, range=(0, 1))
    hist = hist[:len(prob_true)]
    if hist.sum() == 0:
        return np.nan
    weights = hist / hist.sum()
    return np.sum(weights * np.abs(prob_true - prob_pred))


def compute_metrics(gt_bin, pred, prob):
    return {
        "precision": precision_score(gt_bin, pred, zero_division=0),
        "recall": recall_score(gt_bin, pred, zero_division=0),
        "f1": f1_score(gt_bin, pred, zero_division=0),
        "roc_auc": roc_auc_score(gt_bin, prob[:, 1]),
        "ap": average_precision_score(gt_bin, prob[:, 1]),
        "brier": brier_score_loss(gt_bin, prob[:, 1]),
        "ece": expected_calibration_error(gt_bin, prob[:, 1])
    }


def summarize_run(clf_dir):
    """Calculates and appends test metrics to all classifier files in a run directory."""
    results_path = os.path.join(clf_dir, "results.dill")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No results.dill found in {clf_dir}")

    results = dill.load(open(results_path, "rb"))

    classifier_files = [
        f for f in os.listdir(clf_dir)
        if f.startswith("classifier_") and "_results" not in f
    ]
    if not classifier_files:
        raise FileNotFoundError(f"No base classifier_* files found in {clf_dir}")

    print(f"Found {len(classifier_files)} classifiers to update in {clf_dir}")

    gt = np.asarray(results["0_G"]).ravel()
    preds_mars = np.asarray(results["2_pd_fbs_hmm"])
    preds_cbw = np.asarray(results["7_pd_fbs_hmm_cbw"])
    proba_mars = np.asarray(results["4_proba_pd_hmm_fbs"])
    proba_cbw = np.asarray(results["8_proba_pd_hmm_fbs_cbw,"])

    for clf_file in classifier_files:
        clf_path = os.path.join(clf_dir, clf_file)
        clf_data = dill.load(open(clf_path, "rb"))

        beh_name = clf_data.get("beh_name", "unknown")
        beh_id = clf_data.get("beh_id", None)

        if beh_id is None:
            print(f"Skipping {clf_file} — missing 'beh_id'.")
            continue

        print(f"Computing metrics for '{beh_name}' (id={beh_id})")

        gt_bin = (gt == beh_id).astype(int)

        metrics_mars = compute_metrics(gt_bin, preds_mars[:, beh_id], proba_mars[:, beh_id])
        metrics_cbw = compute_metrics(gt_bin, preds_cbw[:, beh_id], proba_cbw[:, beh_id])

        for k, v in metrics_mars.items():
            clf_data[f"{k}_mars_test"] = v
        for k, v in metrics_cbw.items():
            clf_data[f"{k}_cbw_test"] = v

        dill.dump(clf_data, open(clf_path, "wb"))
        print(f"Updated: {clf_file}")

    print("\nAll classifier files updated successfully.")


def aggregate_metrics_across_runs(run_parent_folder, behavior_name, save_csv=True):
    """Aggregates test metrics across multiple runs for a given strat/pct and behavior."""
    metrics = ["precision", "recall", "f1", "roc_auc", "ap", "brier", "ece"]
    methods = ["mars", "cbw"]
    data = {f"{m}_{meth}": [] for m in metrics for meth in methods}

    # Search all runs inside strat/pct folder
    run_dirs = [os.path.join(run_parent_folder, d) for d in os.listdir(run_parent_folder)
                if os.path.isdir(os.path.join(run_parent_folder, d))]

    for rdir in run_dirs:
        clf_path = os.path.join(rdir, f"classifier_{behavior_name}")
        if not os.path.exists(clf_path):
            continue
        try:
            clf = dill.load(open(clf_path, "rb"))
            for m in metrics:
                for meth in methods:
                    key = f"{m}_{meth}_test"
                    val = clf.get(key, np.nan)
                    data[f"{m}_{meth}"].append(val)
        except Exception as e:
            print(f"Error loading {clf_path}: {e}")
            continue

    df = pd.DataFrame(data)
    means = df.mean().rename(lambda x: f"{x}_mean")
    stds = df.std().rename(lambda x: f"{x}_std")

    summary_df = pd.concat([means, stds], axis=0)
    print(f"Aggregated {len(df)} runs for behavior '{behavior_name}' in {run_parent_folder}")

    if save_csv:
        csv_name = f"summary_metrics_{behavior_name}.csv"
        csv_path = os.path.join(run_parent_folder, csv_name)
        summary_df.to_csv(csv_path)
        print(f"Saved summary to: {csv_path}")

    return summary_df


def find_run_dirs(base_dir, strat, pct):
    """Find all run dirs matching given strat and pct"""
    pct_str = str(pct).strip()

    pattern = os.path.join(
        base_dir,
        f"{DIR_PREFIX}_{strat}_{pct_str}pct*"
    )

    dirs = [d for d in glob(pattern) if os.path.isdir(d)]
    dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True) # newest first

    if not dirs:
        print(f"No run folders matched for strat='{strat}', pct='{pct_str}'.")
    else:
        print(f"Found {len(dirs)} run dir(s) for strat='{strat}', pct='{pct_str}':")
        for d in dirs[:5]:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(d)))
            print(f"   - {d}  (mtime: {ts})")
        if len(dirs) > 5:
            print(f"   ... ({len(dirs)-5} more)")
    return dirs


def interpolate_pr_on_fixed_recall(precisions, recalls, recall_grid):
    return np.interp(recall_grid, recalls[::-1], precisions[::-1], left=precisions[0],
                     right=precisions[-1])


def calc_prc_tresh(gt, proba, thr):
    """Return precision and recall at the exact threshold value."""
    y_pred = (proba >= thr).astype(int)
    tp = np.sum((y_pred == 1) & (gt == 1))
    fp = np.sum((y_pred == 1) & (gt == 0))
    fn = np.sum((y_pred == 0) & (gt == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    return precision, recall


def load_run_prc_data(folder, behavior):
    path_mars = os.path.join(folder, f"classifier_{behavior}_results.mat")
    path_cbw = os.path.join(folder, f"classifier_{behavior}_results_cbw.mat")
    if not (os.path.exists(path_mars) and os.path.exists(path_cbw)):
        return None, None
    mars_data = sio.loadmat(path_mars)
    cbw_data = sio.loadmat(path_cbw)
    return (mars_data["gt"].ravel(), mars_data["proba"].ravel()), (cbw_data["gt"].ravel(),
                                                                   cbw_data["proba"].ravel())


def plot_avg_prc_for_strat(base_dir, behavior, strat, pct, recall_grid=None, show_std=True,
                           save_path=None, threshold=0.5):
    """Plot average PRC curves across multiple runs for given strat/pct and behavior."""
    run_dirs = find_run_dirs(base_dir, strat, pct)
    if not run_dirs:
        print(f"No matching runs found for strat='{strat}', pct='{pct}'")
        return

    recall_grid = recall_grid if recall_grid is not None else np.linspace(0, 1, 100)
    mars_curves, cbw_curves = [], []
    mars_thresh_pts, cbw_thresh_pts = [], []

    for run_dir in run_dirs:
        mars_cbw = load_run_prc_data(run_dir, behavior)
        if mars_cbw is None:
            continue
        (gt_mars, proba_mars), (gt_cbw, proba_cbw) = mars_cbw

        for label, gt, proba, collector, thresh_pts in [
            ("MARS", gt_mars, proba_mars, mars_curves, mars_thresh_pts),
            ("CBW", gt_cbw, proba_cbw, cbw_curves, cbw_thresh_pts)]:

            if len(np.unique(gt)) < 2:
                continue
            precision, recall, thresholds = precision_recall_curve(gt, proba)
            interp_precision = interpolate_pr_on_fixed_recall(precision, recall, recall_grid)
            collector.append(interp_precision)

            prec_05, rec_05 = calc_prc_tresh(gt, proba, threshold)
            thresh_pts.append((prec_05, rec_05))

    def plot_curve(ax, recall_grid, curves, label, color, linestyle, marker_recalls, marker_style):
        if not curves:
            print(f"No {label} curves for '{behavior}'")
            return
        curves = np.vstack(curves)
        mean_p = np.mean(curves, axis=0)
        std_p = np.std(curves, axis=0)
        ax.plot(recall_grid, mean_p, label=label, color=color, linestyle=linestyle)

        if show_std:
            ax.fill_between(recall_grid, mean_p - std_p, mean_p + std_p, color=color, alpha=0.2)

        if marker_recalls:
            mean_r = np.nanmean(marker_recalls)
            r_idx = np.argmin(np.abs(recall_grid - mean_r))
            r_final = recall_grid[r_idx]
            p_final = mean_p[r_idx]
            ax.scatter(r_final, p_final, s=110, color=color, edgecolor='black', marker=marker_style,
                       linewidth=1.2, zorder=10, label=f"_nolegend_")

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_curve(ax, recall_grid, mars_curves, "MARS", "tab:blue", '-', mars_thresh_pts, 'o')
    plot_curve(ax, recall_grid, cbw_curves, "CBW", "tab:red", '--', cbw_thresh_pts, 'D')

    ax.set_title(f"Avg PRC — {behavior.capitalize()} ({strat}, {float(pct)*100:.2f}%)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(True, linestyle='--', alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    shared_handles = [
        Line2D([0], [0], marker='o', color='black', markersize=8, linestyle='None', label='T@0.5 MARS'),
        Line2D([0], [0], marker='D', color='black', markersize=8, linestyle='None', label='T@0.5 CBW')]

    ax.legend(handles + shared_handles, labels + ['T@0.5 MARS', 'T@0.5 CBW'],
              loc='lower left', frameon=True, title="Variant")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved PRC plot: {save_path}")
    plt.show()


def plot_pr(ax, y, p, label, color, variant,
            show_thresh=True, show_f1=False, threshold=0.5):
    """Single PR curve helper"""
    precision, recall, thresholds = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)

    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot(
        ax=ax,
        name=f"{label} (AP={ap:.3f})",
        color=color,
        linestyle='--' if variant == "cbw" else '-',
        drawstyle="steps-post"
    )
    ax = disp.ax_

    if show_thresh:
        prec_05, rec_05 = calc_prc_tresh(y, p, threshold)
        if np.isfinite(prec_05) and np.isfinite(rec_05):
            marker = "o" if variant == "mars" else "D"
            ax.scatter(rec_05, prec_05,
                       s=110, marker=marker, color=color,
                       edgecolor="black", linewidths=1.0,
                       zorder=12, label="_nolegend_")

    if show_f1:
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        j = np.nanargmax(f1)
        ax.scatter(recall[j], precision[j], s=140, marker='*', color=color,
                   edgecolor='black', linewidth=1.2)
    return ax


def plot_pr_curves(base_path,
                   strats,
                   pcts,
                   behaviors,
                   exact_folder=None,
                   show_thresh=True,
                   show_f1=False,
                   save_figs=True,
                   threshold=0.5):
    """Plot PR curves for given sampling strategies and percentages.
    - SINGLE MODE: Created unified PRC for all behaviors for given strat, pct
    - MULTI MODE: Plots multiple curves for strat/pct combinations separated by behavior.
    """
    behavior_colors = {
        'attack': 'tab:red',
        'investigation': 'tab:green',
        'mount': 'tab:blue'
        }

    if exact_folder is not None:
        single_mode = True
    else:
        single_mode = (len(strats) == 1 and len(pcts) == 1)
    print(f"\nMode: {'Single' if single_mode else 'Multi'}\n")

    # SINGLE MODE
    if single_mode:
        strat, pct = strats[0], pcts[0]
        if exact_folder is not None:
            run_dirs = [exact_folder]
        else:
            run_dirs = find_run_dirs(base_path, strat, pct)
        if not run_dirs:
            return
        run_dir = run_dirs[0]
        print(f"Using newest run dir: {run_dir}")

        fig, ax = plt.subplots(figsize=(9, 7))
        plotted = False

        for beh in behaviors:
            mars_cbw = load_run_prc_data(run_dir, beh)
            if mars_cbw is None or mars_cbw[0] is None:
                print(f"Missing outputs for {beh} in {run_dir}")
                continue

            (y_m, p_m), (y_c, p_c) = mars_cbw
            color = behavior_colors.get(beh, 'tab:gray')

            plot_pr(ax, y_m, p_m, f"{beh.capitalize()} MARS", color, "mars",
                    show_thresh, show_f1, threshold=threshold)
            plot_pr(ax, y_c, p_c, f"{beh.capitalize()} CBW", color, "cbw",
                    show_thresh, show_f1, threshold=threshold)
            plotted = True

        if not plotted:
            print("Nothing plotted (missing files).")
            return

        ax.set_title(f"Precision–Recall Curves ({strat}, {float(pct)*100:.2f}%)", fontsize=15)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(alpha=0.4, linestyle="--")
        ax.legend(loc="lower left", frameon=True, title="Behavior / Variant")

        handles, labels = ax.get_legend_handles_labels()

        shared_handles = [
            Line2D([0], [0], marker='o', color='black', markersize=9,
                   linestyle='None', label='T@0.5 MARS'),
            Line2D([0], [0], marker='D', color='black', markersize=9,
                   linestyle='None', label='T@0.5 CBW')
        ]

        ax.legend(handles + shared_handles, labels + ['T@0.5 MARS', 'T@0.5 CBW'],
                  loc='lower left', frameon=True, title="Behavior / Variant")

        ax.set_xlim([0.0, 1.01])
        ax.set_ylim([0.0, 1.01])
        plt.tight_layout()

        if save_figs:
            out = os.path.join(run_dir, f"PRC_{strat}_{pct}.png")
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"Saved: {out}")

        plt.show()
        return
    else:
        # MULTI MODE
        combo_list = [(s, p) for s in strats for p in pcts]
        cmap = plt.cm.get_cmap("tab20", max(1, len(combo_list)))
        COLOR_MAP = {combo: cmap(i % 20) for i, combo in enumerate(combo_list)}

        for beh in behaviors:
            fig, ax = plt.subplots(figsize=(10, 8))
            plotted_any = False

            for strat in strats:
                for pct in pcts:
                    run_dirs = find_run_dirs(base_path, strat, pct)
                    if not run_dirs:
                        continue
                    run_dir = run_dirs[0]

                    mars_cbw = load_run_prc_data(run_dir, beh)
                    if mars_cbw is None or mars_cbw[0] is None:
                        continue

                    try:
                        (y_m, p_m), (y_c, p_c) = mars_cbw
                    except Exception as e:
                        print(f"Error reading {beh} {strat} {pct}: {e}")
                        continue

                    color = COLOR_MAP[(strat, pct)]

                    # Same color for MARS/CBW, different linestyle; threshold markers
                    plot_pr(ax, y_m, p_m,
                            f"{strat} {float(pct)*100:.1f}% MARS", color, "mars",
                            show_thresh=True, show_f1=False, threshold=threshold)
                    plot_pr(ax, y_c, p_c,
                            f"{strat} {float(pct)*100:.1f}% CBW", color, "cbw",
                            show_thresh=True, show_f1=False, threshold=threshold)
                    plotted_any = True

            if plotted_any:
                ax.set_title(f'Precision–Recall — {beh.capitalize()}', fontsize=15)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.grid(alpha=0.4, linestyle="--")
                ax.legend(title="Sampling / Variant", fontsize=9,
                          loc="lower left", frameon=True)

                handles, labels = ax.get_legend_handles_labels()
                shared_handles = [
                    Line2D([0], [0], marker='o', color='black', markersize=9,
                           linestyle='None', label='T@0.5 MARS'),
                    Line2D([0], [0], marker='D', color='black', markersize=9,
                           linestyle='None', label='T@0.5 CBW')
                ]

                ax.legend(handles + shared_handles, labels + ['T@0.5 MARS', 'T@0.5 CBW'],
                          loc='lower left', frameon=True, title="Sampling / Variant")

                ax.set_xlim([0.0, 1.01])
                ax.set_ylim([0.0, 1.01])
                plt.tight_layout()

                if save_figs:
                    summary_out = os.path.join(base_path, f"PRC_multi_{beh}.png")
                    plt.savefig(summary_out, dpi=300, bbox_inches="tight")
                    print(f"Saved summary: {summary_out}")

                plt.show()
            else:
                print(f"No data found for behavior: {beh}")


def plot_mean_bout_duration(df, skip_behaviors=None, save_path=None):
    """Plots mean bout duration by behavior."""
    if skip_behaviors:
        df = df[~df.index.isin(skip_behaviors)] # filter df where skip_behaviors
    if df.empty:
        print("Warning: No behaviors left to plot after skipping.")
        return

    plt.figure()
    df['mean_bout_duration'].plot(kind='bar', title='Mean Bout Duration by Behavior')
    plt.ylabel('Mean Duration (seconds)')
    plt.tight_layout()

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}bout_duration_mean.png")
    plt.show()


def plot_median_bout_duration(df, skip_behaviors=None, save_path=None):
    """Plots the median bout duration for each behavior in a bar plot."""
    if skip_behaviors:
        df = df[~df.index.isin(skip_behaviors)]
    if df.empty:
        print("Warning: No behaviors left to plot after skipping.")
        return

    plt.figure()
    df['median_bout_duration'].plot(kind='bar', title='Median Bout Duration by Behavior')
    plt.ylabel('Median Duration (seconds)')
    plt.xlabel('Behavior')
    plt.tight_layout()

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}bout_duration_median.png")
    plt.show()


def plot_bout_duration_histogram(df, skip_behaviors=None, save_path=None):
    """Plots bout duration histogram by behavior."""
    if skip_behaviors:
        df = df[~df.index.isin(skip_behaviors)]  # filter df where skip_behaviors
    if df.empty:
        print("Warning: No behaviors left to plot after skipping.")
        return

    plt.figure()
    for behavior in df.index:
        plt.hist(df.loc[behavior, 'bout_durations'], alpha=0.5, label=behavior)
    plt.legend(loc='upper right')
    plt.xlabel('Bout Duration (seconds)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Bout Durations')

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}bout_duration_histogram.png")
    plt.show()


def plot_bout_duration_box(df, skip_behaviors=None, save_path=None):
    """Plots bout duration box plot by behavior."""
    if skip_behaviors:
        df = df[~df.index.isin(skip_behaviors)]  # filter df where skip_behaviors
    if df.empty:
        print("Warning: No behaviors left to plot after skipping.")
        return

    durations = df['bout_durations'].apply(pd.Series).stack()
    durations.index = durations.index.droplevel(-1)
    durations.name = 'bout_duration'
    df_boxplot = df.drop('bout_durations', axis=1).join(durations)

    fig, ax = plt.subplots(figsize=(10, 6))
    df_boxplot.boxplot(column='bout_duration',
                       by=df_boxplot.index,
                       ax=ax, vert=True,
                       patch_artist=True,
                       showfliers=False)

    plt.ylabel('Bout Duration (seconds)')
    plt.xlabel('Behavior')
    plt.title('Distribution of Bout Durations by Behavior')
    plt.suptitle('')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}bout_duration_boxplot.png")
    plt.show()


def calculate_bout_properties(bout_data, total_frames, framerate: float = 30.0) -> dict:
    """Calculates bout properties."""
    if len(bout_data) == 0:
        return {
            "num_bouts": 0,
            "bout_lengths": [],
            "bout_durations": [],
            "mean_bout_length": 0,
            "median_bout_length": 0,
            "std_bout_length": 0,
            "mean_bout_duration": 0,
            "median_bout_duration": 0,
            "std_bout_duration": 0,
            "proportion_frames": 0,
            "min_bout_length": 0,
            "max_bout_length": 0,
            "min_bout_duration": 0,
            "max_bout_duration": 0,
        }
    bout_lengths = bout_data['collected_bouts'][:, 2]
    # bout_lengths = bout_data[:, 1] - bout_data[:, 0] + 1
    bout_durations = bout_lengths / framerate

    num_bouts = bout_data['bout_count']
    mean_bout_length = np.mean(bout_lengths)
    median_bout_length = np.median(bout_lengths)
    std_bout_length = np.std(bout_lengths)
    mean_bout_duration = np.mean(bout_durations)
    median_bout_duration = np.median(bout_durations)
    std_bout_duration = np.std(bout_durations)
    proportion_frames = np.sum(bout_lengths) / total_frames

    min_bout_length = np.min(bout_lengths)
    max_bout_length = np.max(bout_lengths)
    min_bout_duration = np.min(bout_durations)
    max_bout_duration = np.max(bout_durations)

    properties = {
        "num_bouts": num_bouts,
        "bout_lengths": list(bout_lengths),
        "bout_durations": list(bout_durations),
        "mean_bout_length": mean_bout_length,
        "median_bout_length": median_bout_length,
        "std_bout_length": std_bout_length,
        "mean_bout_duration": mean_bout_duration,
        "median_bout_duration": median_bout_duration,
        "std_bout_duration": std_bout_duration,
        "proportion_frames": proportion_frames,
        "min_bout_length": min_bout_length,
        "max_bout_length": max_bout_length,
        "min_bout_duration": min_bout_duration,
        "max_bout_duration": max_bout_duration,
    }
    return properties


def print_bout_summary(bout_properties, total_frames, skip_behaviors=None, save_path=None):
    """Prints a formatted summary of bout properties for each behavior, skipping specified behaviors."""
    output_string = f"Total frames across all files: {format(total_frames, ',').replace(',', '.') }\n\n"
    for behavior, properties in bout_properties.items():
        if skip_behaviors and behavior in skip_behaviors:
            continue
        output_string += f"--- Behavior: {behavior} ---\n"
        output_string += f"  Number of bouts: {properties['num_bouts']}\n"
        output_string += f"  Mean bout duration: {properties['mean_bout_duration']:.2f} seconds\n"
        output_string += f"  Median bout duration: {properties['median_bout_duration']:.2f} seconds\n"
        output_string += f"  Std. dev. bout duration: {properties['std_bout_duration']:.2f} seconds\n"
        output_string += f"  Min bout duration: {properties['min_bout_duration']:.2f} seconds\n"
        output_string += f"  Max bout duration: {properties['max_bout_duration']:.2f} seconds\n"
        output_string += "\n"
        output_string += f"  Mean bout length: {properties['mean_bout_length']:.2f} frames\n"
        output_string += f"  Median bout length: {properties['median_bout_length']:.2f} frames\n"
        output_string += f"  Std. dev. bout length: {properties['std_bout_length']:.2f} frames\n"
        output_string += f"  Min bout length: {properties['min_bout_length']:.2f} frames\n"
        output_string += f"  Max bout length: {properties['max_bout_length']:.2f} frames\n"
        output_string += f"  Proportion of frames: {properties['proportion_frames']:.2%}\n"
        output_string += "\n"

    print(output_string)

    if save_path:
        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(f"{save_path}bout_summary.txt", 'w', encoding='utf-8') as f:
                f.write(output_string)
            print(f"Bout summary saved to {save_path}bout_summary.txt")
        except Exception as e:
            print(f"Error saving bout summary to file: {e}")


def analyze_behavior_transitions(annotations):
    """
    Analyzes transitions between behavior bouts within each annotation file.
    """
    transitions = {}

    for ann_dict in annotations:
        if 'behs_frame' in ann_dict:
            frame_annotations = ann_dict['behs_frame']
            n = len(frame_annotations)
            i = 0

            while i < n - 1:
                current_behavior = frame_annotations[i]
                j = i + 1
                while j < n and frame_annotations[j] == current_behavior:
                    j += 1
                if j < n:
                    next_behavior = frame_annotations[j]
                    if current_behavior != next_behavior:
                        transition = (current_behavior, next_behavior)
                        transitions[transition] = transitions.get(transition, 0) + 1
                i = j
    return transitions


def print_behavior_transitions(transitions, save_path=None):
    """Prints behavior transition counts, skipping transitions involving specified behaviors."""
    output_string = "--- Behavior Transitions ---\n"
    transition_counts = np.array(list(transitions.values()))
    total_transition_count = np.sum(transition_counts)
    for (beh1, beh2), count in sorted(transitions.items(), key=lambda x: x[1], reverse=True):
        output_string += f"Transition: {beh1} -> {beh2}, Count: {count}\n"

    output_string += f"Total Transitions: {total_transition_count}\n"
    print(output_string)

    if save_path:
        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(f"{save_path}behavior_transitions.txt", 'w', encoding='utf-8') as f:
                f.write(output_string)
            print(f"Behavior transition saved to {save_path}behavior_transitions.txt")
        except Exception as e:
            print(f"Error saving behavior transitions to file: {e}")


def plot_bout_summary():
    ROOT_DIR = "./verify_paper_results/behavior/behavior_data/"
    # ROOT_DIR = "./verify_paper_results/behavior/behavior_data/train"
    # ROOT_DIR = "./verify_paper_results/behavior/behavior_data/validation"
    # ROOT_DIR = "./verify_paper_results/behavior/behavior_data/test_1"
    PLOT_SAVE_DIR = "./verify_paper_results/behavior/trained_classifiers/dataset_analysis/full_dataset/plots/"
    # PLOT_SAVE_DIR = None
    SKIP_BEHAVIORS = ['other']

    # 1. Load annotation files
    annotations = find_parse_annot_files(ROOT_DIR)

    # 2. Collect all unique behavior names across all files
    all_behavior_names = set()
    for ann_dict in annotations:
        if 'behs_frame' in ann_dict:
            all_behavior_names.update(ann_dict['behs_frame'])

    # 3. Apply skip filter
    all_behavior_names = sorted(list(all_behavior_names.difference(SKIP_BEHAVIORS)))
    print("All behaviors considered for analysis:", all_behavior_names)

    # 4. Extract bouts from each annotation file
    all_behavior_data = {}
    n_frames = 0
    for behavior in all_behavior_names:
        all_behavior_data[behavior] = {
            'bout_count': 0,
            'collected_bouts': np.empty((0, 3), dtype=int),  # Initialize an empty NumPy array
            'total_frames': 0
        }

    for ann_dict in annotations:
        if 'behs_frame' not in ann_dict:
            continue

        frame_annotations = ann_dict['behs_frame']
        n_frames += ann_dict.get('nFrames', len(frame_annotations))

        for behavior in all_behavior_names:
            # Count total bouts / duration / total frames by behavior
            if behavior in ann_dict['behs_bout']['Ch1']:
                bouts = ann_dict['behs_bout']['Ch1'][behavior]

                differences = np.diff(bouts, axis=1)
                bouts_start_end_duration = np.concatenate((bouts, differences), axis=1)

                all_behavior_data[behavior]['collected_bouts'] = \
                    np.concatenate((all_behavior_data[behavior]['collected_bouts'],
                                    bouts_start_end_duration))
                all_behavior_data[behavior]['bout_count'] += bouts.shape[0]
                all_behavior_data[behavior]['total_frames'] += n_frames

    bout_properties = {}
    for behavior in all_behavior_names:
        bout_properties[behavior] = calculate_bout_properties(
            all_behavior_data[behavior], n_frames
        )

    # 5. Print bout summary
    print_bout_summary(
        bout_properties, n_frames,
        skip_behaviors=SKIP_BEHAVIORS,
        save_path=PLOT_SAVE_DIR
    )

    # 6. Analyze behavior transitions
    transitions = analyze_behavior_transitions(annotations)
    print_behavior_transitions(transitions, save_path=PLOT_SAVE_DIR)

    # 7. Optional plotting
    df = pd.DataFrame.from_dict(bout_properties, orient='index')
    plot_mean_bout_duration(df, skip_behaviors=SKIP_BEHAVIORS, save_path=PLOT_SAVE_DIR)
    plot_median_bout_duration(df, skip_behaviors=SKIP_BEHAVIORS, save_path=PLOT_SAVE_DIR)
    plot_bout_duration_histogram(df, skip_behaviors=SKIP_BEHAVIORS, save_path=PLOT_SAVE_DIR)
    plot_bout_duration_box(df, skip_behaviors=SKIP_BEHAVIORS, save_path=PLOT_SAVE_DIR)


if __name__ == "__main__":
    # plot_bout_summary()

    BEHAVIORS = ["attack", "investigation", "mount"]
    DIR_PREFIX = 'verify_paper_results_xgb_es50_depth3_child1_wnd'

    BASE_PATH = './verify_paper_results/behavior/trained_classifiers/'#analysis/2x10_hmms/unsmoothed_dirichilet_soft_a'

    STRATS = ["natural", "simple_random", "stratified", "systematic", "cluster", "purposive"]
    PCTS = ['0.01667','0.03334', '0.06668', '0.13336']
    # STRATS = ["natural"]

    plot_pr_curves(
        base_path=BASE_PATH,
        strats=STRATS,
        pcts=PCTS,
        behaviors=BEHAVIORS,
        exact_folder="./verify_paper_results/behavior/trained_classifiers/verify_paper_results_xgb_es50_depth3_child1_wnd_cluster_0.01667pct_20251114_033412",
        show_thresh=True,
        show_f1=False,
        save_figs=True,
        threshold=0.5
    )

    # for beh in BEHAVIORS:
    #     for strat in STRATS:
    #         plot_avg_prc_for_strat(
    #             base_dir=f"verify_paper_results/behavior/trained_classifiers/xx_plot/average_smoothed_sig_1.5/{strat}",
    #             behavior=beh,
    #             strat=strat,
    #             pct=PCTS[1],
    #             recall_grid=np.linspace(0, 1, 200),  # finer resolution
    #             show_std=True,
    #             save_path=f"verify_paper_results/behavior/trained_classifiers/xx_plot/average_smoothed_sig_1.5/{strat}/{strat}_{PCTS[1]}_{beh}.png")

    # summarize_run(f"./verify_paper_results/behavior/trained_classifiers/xx_plot/average_smoothed_sig_1.5/natural/verify_paper_results_xgb_es50_depth3_child1_wnd_natural_0.03334pct_20251109_162153")
    # aggregate_metrics_across_runs(
    #     run_parent_folder=f"./verify_paper_results/behavior/trained_classifiers/xx_plot/average_smoothed_sig_1.5/natural",
    #     behavior_name="attack")
