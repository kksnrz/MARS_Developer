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

# Nature-style muted palette
NATURE_COLORS = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#937860",  # brown
]

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


def list_run_dirs(base_dir, strat=None, pct=None):
    """Return sorted run directories inside the given base_dir."""
    if not os.path.isdir(base_dir):
        print(f"[WARN] Base dir does not exist: {base_dir}")
        return []

    if pct is not None:
        pct_str = str(pct).strip()
    else:
        pct_str = None

    if strat is not None and pct is not None:
        pattern = os.path.join(base_dir, f"{DIR_PREFIX}_{strat}_{pct_str}pct_*")
    else:
        pattern = os.path.join(base_dir, f"{DIR_PREFIX}_*")

    run_dirs = [d for d in glob(pattern) if os.path.isdir(d)]
    run_dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)

    if not run_dirs:
        print(f"[INFO] No run directories found in: {base_dir}")
    else:
        print(f"[OK] Found {len(run_dirs)} run dirs in: {base_dir}")

    return run_dirs


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


def load_baseline_curve(baseline_run_dir, behavior, recall_grid=None, threshold=0.5):
    """Load a single fully-labeled baseline run for a behavior (MARS only)."""
    if baseline_run_dir is None:
        return None

    if not os.path.isdir(baseline_run_dir):
        print(f"[BASELINE WARN] baseline_run_dir does not exist: {baseline_run_dir}")
        return None

    data = load_run_prc_data(baseline_run_dir, behavior)
    if data is None:
        print(f"[BASELINE WARN] Missing baseline files for {behavior} in {baseline_run_dir}")
        return None

    (gt_m, p_m), _ = data  # use supervised baseline

    if len(np.unique(gt_m)) < 2:
        print(f"[BASELINE WARN] Not enough positive/negative samples for {behavior}")
        return None

    prec, rec, _ = precision_recall_curve(gt_m, p_m)
    ap = average_precision_score(gt_m, p_m)
    thr_prec, thr_rec = calc_prc_tresh(gt_m, p_m, threshold)

    if recall_grid is not None:
        prec_interp = interpolate_pr_on_fixed_recall(prec, rec, recall_grid)
        rec_use = recall_grid
        prec_use = prec_interp
    else:
        rec_use = rec
        prec_use = prec

    return {
        "recall": rec_use,
        "precision": prec_use,
        "ap": ap,
        "thr_prec": thr_prec,
        "thr_rec": thr_rec
    }


def _plot_mean_curve(ax, curves, recall_grid, color, linestyle, label,
                     thr_points=None, marker='o'):
    """Plot mean PR curve from a list of interpolated precision arrays (same recall_grid)."""
    if not curves:
        return

    C = np.vstack(curves)
    mean_p = C.mean(axis=0)

    ax.plot(recall_grid, mean_p, color=color, lw=1.0, linestyle=linestyle, label=label)

    # Threshold marker (averaged across runs)
    if thr_points:
        thr_points = np.array(thr_points)  # shape: (n_runs, 2) -> (prec, rec)
        mean_prec = np.nanmean(thr_points[:, 0])
        mean_rec = np.nanmean(thr_points[:, 1])
        if np.isfinite(mean_prec) and np.isfinite(mean_rec):
            idx = np.argmin(np.abs(recall_grid - mean_rec))
            r_plot = recall_grid[idx]
            p_plot = mean_p[idx]
            ax.scatter(
                r_plot,
                p_plot,
                s=40,
                marker=marker,
                color=color,
                edgecolor='black',
                linewidths=0.6,
                zorder=10,
                label="_nolegend_"
            )


def _plot_baseline_on_ax(ax, baseline_info, label_suffix="Baseline (16.67%)"):
    """Plot baseline curve with black color and square threshold marker."""
    if baseline_info is None:
        return

    rec = baseline_info["recall"]
    prec = baseline_info["precision"]
    ap = baseline_info["ap"]
    thr_prec = baseline_info["thr_prec"]
    thr_rec = baseline_info["thr_rec"]

    ax.plot(
        rec,
        prec,
        color='black',
        lw=1.2,
        linestyle='-',
        label="_baseline_"
        # label=f"{label_suffix} (AP={ap:.3f})"
    )

    if np.isfinite(thr_prec) and np.isfinite(thr_rec):
        ax.scatter(
            thr_rec,
            thr_prec,
            s=45,
            marker='s',
            color='black',
            edgecolor='white',
            linewidths=0.6,
            zorder=12,
            label="_nolegend_"
        )


def _format_prc_axis(ax, axis_mode="tight"):
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)

    if axis_mode == "zoom":
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0.5, 1.0)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    elif axis_mode == "superzoom":
        ax.set_xlim(0.65, 1.0)
        ax.set_ylim(0.65, 1.0)
        ax.set_xticks([0.7, 0.8, 0.9, 1.0])
        ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    else:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)


def plot_pr(ax, y, p, label, color, variant,
            show_thresh=True, show_f1=False, threshold=0.5):
    """
    Single PR curve helper (no averaging).
    """
    precision, recall, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)

    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot(
        ax=ax,
        name=f"{label} (AP={ap:.3f})",
        color=color,
        linestyle='--' if variant == "cbw" else '-',
        drawstyle="steps-post",
        linewidth=1.0
    )
    ax = disp.ax_

    if show_thresh:
        prec_05, rec_05 = calc_prc_tresh(y, p, threshold)
        if np.isfinite(prec_05) and np.isfinite(rec_05):
            marker = "o" if variant == "mars" else "D"
            ax.scatter(
                rec_05,
                prec_05,
                s=40,
                marker=marker,
                color=color,
                edgecolor="black",
                linewidths=0.6,
                zorder=12,
                label="_nolegend_"
            )

    if show_f1:
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        j = np.nanargmax(f1)
        ax.scatter(
            recall[j],
            precision[j],
            s=60,
            marker='*',
            color=color,
            edgecolor='black',
            linewidth=0.6
        )
    return ax

def _make_three_panel_figure(vstack=False):
    """Create vertical or horizontal aligned three panel figure"""
    if vstack:
        fig = plt.figure(figsize=(6, 10))
        gs = fig.add_gridspec(
            4, 1,
            height_ratios=[1.4, 1.4, 1.4, 0.25],
            hspace=0.5,
            top=0.93, bottom=0.06, left=0.10, right=0.97
        )
        axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    else:
        fig = plt.figure(figsize=(10, 5))
        gs = fig.add_gridspec(
            2, 3,
            height_ratios=[10, 2],
            hspace=0.05,
            wspace=0.18,
            left=0.06,
            right=0.99,
            top=0.90,
            bottom=0.17,
        )
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax in axes:
        ax.set_aspect('equal', adjustable='box')
    return fig, axes


def _build_unified_legend(fig, axes, baseline_label="Baseline (16.67%)", vstack=False):
    """Collect handles/labels from all axes"""
    handles_all, labels_all = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles_all.extend(h)
        labels_all.extend(l)

    label_to_handle = {}
    for h, lab in zip(handles_all, labels_all):
        if lab in ("_nolegend_", "_baseline_"):
            continue
        if lab not in label_to_handle:
            label_to_handle[lab] = h

    extra_handles = [
        Line2D([0], [0], color='black', lw=1.2, linestyle='-',
               label=baseline_label),
        Line2D([0], [0], marker='o', color='black', linestyle='None',
               label='T@0.5 MARS', markersize=5),
        Line2D([0], [0], marker='D', color='black', linestyle='None',
               label='T@0.5 CBW', markersize=5),
        Line2D([0], [0], marker='s', color='black', linestyle='None',
               label='T@0.5 Baseline ', markersize=5),
    ]
    extra_labels = [h.get_label() for h in extra_handles]


    if vstack:
        fig.legend(
            list(label_to_handle.values()) + extra_handles,
            list(label_to_handle.keys()) + extra_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.02),
            ncol=3,
            frameon=True,
            fontsize=8,
        )

        plt.subplots_adjust(bottom=0.22)
    else:
        fig.legend(
            list(label_to_handle.values()) + extra_handles,
            list(label_to_handle.keys()) + extra_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.06),
            ncol=3,
            frameon=True,
            fontsize=8,
        )
        plt.subplots_adjust(bottom=0.12)


def plot_pr_curves(base_path,
                   strats,
                   pcts,
                   behaviors,
                   exact_folder=None,
                   show_thresh=True,
                   show_f1=False,
                   save_figs=True,
                   threshold=0.5,
                   baseline_run_dir=None,
                   axis_mode="tight",
                   vstack=False):
    """Plot exact three panel PR curve"""
    behavior_colors = {
        'attack': NATURE_COLORS[3],
        'investigation': NATURE_COLORS[2],
        'mount': NATURE_COLORS[0]
    }

    if exact_folder is not None:
        single_mode = True
    else:
        single_mode = (len(strats) == 1 and len(pcts) == 1)

    print(f"\nMode: {'Single' if single_mode else 'Multi'}\n")

    if not single_mode:
        raise ValueError("plot_pr_curves is intended for single (strat,pct) or exact_folder.")

    if exact_folder is not None:
        strat, pct = strats, pcts
        run_dir = exact_folder
    else:
        strat, pct = strats[0], pcts[0]
        base_dir = os.path.join(base_path, strat, pct)
        run_dirs = list_run_dirs(base_dir, strat, pct)
        if not run_dirs:
            print("No runs found.")
            return
        run_dir = run_dirs[0]

    print(f"Using run dir: {run_dir}")

    fig, axes = _make_three_panel_figure(vstack=vstack)

    for ax, beh in zip(axes, behaviors):
        mars_cbw = load_run_prc_data(run_dir, beh)
        if mars_cbw is None:
            print(f"Missing outputs for {beh} in {run_dir}")
            continue

        (y_m, p_m), (y_c, p_c) = mars_cbw
        color = behavior_colors.get(beh, NATURE_COLORS[0])

        plot_pr(ax, y_m, p_m, f"{beh.capitalize()} MARS", color, "mars",
                show_thresh, show_f1, threshold=threshold)
        plot_pr(ax, y_c, p_c, f"{beh.capitalize()} CBW", color, "cbw",
                show_thresh, show_f1, threshold=threshold)

        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

        baseline_info = load_baseline_curve(baseline_run_dir, beh, recall_grid=None,
                                            threshold=threshold)
        _plot_baseline_on_ax(ax, baseline_info)

        ax.set_title(beh.capitalize(), fontsize=9)
        ax.set_aspect('auto')
        _format_prc_axis(ax, axis_mode=axis_mode)

    fig.suptitle(f"Precision-Recall ({strat}, {float(pct)*100:.2f}%)", fontsize=11)
    _build_unified_legend(fig, axes, vstack=vstack)

    if save_figs:
        out = os.path.join(run_dir, f"PRC_3panel_{strat}_{pct}.pdf")
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")

    # plt.show()


def plot_pr_curves_all_runs(base_path, strat, pct, behaviors, **kwargs):
    base_dir = os.path.join(base_path, strat, pct)
    run_dirs = list_run_dirs(base_dir, strat, pct)
    if not run_dirs:
        print("No runs found.")
        return

    for rd in run_dirs:
        print(f"\nPlotting for run: {rd}")
        plot_pr_curves(
            base_path=base_path,
            strats=strat,
            pcts=pct,
            behaviors=behaviors,
            exact_folder=rd,
            **kwargs
        )


def plot_avg_prc_for_strat(base_path,
                           behaviors,
                           strat,
                           pct,
                           recall_grid=None,
                           show_std=False,
                           save_path=None,
                           threshold=0.5,
                           baseline_run_dir=None,
                           axis_mode="tight",
                           vstack=False):
    """Plot average three panel PRC for a fixed (strat, pct) across multiple runs."""
    recall_grid = recall_grid if recall_grid is not None else np.linspace(0, 1, 200)
    strat_pct_dir = os.path.join(base_path, strat, pct)
    run_dirs = list_run_dirs(strat_pct_dir, strat, pct)
    if not run_dirs:
        print(f"No matching runs found for strat='{strat}', pct='{pct}'")
        return

    fig, axes = _make_three_panel_figure(vstack=vstack)

    for ax, beh in zip(axes, behaviors):
        mars_curves, cbw_curves = [], []
        mars_thr_pts, cbw_thr_pts = [], []

        for rd in run_dirs:
            mars_cbw = load_run_prc_data(rd, beh)
            if mars_cbw is None:
                continue

            (gt_m, pm), (gt_c, pc) = mars_cbw

            for gt, p, collector, thr_list in [
                (gt_m, pm, mars_curves, mars_thr_pts),
                (gt_c, pc, cbw_curves, cbw_thr_pts)
            ]:
                if len(np.unique(gt)) < 2:
                    continue
                prec, rec, _ = precision_recall_curve(gt, p)
                interp_prec = interpolate_pr_on_fixed_recall(prec, rec, recall_grid)
                collector.append(interp_prec)

                thr_p, thr_r = calc_prc_tresh(gt, p, threshold)
                thr_list.append((thr_p, thr_r))

        color_m = NATURE_COLORS[0]
        color_c = NATURE_COLORS[3]

        _plot_mean_curve(ax, mars_curves, recall_grid, color_m, '-', "MARS",
                         thr_points=mars_thr_pts, marker='o')
        _plot_mean_curve(ax, cbw_curves, recall_grid, color_c, '--', "CBW",
                         thr_points=cbw_thr_pts, marker='D')

        baseline_info = load_baseline_curve(baseline_run_dir, beh, recall_grid=recall_grid,
                                            threshold=threshold)
        _plot_baseline_on_ax(ax, baseline_info)

        ax.set_title(beh.capitalize(), fontsize=11)
        ax.set_aspect('auto')
        _format_prc_axis(ax, axis_mode=axis_mode)

    fig.suptitle(f"Avg. Precision-Recall ({strat}, {float(pct)*100:.2f}%)", fontsize=11)

    _build_unified_legend(fig, axes, vstack=vstack)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved PRC plot: {save_path}")

    # plt.show()


def plot_avg_prc_across_pcts_for_strat(base_path,
                                       behaviors,
                                       strat,
                                       pcts,
                                       recall_grid=None,
                                       show_std=False,
                                       save_path=None,
                                       threshold=0.5,
                                       baseline_run_dir=None,
                                       axis_mode="tight",
                                       vstack=False):
    """Plot average three panel PRC for a fixed strategy across multiple sampling pcts."""
    recall_grid = recall_grid if recall_grid is not None else np.linspace(0, 1, 200)

    fig, axes = _make_three_panel_figure(vstack=vstack)

    for ax, beh in zip(axes, behaviors):
        for idx, pct in enumerate(pcts):
            pct_dir = os.path.join(base_path, strat, pct)
            run_dirs = list_run_dirs(pct_dir, strat, pct)
            if not run_dirs:
                continue

            color = NATURE_COLORS[idx % len(NATURE_COLORS)]

            mars_curves, cbw_curves = [], []
            mars_thr_pts, cbw_thr_pts = [], []

            for rd in run_dirs:
                mars_cbw = load_run_prc_data(rd, beh)
                if mars_cbw is None:
                    continue
                (gt_m, pm), (gt_c, pc) = mars_cbw

                for gt, p, collector, thr_list in [
                    (gt_m, pm, mars_curves, mars_thr_pts),
                    (gt_c, pc, cbw_curves, cbw_thr_pts)
                ]:
                    if len(np.unique(gt)) < 2:
                        continue
                    prec, rec, _ = precision_recall_curve(gt, p)
                    interp_prec = interpolate_pr_on_fixed_recall(prec, rec, recall_grid)
                    collector.append(interp_prec)
                    thr_p, thr_r = calc_prc_tresh(gt, p, threshold)
                    thr_list.append((thr_p, thr_r))

            pct_label = f"{float(pct)*100:.2f}%"

            _plot_mean_curve(ax, mars_curves, recall_grid, color, '-',
                             f"MARS {pct_label}", thr_points=mars_thr_pts, marker='o')
            _plot_mean_curve(ax, cbw_curves, recall_grid, color, '--',
                             f"CBW {pct_label}", thr_points=cbw_thr_pts, marker='D')

        baseline_info = load_baseline_curve(baseline_run_dir, beh, recall_grid=recall_grid,
                                            threshold=threshold)
        _plot_baseline_on_ax(ax, baseline_info)

        ax.set_title(beh.capitalize(), fontsize=9)
        ax.set_aspect('auto')
        _format_prc_axis(ax, axis_mode=axis_mode)

    fig.suptitle(f"Avg. Precision-Recall ({strat}, varying label fractions)", fontsize=10)

    _build_unified_legend(fig, axes, vstack=vstack)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved to {save_path}")

    # plt.show()


def plot_avg_prc_across_strats_for_pct(base_path,
                                       behaviors,
                                       pct,
                                       strats,
                                       recall_grid=None,
                                       show_std=False,
                                       save_path=None,
                                       threshold=0.5,
                                       baseline_run_dir=None,
                                       axis_mode="tight",
                                       vstack=False):
    """Plot average three panel PRC for fixed label fraction across multiple strategies."""
    recall_grid = recall_grid if recall_grid is not None else np.linspace(0, 1, 200)

    fig, axes = _make_three_panel_figure(vstack=vstack)

    for ax, beh in zip(axes, behaviors):
        for idx, strat in enumerate(strats):
            pct_dir = os.path.join(base_path, strat, pct)
            run_dirs = list_run_dirs(pct_dir, strat, pct)
            if not run_dirs:
                continue

            color = NATURE_COLORS[idx % len(NATURE_COLORS)]

            mars_curves, cbw_curves = [], []
            mars_thr_pts, cbw_thr_pts = [], []

            for rd in run_dirs:
                mars_cbw = load_run_prc_data(rd, beh)
                if mars_cbw is None:
                    continue
                (gt_m, pm), (gt_c, pc) = mars_cbw

                for gt, p, collector, thr_list in [
                    (gt_m, pm, mars_curves, mars_thr_pts),
                    (gt_c, pc, cbw_curves, cbw_thr_pts)
                ]:
                    if len(np.unique(gt)) < 2:
                        continue
                    prec, rec, _ = precision_recall_curve(gt, p)
                    interp_prec = interpolate_pr_on_fixed_recall(prec, rec, recall_grid)
                    collector.append(interp_prec)
                    thr_p, thr_r = calc_prc_tresh(gt, p, threshold)
                    thr_list.append((thr_p, thr_r))

            _plot_mean_curve(ax, mars_curves, recall_grid, color, '-',
                             f"MARS {strat}", thr_points=mars_thr_pts, marker='o')
            _plot_mean_curve(ax, cbw_curves, recall_grid, color, '--',
                             f"CBW {strat}", thr_points=cbw_thr_pts, marker='D')

        baseline_info = load_baseline_curve(baseline_run_dir, beh, recall_grid=recall_grid,
                                            threshold=threshold)
        _plot_baseline_on_ax(ax, baseline_info)

        ax.set_title(beh.capitalize(), fontsize=10)
        ax.set_aspect('auto')
        _format_prc_axis(ax, axis_mode=axis_mode)

    fig.suptitle(f"Avg. Precision-Recall ({float(pct)*100:.2f}%, varying strategies)",
                 fontsize=12)
    _build_unified_legend(fig, axes, vstack=vstack)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved to {save_path}")

    # plt.show()


def plot_f1_vs_pct(base_path,
                   behaviors,
                   strats,
                   pcts,
                   save_path=None,
                   baseline_f1=None,
                   vstack=True):
    def read_metric(df, key):
        try:
            return float(df.loc[key, "0"])
        except Exception:
            return np.nan

    pcts_float = np.arange(len(pcts))
    xtick_labels = [f"{100*float(p):.2f}%" for p in pcts]

    fig, axes = _make_three_panel_figure(vstack=vstack)

    for ax in axes:
        ax.set_aspect("auto")

    n_strats = len(strats)
    bar_width = 0.08
    group_width = (bar_width * 2) * n_strats

    offsets = np.linspace(
        -group_width/2,
        +group_width/2 - bar_width*2,
        n_strats
    )

    for b_idx, beh in enumerate(behaviors):
        ax = axes[b_idx]
        ax.set_title(f"{beh.capitalize()}", fontsize=10)

        if baseline_f1:
            bval = baseline_f1.get(beh, None)
            if bval is not None:
                ax.axhline(bval, color="black", linestyle="--", lw=1.2)
                ax.text(
                    0.99, bval + 0.01,
                    f"{bval:.3f}",
                    ha="right", va="bottom",
                    transform=ax.get_yaxis_transform(),
                    fontsize=9,
                )

        for s_idx, strat in enumerate(strats):
            color = NATURE_COLORS[s_idx % len(NATURE_COLORS)]
            f1_mars, f1_cbw = [], []

            for pct in pcts:
                csv_path = os.path.join(
                    base_path, strat, pct,
                    f"summary_metrics_{beh}.csv"
                )

                if not os.path.exists(csv_path):
                    f1_mars.append(np.nan)
                    f1_cbw.append(np.nan)
                    continue

                df = pd.read_csv(csv_path, index_col=0)
                f1_mars.append(read_metric(df, "f1_mars_mean"))
                f1_cbw.append(read_metric(df, "f1_cbw_mean"))

            f1_mars = np.array(f1_mars)
            f1_cbw = np.array(f1_cbw)

            x_mars = pcts_float + offsets[s_idx]
            x_cbw  = pcts_float + offsets[s_idx] + bar_width

            ax.bar(
                x_mars, f1_mars,
                width=bar_width,
                color=color,
                label=f"{strat} (MARS)" if b_idx == 0 else "_nolegend_"
            )

            ax.bar(
                x_cbw, f1_cbw,
                width=bar_width,
                color=color,
                edgecolor="black",
                hatch="///",
                label=f"{strat} (CBW)" if b_idx == 0 else "_nolegend_"
            )

        ax.set_title(beh.capitalize(), fontsize=11)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_ylim(0.50, 1.0)
        ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.4)
        ax.set_xticks(pcts_float)
        ax.set_xticklabels(xtick_labels)

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    final_handles = []
    final_labels = []

    for h, l in zip(handles, labels):
        if l not in seen and l != "_nolegend_":
            seen.add(l)
            final_handles.append(h)
            final_labels.append(l)

    baseline_handle = Line2D(
        [0], [0],
        color='black',
        linestyle='--',
        lw=1.4,
        label='Baseline (16.67%)'
    )

    fig.legend(
        final_handles + [baseline_handle],
        final_labels + ["Baseline (16.67%)"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=3,
        fontsize=9,
        frameon=True
    )

    fig.suptitle(f"F1 vs Label Fraction", fontsize=12)
    fig.subplots_adjust(bottom=0.18, hspace=0.45)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"Saved F1 plot: {save_path}")


    plt.show()


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

    fig, ax = plt.subplots(figsize=(7, 5))

    bp = df_boxplot.boxplot(
        column="bout_duration",
        by=df_boxplot.index,
        ax=ax,
        vert=True,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        boxprops=dict(color="black")
    )

    for patch in ax.artists:
        patch.set_facecolor(NATURE_COLORS[0])

    # Center x-tick labels
    plt.xticks(rotation=0, ha="center")

    plt.ylabel('Bout Duration (seconds)', fontsize=11)
    plt.xlabel('Behavior', fontsize=11)
    plt.title('Distribution of Bout Durations by Behavior', fontsize=12)
    plt.suptitle('')

    plt.xticks(rotation=0, ha='center')
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}bout_duration_boxplot.pdf")

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

    BASE_PATH = './verify_paper_results/behavior/trained_classifiers/analysis/2x2_hmms'
    BEHAVIORS = ["attack", "investigation", "mount"]

    STRATS = ["natural", "simple_random", "stratified", "systematic", "cluster", "purposive"]
    # STRATS = ["natural", "simple_random", "stratified", "systematic", "cluster", "purposive"]
    PCTS = ['0.01667','0.03334', '0.06668', '0.13336']
    # PCTS = ['0.01667', '0.03334', '0.06668', '1']

    baseline_run_dir = (
        f"{BASE_PATH}/baseline/1/"
        f"{DIR_PREFIX}_baseline_1pct_20251119_102858"
    )

    plot_bout_summary()

    for strat in STRATS:
    #     # aggregate_metrics_across_runs(run_parent_folder=f"{BASE_PATH}/{strat}")
        for pct in PCTS:
    #         # # 1) Exact PRC for a specific run
    #         # plot_pr_curves(
    #         #     base_path=BASE_PATH,
    #         #     strats=[strat],
    #         #     pcts=[pct],
    #         #     behaviors=BEHAVIORS,
    #         #     exact_folder=None,      # or an explicit run dir
    #         #     save_figs=True,
    #         #     baseline_run_dir=baseline_run_dir,
    #         #     axis_mode="zoom",
    #         #     # vstack=True
    #         # )
    #         plot_pr_curves_all_runs(
    #                     base_path=BASE_PATH,
    #                     strat=strat,
    #                     pct=pct,
    #                     behaviors=BEHAVIORS,
    #                     save_figs=True,
    #                     baseline_run_dir=baseline_run_dir,
    #                     axis_mode="zoom")

            # 2) Avg PRC for one strat & one pct
            plot_avg_prc_for_strat(
                base_path=BASE_PATH,
                behaviors=BEHAVIORS,
                strat=strat,
                pct=pct,
                baseline_run_dir=baseline_run_dir,
                axis_mode="zoom",
                save_path=os.path.join(BASE_PATH, strat, pct, f"avg_prc_for_{strat}_{pct}.pdf"),
                vstack=True
            )

    #     # 3) Avg PRC for one strat across pcts
    #     plot_avg_prc_across_pcts_for_strat(
    #         base_path=BASE_PATH,
    #         behaviors=BEHAVIORS,
    #         strat=strat,
    #         pcts=PCTS,
    #         baseline_run_dir=baseline_run_dir,
    #         axis_mode="zoom",
    #         save_path=os.path.join(BASE_PATH, strat, f"avg_prc_for_{strat}.pdf"),
    #         vstack=True
    #     )

    # # 4) Avg PRC for one pct across strats
    # for pct in PCTS:
    #     plot_avg_prc_across_strats_for_pct(
    #         base_path=BASE_PATH,
    #         behaviors=BEHAVIORS,
    #         pct=pct,
    #         strats=STRATS,
    #         baseline_run_dir=baseline_run_dir,
    #         axis_mode="zoom",
    #         save_path=os.path.join(BASE_PATH, f"avg_prc_for_{pct}.pdf"),
    #         vstack=True
    #             )

    plot_f1_vs_pct(
        base_path=BASE_PATH,
        behaviors=BEHAVIORS,
        strats=STRATS,
        pcts=['0.01667', '0.03334', '0.06668', '0.13336'],
        save_path=os.path.join(BASE_PATH, "F1_strats_pcts.pdf"),
        baseline_f1={"attack":0.841, "investigation":0.815, "mount":0.952}
        )
    