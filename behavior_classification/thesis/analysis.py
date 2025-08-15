import dill
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import numpy as np


def read_dill_file(file_path):
    with open(file_path, 'rb') as f:
        return dill.load(f)


def plot_precision_recall_curve(ground_truth: np.ndarray,
                                probabilities: np.ndarray,
                                classifier_name: str,
                                ax: plt.Axes=None) -> plt.Axes:
    """
    Plots precision-recall curve for given classifier.

    Args:
        ground_truth: Ground truth labels (0 or 1).
        probabilities: Predicted probabilities for positive class.
        classifier_name: Name of classifier for plot title.
        ax: Axes object to plot on. If None, new figure and axes are created. Defaults to None.
    Returns:
      ax: Axes object containing plot.
    """

    precision, recall, _ = precision_recall_curve(ground_truth, probabilities)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(recall, precision, marker='.')
    ax.set_xlabel('Recall vs human 1')
    ax.set_ylabel('Precision vs human 1')
    ax.set_title(f'Precision-Recall Curve - {classifier_name}')
    ax.grid(True)
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0.2, 1.0)
    return ax


if __name__ == "__main__":

    P_mat = read_dill_file("verify_paper_results/behavior/trained_classifiers/verify_paper_results_xgb_es50_depth3_child1_wnd/results.dill")

    vocabulary = {"other": 0, "mount": 1, "attack": 2, "investigation": 3} # extracted from behavior/behavior_jsons/test_data.json
    class_names = list(vocabulary.keys())[1:]  # Exclude 'other'
    y_gt = P_mat['0_Gc'] # dict of classifiers with ground truth labels (0 or 1)
    proba = P_mat['4_proba_pd_hmm_fbs'] # Predicted probabilities for each class shape (#frames, 4, 2)

    fig, axes = plt.subplots(1, len(class_names),
                             figsize=(8 * len(class_names), 6))
    if len(class_names) == 1:
        axes = [axes]
    for i, class_name in enumerate(class_names):
        gt_class = y_gt[class_name]
        proba_class = proba[:, i+1, 1]  # +1 to skip 'other' # , 1 to just take pos. class. proba.
        axis = plot_precision_recall_curve(gt_class, proba_class, class_name,
                                        axes[i])
    plt.tight_layout()
    plt.show()
