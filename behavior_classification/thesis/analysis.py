import sys
import os

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from behavior_classification.thesis.parsers import find_parse_annot_files


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


def plot_mean_bout_duration(df, skip_behaviors=None):
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
    plt.show()


def plot_bout_duration_histogram(df, skip_behaviors=None):
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
    plt.show()


def plot_bout_duration_box(df, skip_behaviors=None):
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
    plt.show()


def calculate_bout_properties(bout_data, total_frames, framerate: float = 30.0) -> dict:
    """Calculates bout properties."""
    if bout_data.size == 0:
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

    bout_lengths = bout_data[:, 1] - bout_data[:, 0] + 1
    bout_durations = bout_lengths / framerate

    num_bouts = len(bout_lengths)
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


def print_bout_summary(bout_properties, total_frames, skip_behaviors=None):
    """Prints bout properties, skipping specified behaviors."""
    for behavior, properties in bout_properties.items():
        if skip_behaviors and behavior in skip_behaviors:
            continue
        print(f"--- Behavior: {behavior} ---")
        print(f"  Number of bouts: {properties['num_bouts']}")
        print(f"  Mean bout duration: {properties['mean_bout_duration']:.2f} seconds")
        print(f"  Median bout duration: {properties['median_bout_duration']:.2f} seconds")
        print(f"  Std. dev. bout duration: {properties['std_bout_duration']:.2f} seconds")
        print(f"  Min bout duration: {properties['min_bout_duration']:.2f} seconds")
        print(f"  Max bout duration: {properties['max_bout_duration']:.2f} seconds")
        print("\n")
        print(f"  Mean bout length: {properties['mean_bout_length']:.2f} frames")
        print(f"  Median bout length: {properties['median_bout_length']:.2f} frames")
        print(f"  Std. dev. bout length: {properties['std_bout_length']:.2f} frames")
        print(f"  Min bout length: {properties['min_bout_length']:.2f} frames")
        print(f"  Max bout length: {properties['max_bout_length']:.2f} frames")
        print(f"  Proportion of frames: {properties['proportion_frames']:.2%}")
        print("\n")


if __name__ == "__main__":
    ROOT_DIR = "./verify_paper_results/behavior/behavior_data/train"
    # ROOT_DIR = "./verify_paper_results/behavior/behavior_data/validation"
    # ROOT_DIR = "./verify_paper_results/behavior/behavior_data/test_1"
    SKIP_BEHAVIORS = ['other']

    annotations = find_parse_annot_files(ROOT_DIR)

    # Extract all different behavior names
    all_behavior_names = set()
    for ann_dict in annotations:
        if 'behs_bout' in ann_dict and 'Ch1' in ann_dict['behs_bout']:
            all_behavior_names.update(ann_dict['behs_bout']['Ch1'].keys())

    # Remove specified behaviors
    if SKIP_BEHAVIORS:
        all_behavior_names = all_behavior_names.difference(SKIP_BEHAVIORS)
    all_behavior_names = sorted(list(all_behavior_names))

    # Adjust 'other' annotations and calculate total frames
    # (as 'other' is default annotation from start to end frame for eaach video)
    total_frames = 0
    for ann_dict in annotations:
        if 'behs_bout' in ann_dict and 'Ch1' in ann_dict['behs_bout'] and 'other' in ann_dict['behs_bout']['Ch1']:
            other_bouts = ann_dict['behs_bout']['Ch1']['other']
            n_frames = ann_dict['nFrames']
            total_frames += n_frames

            remaining_behavior = np.empty((0, 2), int)
            for beh in all_behavior_names:
                if beh == 'other': continue  # Skip 'other' behavior
                # Collect all remaining behaviors
                if 'behs_bout' in ann_dict and 'Ch1' in ann_dict['behs_bout'] and beh in ann_dict['behs_bout']['Ch1']:
                    remaining_behavior = np.vstack((remaining_behavior, ann_dict['behs_bout']['Ch1'][beh]))

            # Mark annotated frames (excluding 'other')
            annotated_frames = np.zeros(n_frames, dtype=bool)
            if remaining_behavior.size > 0:
                for start, end in remaining_behavior:
                    annotated_frames[(start-1):end] = True

            # Calculate start/end frames for 'other'
            start_frames = np.where((annotated_frames[:-1] == True) & (annotated_frames[1:] == False))[0] + 1
            end_frames = np.where((annotated_frames[:-1] == False) & (annotated_frames[1:] == True))[0] + 1

            # Edge cases:
            if not annotated_frames[0]:
                start_frames = np.insert(start_frames, 0, 1)
            if not annotated_frames[-1]:
                end_frames = np.append(end_frames, n_frames)

            # Update 'other' annotations with calculated intervals
            new_other = np.column_stack((start_frames, end_frames))
            ann_dict['behs_bout']['Ch1']['other'] = new_other

    # Collect all bout data
    bout_data_by_behavior = {}
    for behavior in all_behavior_names:
        bout_data_by_behavior[behavior] = np.empty((0, 2), int)
        for ann_dict in annotations:
            if 'behs_bout' in ann_dict and 'Ch1' in ann_dict['behs_bout'] and behavior in ann_dict['behs_bout']['Ch1']:
                bout_data = ann_dict['behs_bout']['Ch1'][behavior]
                bout_data_by_behavior[behavior] = np.vstack((bout_data_by_behavior[behavior],
                                                             bout_data))

    bout_properties = {}
    for behavior in all_behavior_names:
        bout_properties[behavior] = calculate_bout_properties(bout_data_by_behavior[behavior],
                                                              total_frames)
    print_bout_summary(bout_properties, total_frames, skip_behaviors=SKIP_BEHAVIORS)

    df = pd.DataFrame.from_dict(bout_properties, orient='index')
    plot_mean_bout_duration(df, skip_behaviors=SKIP_BEHAVIORS)
    plot_bout_duration_histogram(df, skip_behaviors=SKIP_BEHAVIORS)
    plot_bout_duration_box(df, skip_behaviors=SKIP_BEHAVIORS)
