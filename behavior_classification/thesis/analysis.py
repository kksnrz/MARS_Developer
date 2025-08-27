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
                                ax: plt.Axes=None,
                                save_path=None) -> plt.Axes:
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

    if save_path:
        plt.savefig(f"{save_path}precision_recall_curve.png")
    return ax


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
            with open(f"{save_path}behavior_transitions.txt", 'w', encoding='utf-8') as f:
                f.write(output_string)
            print(f"Behavior transition saved to {save_path}behavior_transitions.txt")
        except Exception as e:
            print(f"Error saving behavior transitions to file: {e}")


def main():
    ROOT_DIR = "./verify_paper_results/behavior/behavior_data/"
    # ROOT_DIR = "./verify_paper_results/behavior/behavior_data/train"
    # ROOT_DIR = "./verify_paper_results/behavior/behavior_data/validation"
    # ROOT_DIR = "./verify_paper_results/behavior/behavior_data/test_1"
    PLOT_SAVE_DIR = "./verify_paper_results/behavior/trained_classifiers/plots/full_"
    # PLOT_SAVE_DIR = None
    SKIP_BEHAVIORS = ['other']

    # 1. Load annotation files
    annotations = find_parse_annot_files(ROOT_DIR)

    # 2. Collect all unique behavior names across all files
    all_behavior_names = set()
    for ann_dict in annotations:
        if 'behs_frame' in ann_dict:
            all_behavior_names.update(ann_dict['behs_frame'])

    # 3. Apply skip filter (always, even if empty)
    all_behavior_names = sorted(list(all_behavior_names.difference(SKIP_BEHAVIORS)))
    print("All behaviors considered for analysis:", all_behavior_names)

    # 4. Initialize bout data containers
    bout_data_by_behavior = {behavior: [] for behavior in all_behavior_names}


    # ----------------------------------------------------------------------------------------------
    # TODO: Code below is messy - REFACTOR
    # TODO: 'Other' is not calc. correctly, as there are no annotations for it
        # So it can be skipped or needs to be build manually
    # TODO: Transition counts is not tested right now

    # 5. Extract bouts from each annotation file
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
            # Collect and count total bouts by behavior
            if behavior in ann_dict['behs_bout']['Ch1']:
                bouts = ann_dict['behs_bout']['Ch1'][behavior]

                differences = np.diff(bouts, axis=1)
                transformed_bouts = np.concatenate((bouts, differences), axis=1)

                all_behavior_data[behavior]['collected_bouts'] = \
                    np.concatenate((all_behavior_data[behavior]['collected_bouts'],
                                    transformed_bouts))
                all_behavior_data[behavior]['bout_count'] += bouts.shape[0]
                all_behavior_data[behavior]['total_frames'] += n_frames

    bout_properties = {}
    for behavior in all_behavior_names:
        bout_properties[behavior] = calculate_bout_properties(
            all_behavior_data[behavior], n_frames
        )

    # 8. Print bout summary
    print_bout_summary(
        bout_properties, n_frames,
        skip_behaviors=SKIP_BEHAVIORS,
        save_path=PLOT_SAVE_DIR
    )

    # 9. Analyze transitions (now also skip behaviors explicitly)
    transitions = analyze_behavior_transitions(annotations)
    print_behavior_transitions(transitions, save_path=PLOT_SAVE_DIR)

    # 10. Optional plotting
    df = pd.DataFrame.from_dict(bout_properties, orient='index')
    # plot_mean_bout_duration(df, skip_behaviors=SKIP_BEHAVIORS, save_path=PLOT_SAVE_DIR)
    # plot_median_bout_duration(df, skip_behaviors=SKIP_BEHAVIORS, save_path=PLOT_SAVE_DIR)
    # plot_bout_duration_histogram(df, skip_behaviors=SKIP_BEHAVIORS, save_path=PLOT_SAVE_DIR)
    # plot_bout_duration_box(df, skip_behaviors=SKIP_BEHAVIORS, save_path=PLOT_SAVE_DIR)


if __name__ == "__main__":
    main()
