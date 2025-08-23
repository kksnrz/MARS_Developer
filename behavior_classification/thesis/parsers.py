import os
import sys

import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import annotation_parsers as ap


def find_parse_annot_files(root_dir):
    """
    Recursively searches for .annot files in a dir/subdirs.
    Args:
        root_dir: root dir to start search.
    Returns:
        List of annotation dicts, for each .annot file found.
    """

    print("INFO: Start parsing annotation files")
    annotation_dicts = []

    if not os.path.exists(root_dir):
        print(f"ERROR: Root directory '{root_dir}' does not exist.")
        return annotation_dicts

    if not os.path.isdir(root_dir):
        print(f"ERROR: '{root_dir}' is not a directory.")
        return annotation_dicts

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".annot"):
                filepath = os.path.join(dirpath, filename)
                try:
                    ann_dict = ap.parse_annotations(filepath)
                    ann_dict['dirname'] = os.path.basename(dirpath)
                    annotation_dicts.append(ann_dict)
                except Exception as e:
                    print(f"ERROR parsing {filepath}: {e}")
    print("INFO: Finished parsing annotation files")
    return annotation_dicts


if __name__ == "__main__":
    # ROOT_DIR = "./verify_paper_results/behavior/behavior_data/train"
    # ROOT_DIR = "./verify_paper_results/behavior/behavior_data/validation"
    ROOT_DIR = "./verify_paper_results/behavior/behavior_data/test_1"
    FIRST_N_FILES_TO_ANALYZE = 1

    annotations = find_parse_annot_files(ROOT_DIR)

    print()
    print(f"Found {len(annotations)} annotation files.")
    print()
    for i in range(FIRST_N_FILES_TO_ANALYZE):
        print(f"File {i+1}: {annotations[i]['dirname']}")
        print(f"Frames: {annotations[i]['nFrames']}")
        for behavior in annotations[i]['behs_bout']['Ch1']:
            print(f"Behavior: {behavior}")

            print(f"Start, end frames \n{np.matrix(annotations[i]['behs_bout']['Ch1'][behavior])}")
            print()
