import os
import sys
import json

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


def save_ann_dicts(annotation_dicts, out_file):
    """Save ann_dict results."""
    # Ensure numpy arrays are converted
    def convert(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj

    serializable = [convert(ann) for ann in annotation_dicts]

    with open(out_file, "w") as f:
        json.dump(serializable, f, indent=2, sort_keys=True)
    print(f"Saved {len(serializable)} ann_dicts to {out_file}")


def compare_ann_files(file1, file2, max_diffs=120):
    """Compare two saved ann_dict JSONs."""
    import numpy as np

    with open(file1) as f:
        run1 = json.load(f)
    with open(file2) as f:
        run2 = json.load(f)

    diffs = []
    for i, (ann1, ann2) in enumerate(zip(run1, run2)):
        fname = ann1.get("dirname", f"file{i}")
        if ann1['nFrames'] != ann2['nFrames']:
            diffs.append(f"[{fname}] nFrames: {ann1['nFrames']} vs {ann2['nFrames']}")

        if ann1['behs'] != ann2['behs']:
            diffs.append(f"[{fname}] Behaviors mismatch")

        # Compare behavior raster
        behs1, behs2 = ann1['behs_frame'], ann2['behs_frame']
        if behs1 != behs2:
            mismatch_count = sum(b1 != b2 for b1, b2 in zip(behs1, behs2))
            diffs.append(f"[{fname}] behs_frame differs in {mismatch_count} frames")

        # Compare bout counts
        for ch in ann1['behs_bout']:
            b1, b2 = ann1['behs_bout'][ch], ann2['behs_bout'].get(ch, {})
            for beh in b1:
                bouts1 = np.array(b1[beh])
                bouts2 = np.array(b2.get(beh, []))
                if bouts1.shape != bouts2.shape or not np.array_equal(bouts1, bouts2):
                    diffs.append(f"[{fname}] Channel {ch}, behavior {beh}: "
                                 f"{len(bouts1)} vs {len(bouts2)} bouts")

        if len(diffs) > max_diffs:
            break

    print("=== Comparison summary ===")
    for d in diffs:
        print(d)
    print(f"Compared {len(run1)} files, {len(diffs)} discrepancies shown (max {max_diffs}).")


def main():
    ROOT_DIR = "./verify_paper_results/behavior/behavior_data/train"
    # ROOT_DIR = "./verify_paper_results/behavior/behavior_data/validation"
    # ROOT_DIR = "./verify_paper_results/behavior/behavior_data/test_1"
    ROOT_DIR = "./verify_paper_results/behavior/behavior_data/"
    FIRST_N_FILES_TO_ANALYZE = 1

    # # Test multiple runs and compare results
    # anns = find_parse_annot_files(ROOT_DIR)
    # save_ann_dicts(anns, "run1.json")

    # anns = find_parse_annot_files(ROOT_DIR)
    # save_ann_dicts(anns, "run2.json")

    # compare_ann_files("run1.json", "run2.json")
    

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


if __name__ == "__main__":
    main()
