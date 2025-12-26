The steps below describe the minimum setup required to run training/testing experiments and reproduce thesis analysis outputs.

1. Follow the official MARS `README.md` to setup the conda environment and installed dependencies.
    1. If cloned from repo, all relevant changes are in branch 'verify_paper_results'
2. Probably patch the external `Util` import (case-sensitive fix needed at least for linux)
	1. The external  `Util` package contains an import that must be updated to match the folder name
	2. Open: `Util/genericVideo.py`
	3. Change the import statement:
		1. From: `from util.seqIo import *`
		2. To: `from Util.seqIo import *`
	4. This is needed as the package is managed externally and the import could not be corrected 
3. Prepare the Dataset and run the `MARS_behavior_tutorial.ipynb` 
	1. The provided notebook explains all steps to run preprocessing/training/testing
4. Diagrams were created by running the `analysis.py`  file