# Author model sources

Run `python scripts/prepare_sources.py MODEL` to fetch a source version
listed in `../sources.json`. The selected model files are placed here.
Downloaded author sources remain separate from this project's code and
retain their original licensing terms. They are excluded from Git.

The attention implementation and dataset protocols are in
`src/representative_context/`. The pinned upstream versions supply their
task architectures; complete task weights initialize those architectures.
