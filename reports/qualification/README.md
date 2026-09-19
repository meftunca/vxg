# VXG qualification artifacts

`manifest.json` freezes the eight synthetic Q95 byte streams: `json-records` plus
three held-out JSON/log families, each at 16 and 64 MiB. The corpus files are generated
under ignored `corpus/qualification/`; their exact sizes and SHA-256 values are tracked
here so their bytes can be verified before every run.

Generate or verify them with:

```sh
python3 scripts/generate_qualification_corpus.py
python3 scripts/generate_qualification_corpus.py --verify
```

Run the serial baseline with an explicit new output directory:

```sh
scripts/run_qualification.sh reports/qualification/runs/P0-baseline
```

The runner builds one `vxg-bench` binary, saves its SHA-256 and environment metadata,
then executes all codecs within each process for every Q95 member. Each process keeps a
summary CSV, a CSV containing every timed sample (ranked by duration), and its console
log. Do not overwrite accepted runs. The current runner is a baseline runner; when two
candidate binaries exist, extend it to randomize paired candidate/control process order
and retain every pair before using it for promotion decisions.
