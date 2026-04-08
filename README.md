# not-monge

Case study 1 compares the final paper solvers on the cached `paper_mix3to10` benchmark and writes the table-ready summary plus per-seed figures.

## Setup

```bash
python3 -m pip install -e .
```

## Run The Full Case Study 1 Suite

```bash
python3 scripts/paper_case1_suite.py \
  --device mps \
  --output-root outputs/paper_case1_suite \
  --rerun
```

This runs the paper table solvers `gaussian,mm,mmv2,tw2,mm_b,qc`, keeps only `best.pt` for each seed, and saves:

- per-seed `results.json`
- per-seed `visualizations/transport_geometry.{png,pdf}`
- per-seed `visualizations/saddle_geometry.{png,pdf}`
- per-seed sampled saddle plots under `visualizations/saddle_samples/`
- top-level `case1_results_table.tex`, `seed_summary.json`, and `seed_summary.md`

## Match The Tuned Runs

```bash
python3 scripts/paper_case1_suite.py \
  --device mps \
  --solvers mm,mmv2 \
  --output-root outputs/paper_case1_suite_mm_mmv2_tuned_v1 \
  --rerun
```

```bash
python3 scripts/paper_case1_suite.py \
  --device mps \
  --solvers tw2,mm_b,qc \
  --output-root outputs/paper_case1_suite_tw2_mmb_qc_v1 \
  --rerun
```

```bash
python3 scripts/paper_case1_suite.py \
  --device mps \
  --solvers qc \
  --output-root outputs/paper_case1_suite_tw2_mmb_qc_v1 \
  --rerun
```

`otp` is still supported by the suite config, but it is no longer part of the default case-study-1 paper table.
