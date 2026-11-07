# Artifact lock: timescale sweeps behind Figs. 3-6

- Git commit: `fe33e61227232cfe71ac75a7e57242b146a165a1`. Working tree at lock time (`git status --porcelain`):
```
M scripts/post_eval_timescale_metrics.py
 M tests/unit/test_post_eval_timescale_metrics.py
 M third_party/NeuralOptimalTransport
 M third_party/OT-ICNN
 M third_party/OptimalTransportModeling
 M third_party/monge_map_solver
?? requirements-lock.txt
?? scripts/create_clean_cosine_scheduler_figure.py
?? scripts/plot_plots1_2_results.py
?? scripts/post_eval_relative_potential_error.py
```
- MongeMap/OTM/MaxCorr were trained at this commit with the plotting edits listed above uncommitted (post-eval scripts only). OTP was trained on 2026-05-05 from the original sweep; the last commit before it started was a5e4c1a, and the config code is unchanged between a5e4c1a and this commit.
- Environment: Python 3.9.6, torch 2.8.0, macOS 27.0.1 arm64, MPS available: True. Exact packages: `requirements-lock.txt`.
- The sweep does not save resolved configs; resolved_config.json was rebuilt with the sweep's own _build_grid_config from each run_spec.json.
- Runs: 500 (4 solvers x K in {1,2,5,10,20} x ratio in {0.05,0.1,0.25,0.5,1} x seeds 0-4). Missing files: 0.
- Full per-file SHA-256 and sizes: `artifacts.lock.json`.

## Runs

| Solver | Runs | last.pt | best.pt | metrics.jsonl | results.json | resolved_config.json | Source folder |
| --- | --- | --- | --- | --- | --- | --- | --- |
| otp | 125 | 125 | 125 | 125 | 125 | 125 | `outputs/timescale_grid_synthetic_harder_mps_5seed_otp` |
| monge_map | 125 | 125 | 125 | 125 | 125 | 125 | `outputs/timescale_grid_synthetic_harder_mps_5seed_fixed_monge_map` |
| otm | 125 | 125 | 125 | 125 | 125 | 125 | `outputs/timescale_grid_synthetic_harder_mps_5seed_fixed_otm` |
| maxcorr | 125 | 125 | 125 | 125 | 125 | 125 | `outputs/timescale_grid_synthetic_harder_mps_5seed_fixed_maxcorr` |

## Outputs

| Folder | Files | Source |
| --- | --- | --- |
| `post_eval/` | 180 | `outputs/timescale_grid_synthetic_harder_mps_5seed_fixed_post_eval` |
| `post_eval_relative_v2/` | 36 | `outputs/timescale_grid_synthetic_harder_mps_5seed_fixed_post_eval_relative_v2` |
| `paper_figures/` | 24 | `outputs/paper_figures_revision_20261107` |
