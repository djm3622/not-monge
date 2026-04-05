# not-monge

`not-monge` is a research benchmark suite for neural optimal transport. The repository is organized to compare solver families under a shared training budget and answer a narrow empirical question:

> do explicit convexity and c-concavity constraints materially improve OT map recovery?

The codebase supports synthetic continuous OT benchmarks with known ground-truth maps, c-concavity diagnostics, and latent-space OT experiments for diffusion pipelines.

## Scope

The benchmark suite includes three solver groups.

- `learned_w2`: `minimax`, `icnn`, `tw2`, `mmv2`, `mm`, `mm_b`, `qc`
- `reference`: `sinkhorn`, `gaussian`
- `advanced_alt`: `entropic`, `w1`

Every solver resolves through the same registry and benchmark interface:

```python
class OTSolver(Protocol):
    solver_name: str
    solver_group: str
    supports_training: bool
    supports_potential: bool

    def configure_optimizers(self, total_steps: int) -> None: ...
    def training_step(self, batch, scaler, autocast_context, gradient_clip_norm) -> Mapping[str, float]: ...
    def validation_step(self, batch) -> Mapping[str, float]: ...
    def compute_map(self, x: torch.Tensor) -> torch.Tensor: ...
    def compute_potential(self, x: torch.Tensor) -> torch.Tensor | None: ...
    def fit_reference(self, train_loader, val_loader | None = None) -> None: ...
```

## Repository Layout

```text
configs/
  dataset/
  model/
  solver/
  training/
scripts/
  train_baseline.py
  eval_baseline.py
  generate_tables.py
  train_ot.py
  eval.py
  train_diffusion.py
src/
  benchmarking.py
  datasets/
  evaluation/
  models/
  solvers/
  training/
  utils/
tests/
```

Key OT solver modules:

- `src/solvers/base.py`: shared solver protocol and base classes
- `src/solvers/registry.py`: solver registry and factory
- `src/solvers/minimax_ot.py`: unconstrained minimax baseline
- `src/solvers/icnn_baseline.py`: Makkuva-style convex baseline
- `src/solvers/benchmark_baselines.py`: `tw2`, `mmv2`, `mm`, `mm_b`, `qc`
- `src/solvers/reference_baselines.py`: `sinkhorn`, `gaussian`
- `src/solvers/advanced_baselines.py`: `entropic`, `w1`

## Dependencies

The benchmark stack uses:

- `torch`
- `torchvision`
- `numpy`
- `scipy`
- `hydra-core`
- `wandb`
- `timm`
- `diffusers`
- `POT`

Install in editable mode:

```bash
python3 -m pip install -e .
```

Python 3.9+ is assumed.

## Fair Comparison Policy

Learned baselines share the same benchmark budget from `configs/training/base.yaml`.

- same batch size
- same `max_steps`
- same AdamW optimizer family
- same OneCycleLR schedule family
- same seed policy
- same logging and eval cadence

Capacity is also matched by family.

- unconstrained MLP solvers share one width/depth policy
- ICNN-family solvers share one ICNN width/depth policy
- auxiliary networks such as the MM inverse network use the same MLP budget as the transport map

Reference solvers are one-shot fits and are reported separately from optimizer-budget-matched learned methods.

## Config Entry Points

Hydra entry configs:

- `configs/ot.yaml`
- `configs/eval.yaml`
- `configs/diffusion.yaml`

Main OT config groups:

- `configs/solver/minimax.yaml`
- `configs/solver/icnn.yaml`
- `configs/solver/tw2.yaml`
- `configs/solver/mmv2.yaml`
- `configs/solver/mm.yaml`
- `configs/solver/mm_b.yaml`
- `configs/solver/qc.yaml`
- `configs/solver/sinkhorn.yaml`
- `configs/solver/gaussian.yaml`
- `configs/solver/entropic.yaml`
- `configs/solver/w1.yaml`

Common overrides:

- `solver=<solver_id>`
- `dataset=<dataset_id>`
- `training.max_steps=<n>`
- `training.compile=true`
- `training.precision=bf16`
- `training.logging.backend=wandb`
- `experiment.id=ot_recovery|c_concavity|diffusion_latent`

## Training OT Baselines

Unified training entrypoint:

```bash
python3 scripts/train_baseline.py solver=minimax
```

Convex-enforced baseline:

```bash
python3 scripts/train_baseline.py solver=icnn
```

Reference baseline:

```bash
python3 scripts/train_baseline.py solver=sinkhorn
```

Advanced alternative baseline:

```bash
python3 scripts/train_baseline.py solver=w1
```

Small synthetic smoke run:

```bash
python3 scripts/train_baseline.py \
  solver=minimax \
  training.max_epochs=1 \
  training.max_steps=2 \
  dataset.input_dim=2 \
  dataset.n_train=32 \
  dataset.n_val=16 \
  dataset.n_test=16 \
  dataset.batch_size=8 \
  dataset.generator_hidden_dims='[8,8]' \
  model.input_dim=2 \
  model.output_dim=2 \
  model.hidden_dims='[8,8]' \
  solver.potential.hidden_dims='[8,8]'
```

The legacy entrypoint `scripts/train_ot.py` remains as a thin wrapper over `scripts/train_baseline.py`.

## Evaluating Baselines

Unified evaluator:

```bash
python3 scripts/eval_baseline.py \
  solver=minimax \
  evaluation.checkpoint_path=outputs/ot_benchmark/checkpoints/best.pt
```

For c-concavity analysis:

```bash
python3 scripts/eval_baseline.py \
  solver=icnn \
  experiment.id=c_concavity \
  evaluation.checkpoint_path=outputs/ot_benchmark/checkpoints/best.pt
```

The legacy entrypoint `scripts/eval.py` remains as a wrapper over `scripts/eval_baseline.py`.

## Experiments

### 1. OT Recovery

Default experiment id: `ot_recovery`

Supported on all 11 baselines. Metrics include:

- map L2 error
- gradient error when ground truth is available
- pushforward W2 approximation
- MMD

### 2. C-Concavity

Experiment id: `c_concavity`

Intended comparison:

- `minimax`
- `icnn`
- `tw2`
- `mmv2`

Metrics include:

- envelope gap
- convexity violation
- Hessian spectrum summaries

### 3. Diffusion-Latent OT

Experiment id: `diffusion_latent`

Default solver subset:

- `minimax`
- `icnn`
- `tw2`
- `mmv2`
- `entropic`
- `w1`

Metrics include:

- transport metrics in latent space
- FID
- precision
- recall

## Table Generation

Aggregate run outputs into per-experiment CSV and LaTeX tables:

```bash
python3 scripts/generate_tables.py \
  --search-root outputs \
  --table-root artifacts/tables
```

This writes:

- `artifacts/tables/ot_recovery.csv`
- `artifacts/tables/ot_recovery.tex`
- `artifacts/tables/c_concavity.csv`
- `artifacts/tables/c_concavity.tex`
- `artifacts/tables/diffusion_latent.csv`
- `artifacts/tables/diffusion_latent.tex`

Each row stores:

- solver id
- solver group
- dataset id
- experiment id
- seed
- batch size
- max steps
- optimizer
- scheduler
- metric columns

## Diffusion Training

The diffusion stack is still available independently of the OT benchmark suite.

Train a DDPM:

```bash
python3 scripts/train_diffusion.py
```

Offline smoke run with fake data:

```bash
python3 scripts/train_diffusion.py \
  dataset.name=fake_data \
  dataset.image_size=32 \
  dataset.batch_size=4 \
  +dataset.train_size=16 \
  +dataset.val_size=8 \
  +dataset.test_size=8 \
  model.sample_size=32 \
  model.base_channels=32 \
  model.channel_multipliers='[1,2]' \
  model.num_res_blocks=1 \
  model.attention_resolutions='[8]' \
  training.max_epochs=1 \
  training.max_steps=2
```

## Outputs

Training and evaluation runs write structured artifacts into the configured output directory, including:

- `metrics.jsonl`
- `checkpoints/best.pt`
- `checkpoints/epoch_XXXX.pt`
- `results.json`

Reference solvers also serialize deterministic checkpoint artifacts so `eval_baseline.py` can reload them through the same interface as learned solvers.

## Verification

The current workspace has been smoke-tested with:

- `python3 scripts/train_baseline.py` for learned, reference, and advanced baselines
- `python3 scripts/eval_baseline.py` on a saved checkpoint
- `python3 scripts/generate_tables.py`
- `python3 -m pytest -q`

## References

This benchmark suite is aligned with the neural OT comparison literature around semi-dual, convex-enforced, maximin, entropic, and W1 formulations, including:

- Makkuva et al., "Optimal transport mapping via input convex neural networks"
- Korotin et al., "Do Neural Optimal Transport Solvers Work? A Continuous Wasserstein-2 Benchmark"
- Gushchin et al., "Entropic Neural Optimal Transport via Diffusion Processes"
- Chan et al., "Fast and scalable Wasserstein-1 neural optimal transport"
