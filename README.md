# Semiparametric Efficient Bilevel Gradient Estimation

This repository contains the code to reproduce the experiments of the paper ''**Semiparametric Efficient Bilevel Gradient Estimation**'' in IV and fitted Q evaluation settings. The name for the orthogonal bilevel-gradient estimator is **OBiGrad**.

## Contents

- `IV_figure1_obigrad_gradient_experiment.py` to `IV_figure4_root_estimation_experiment.py`: IV experiments.
- `Bellman_b1_obigrad_experiment.py` to `Bellman_b4_kbo_regularization_experiment.py`: projected Bellman experiments.
- `results/IV/...`: IV outputs grouped by figure.
- `results/Bellman/...`: Bellman outputs grouped by experiment.

## Setup

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing Experiments

Run the scripts from the repository root.

IV Figure 1:

```bash
python IV_figure1_obigrad_gradient_experiment.py \
  --n-grid 200,400,800,1600,3200 \
  --reps 300 \
  --learner sum_fourier \
  --outdir results/IV/figure1
```

IV Figure 2:

```bash
python IV_figure2_obigrad_inference_experiment.py \
  --n-grid 200,400,800,1600,3200 \
  --reps 500 \
  --learner sum_fourier \
  --outdir results/IV/figure2
```

IV Figure 3:

```bash
python IV_figure3_kbo_regularization_experiment.py \
  --n 600 \
  --pop-n 3000 \
  --reps 300 \
  --learner sum_fourier \
  --lambdas 1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1 \
  --outdir results/IV/figure3
```

IV Figure 4:

```bash
python IV_figure4_root_estimation_experiment.py \
  --n-grid 100,200,400,800,1600 \
  --reps 300 \
  --learner linear \
  --pop-n 2500 \
  --outdir results/IV/figure4
```

Bellman B1:

```bash
python Bellman_b1_obigrad_experiment.py \
  --n-grid 200,400,800,1600,3200 \
  --reps 200 \
  --seed 20260425 \
  --learner bellman_basis \
  --outdir results/Bellman/b1
```

Bellman B2:

```bash
python Bellman_b2_inference_experiment.py \
  --n-grid 200,400,800,1600,3200 \
  --reps 200 \
  --seed 20260425 \
  --coord 0 \
  --outdir results/Bellman/b2
```

Bellman B3:

```bash
python Bellman_b3_root_experiment.py \
  --n-grid 200,400,800,1600,3200 \
  --reps 200 \
  --seed 20260425 \
  --outdir results/Bellman/b3
```

Bellman B4:

```bash
python Bellman_b4_kbo_regularization_experiment.py \
  --n 600 \
  --pop-n 12000 \
  --reps 200 \
  --seed 20260425 \
  --obigrad-learner bellman_basis \
  --outdir results/Bellman/b4
```

## Notes

- Each experiment writes manuscript-ready `.tex` tables and `.png` figures only. Tables report Monte Carlo error bars where the quantity is stochastic; deterministic oracle/population bias columns are left without artificial uncertainty.
- Aggregate performance figures use error bars or uncertainty bands. QQ and histogram diagnostics are distributional checks rather than point-estimator curves.
- All scripts use seeded RNGs for repeatable simulations.
- Any `oracle` outputs in the saved results are diagnostic comparisons only, not feasible estimators.

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{elkhoury2026semiparametric,
  title={Semiparametric Efficient Bilevel Gradient Estimation},
  author={El Khoury, Fares and Zenati, Houssam and Kallus, Nathan and Arbel, Michael and Bibaut, Aur{\'e}lien},
  journal={arXiv preprint},
  year={2026}
}
```
