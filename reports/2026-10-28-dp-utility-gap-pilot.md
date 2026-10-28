# Exploratory DP versus non-DP federated-learning pilot

**Date:** 2026-10-28
**Status:** Contributor research experiment; **not** reviewed, accepted, or endorsed by OpenDP.
**GitHub Actions:** https://github.com/nikhilred2619/OpenDP-Opacus-FL-Example/actions/runs/40658105534
**Measured-results artifact:** https://github.com/nikhilred2619/OpenDP-Opacus-FL-Example/actions/runs/40658105534/artifacts/12802327246

## Question and protocol

For the same client data split and initial model, compare normal PyTorch
federated learning and the Stage 1 Opacus DP-SGD implementation, asking:
**At which federated round does the baseline first attain the DP arm's
final accuracy?** A missing value means it did not reach that target within
the six rounds observed; no unobserved training time is extrapolated.

- Public scikit-learn handwritten digits (8 x 8 pixels; 1,797 samples);
  binary 0–4 versus 5–9 target. Not equivalent to full MNIST or FEMNIST.
- Public 80/20 stratified train/test split for each seed (11, 23, 37).
- Three clients; public IID random split or public-label Dirichlet split (alpha=0.3).
- Identical 64→16→1 model initialization within each paired run, FedAvg with
  uniform public weights, six rounds and three local steps per client/round.
- Same nominal Poisson sampling rate q=0.1, fixed public normalizer B=40,
  learning rate 0.4, and held-out public evaluation data.
- Baseline: plain PyTorch with no clipping/noise. DP: Opacus clip C=1,
  noise multiplier sigma=1.2, OpenDP custom per-client RDP odometers,
  delta=1e-5. Batches are independently drawn (not identical between arms).
- All six new benchmark unit tests passed, the one-round smoke run passed,
  the full two-scenario/six-pair experiment completed, and raw logs are
  saved in the linked artifact.

## Results — medians across three seeds

| Split | Non-DP final accuracy | DP final accuracy | Estimated max-client epsilon | Median non-DP training seconds | Median DP training seconds | Baseline reached DP final target |
|---|---:|---:|---:|---:|---:|---:|
| IID | 0.7861 | 0.7139 | 2.8892 | 0.05 | 3.29 | 2/3 |
| Label-Dirichlet | 0.7778 | 0.5917 | 2.8892 | 0.06 | 3.31 | 2/3 |

## Matched per-seed target comparisons

| Split | Seed | Non-DP final accuracy | DP final accuracy and target | First non-DP round attaining target | First DP round attaining target |
|---|---:|---:|---:|---:|---:|
| IID | 11 | 0.8278 | 0.7139 | 3 | 6 |
| IID | 23 | 0.7611 | 0.5500 | 1 | 1 |
| IID | 37 | 0.7861 | 0.8167 | Not reached | 5 |
| Label-Dirichlet | 11 | 0.7778 | 0.6194 | 2 | 1 |
| Label-Dirichlet | 23 | 0.7861 | 0.5917 | 1 | 6 |
| Label-Dirichlet | 37 | 0.5111 | 0.5639 | Not reached | 6 |

Non-IID client sizes were uneven, including 52/1257/128 examples for seed
11. That is part of the benchmark configuration, not an estimate of realistic
deployment sizes. Per-client label fractions are saved in the raw results.

## Interpretation and limitations

These results demonstrate the **method** for an apples-to-apples *protocol*
comparison, not a universal performance claim: one IID seed scored higher for
the DP model than its non-DP counterpart, and the sample contains only three
seeds per scenario. A six-round horizon is too short to conclude that a model
will never reach a target; "not reached" means only within the measured window.

The target is defined post-hoc by DP final accuracy on the public test set,
so results are exploratory. A confirmatory experiment should predeclare an
accuracy threshold and use a separate validation/test plan.

Time includes PyTorch, Opacus and OpenDP bookkeeping, and is not an isolated
measurement of OpenDP RDP-accountant speed.

The Stage 1 DP noise uses PyTorch's non-cryptographic RNG, with an ideal-noise
assumption. Its custom RDP map, budget policy and other assumptions require
independent mathematical review. CI cannot certify the privacy guarantee.
This pilot does not establish secure aggregation or client-level DP.

The label-Dirichlet split uses **public benchmark labels**. Applying
label-dependent client discovery to sensitive data could change the privacy
problem and requires additional analysis.

## Reproducibility

Run from the repository root on this branch:

    python -m pip install -r requirements.txt "scikit-learn>=1.5"
    python -m pytest -q tests/test_dp_utility_gap.py
    python -m benchmarks.dp_utility_gap --scenarios iid,label_dirichlet --seeds 11,23,37 --rounds 6 --local-steps 3 --out benchmark-output

The artifact contains results.json, results.md, rounds.csv, full CI logs and
the dependency snapshot. Model initialization and public-data splitting are
seeded; **DP Gaussian noise is intentionally not seeded**, so reruns can vary.

**Next step:** Request OpenDP maintainer feedback on the usefulness of this
pilot, benchmark dataset selection, target-performance comparison, and any
future integration. No official acceptance is implied.
