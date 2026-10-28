# Exploratory federated-learning utility benchmark

This is an independent benchmark extension suggested as an area of exploration
by an OpenDP developer, **not an accepted or formally assigned Stage 2 task**.
The original Stage 1 project remains untouched on the main branch.

## Technical question

Compare ordinary PyTorch federated learning with Stage 1's Opacus DP-SGD and
OpenDP privacy-accounting integration under matched training conditions.

- **Public dataset:** scikit-learn handwritten digits, 8×8 features, 1,797
  observations. Binary label: digits 0–4 vs 5–9. This is not full MNIST/FEMNIST.
- **Data split:** stratified public 80/20 train/test; three clients; IID versus
  Dirichlet label-skew (alpha 0.3) with seed-specific, fixed public partitions.
- **Matched:** identical 64→16→1 MLP initialization, FedAvg weights, nominal
  Poisson sample rate 0.10, learning rate 0.4, public gradient divisor B=40,
  training rounds and local steps. Random batch draws are independent.
- **DP arm:** Opacus clipping C=1, noise multiplier 1.2, Stage 1's OpenDP
  per-client RDP odometers and conditional epsilon estimate.
- **Primary comparison:** after each DP run reaches its final accuracy, find
  the first round at which non-DP training reached that *same* accuracy. Explicitly
  report "not reached" if necessary.
- **Additional metrics:** per-round accuracy/loss, client label distributions,
  per-client epsilon maxima, wall-clock training time, and seed variability.

## Reproduce locally

From the repository root, in a Python 3.13 environment:

    python -m pip install -r requirements.txt "scikit-learn>=1.5"
    python -m pytest -q tests/test_dp_utility_gap.py
    python -m benchmarks.dp_utility_gap --out benchmark-output

Quick smoke experiment:

    python -m benchmarks.dp_utility_gap --scenarios iid --seeds 11 --rounds 1 --local-steps 1 --out benchmark-smoke

Results are created by actual execution, never hardcoded: results.json contains
per-round histories, results.md has summary tables, and rounds.csv is suitable
for plotting. GitHub Actions uploads those files with logs and dependency versions.

## Boundaries and limitations

This public-data benchmark uses label-based splits only on public labels. A real
federated system cannot derive client partitioning from private data and assume
the same record-level privacy claim without further analysis. The Stage 1
Gaussian RNG is not cryptographically secure, and the custom RDP privacy map
still requires independent mathematical review. Tests do not constitute a proof.

DP noise is intentionally unseeded, so exact accuracies vary even with seeded
dataset and model initialization. Our baseline uses the same *sampling protocol*,
not exactly identical minibatches. The DP final test accuracy is a post-hoc
exploratory target, not a preregistered confirmatory target. Runtime includes
both training and privacy accounting and is not an isolated RDP speed benchmark.

This is a small pilot, not a claim of parity with published MNIST, LEAF, or
FEMNIST results. Any wider dataset study can follow maintainer feedback.
