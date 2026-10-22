# Maintainer review guide

A two-minute map of this Stage 1 example for OpenDP reviewers. It is **not** part
of OpenDP, and it has not been reviewed or accepted.

## 1. What it demonstrates

Federated DP-SGD (3 simulated clients, FedAvg) in which the privacy accounting
is done by OpenDP. Opacus does the training. Each local round is an OpenDP
measurement, executed through that client's OpenDP odometer. The guarantee is
record-level (ε, δ)-DP under add/remove adjacency: ε = 4.1117 at δ = 1e-5 for the
notebook configuration.

## 2. OpenDP APIs used (opendp 0.16.0)

| API | Where | Purpose |
|---|---|---|
| `dp.enable_features("contrib", "honest-but-curious")` | `accounting.enable_opendp_features` | required by `make_user_measurement` |
| `dp.user_domain` | `fl.client_domain` | non-empty `ClientDataset` input domain |
| `dp.symmetric_distance()` | `accounting.make_dp_sgd_round` | add/remove adjacency, d_in = 1 |
| `dp.renyi_divergence()` | same | output measure (RDP curve α ↦ ε(α)) |
| `dp.m.make_user_measurement` | same | one local round = one measurement; its function runs Opacus |
| `dp.c.make_fully_adaptive_composition` | `ClientPrivacyOdometer.__init__` | per-client odometer that executes rounds |
| `OdometerQueryable.privacy_loss` | `ClientPrivacyOdometer.odometer_curve` | odometer's own record, compared in tests |
| `dp.c.make_composition` | `ClientPrivacyOdometer.spent_curve` / `prospective_epsilon`, `sequential_bound` | composes RDP curves for reporting, the pre-check, and the sequential bound |

**What OpenDP checks automatically:** domain membership of the input, metric
and measure compatibility in composition, and additive RDP composition.
**What the developer must justify (honest-but-curious):** that each privacy map
is correct for its function. Here that means `sampled_gaussian_rdp` × K matches
what Opacus actually does. The justification is in `PRIVACY_PROOF_SKETCH.md` §2–4
and is supported by tests.

## 3. Why Opacus

It provides per-example gradients (`GradSampleModule`), flat L2 clipping and
Gaussian noise (`DPOptimizer`, `secure_mode=True`) for arbitrary PyTorch models.
`PrivacyEngine` and its accountant are deliberately not used. Poisson sampling
uses a small CSPRNG sampler with the same semantics as Opacus's
`UniformWithReplacementSampler`.

## 4. How accounting is integrated

`make_dp_sgd_round(domain, function, sample_rate, noise_multiplier, steps)`
→ measurement with map d_in = 1 ↦ K·ε_SG(α). `ClientPrivacyOdometer.run_round`
works in three steps:

1. compose the prospective curve with `make_composition` and convert to ε,
   refusing the round if it is over budget or cannot be evaluated;
2. charge the round;
3. submit it to the odometer, which runs the training.

The federation-level ε is the max over clients, because clients are disjoint
(enforced in `make_clients`).

## 5. Implementation vs. mathematical assumptions

* Implemented and tested: public-only parameters, disjointness, exact sampler
  rate ≤ q, fixed clipping, noise scale, step count, fail-closed accounting, and
  release of the model state only.
* Relied on from the literature: MTZ19 (sampled-Gaussian RDP and dominance),
  Mironov 2017 (adaptive composition), Balle et al. 2020 / CKS20 (conversion).
* Assumed: ideal Gaussian noise (the noise RNG is not a CSPRNG), real
  arithmetic, client isolation (one process here), and Opacus correctness.

## 6. Independently executed

In a fresh virtual environment (see `EXECUTION_REPORT.md`): `pip install -r
requirements.txt`, then `pytest` (91 tests, all passed), then the full notebook
executed with `nbconvert`. A mutation check (7 deliberate bugs, each caught by at
least one test) is described in the report.

## 7. Intentionally outside Stage 1

Flower or any network transport, secure aggregation, client-level DP, client
subsampling, real datasets, PRV/PLD accounting, hyperparameter tuning, and
production RNG.

## 8. Questions for Eddie

1. **Native mechanism:** what constructor and measure will the OpenDP–Opacus
   branch expose (RDP, PRV, approximate DP)? It replaces `make_dp_sgd_round` and
   `sampled_gaussian_rdp`.
2. **Parallel composition:** is there, or should there be, an OpenDP-native way
   to express "disjoint partitions → max", instead of the max taken in Python?
3. **Privacy filter on RDP:** `make_privacy_filter(odometer, d_in=1, d_out=curve)`
   over a `renyi_divergence` odometer aborts the process (Rust panic
   `unrecognized generic "Function"`). Is it unsupported, or a bug? A reproducer
   is in `contribution_summary.md`.
4. **Odometer and broken maps:** the odometer runs a measurement whose privacy map
   raises, and fails only at `privacy_loss`. Is that intended? It is handled
   here by a pre-check.
5. **Sampler rate:** `torch.rand < q` includes with probability ⌈q·2²⁴⌉/2²⁴ ≥ q.
   Is this relevant to the sampler fixes you mentioned?
6. **Repository shape:** dependencies and layout for `fl/`. Should tests go in
   the repo's CI?

## 9. Adapting to `fl/` and the OpenDP–Opacus branch

* Copy `fl/` as is. The dependency surface is `requirements.txt`; `scipy` and
  `dp-accounting` are only needed for tests and the cross-check cell.
* When the branch lands, replace the body of `make_dp_sgd_round` with the native
  constructor. If its measure is not RDP, change the odometer's output measure
  and drop `rdp_to_approx_dp`. Then rerun `tests/`: the RDP reference tests
  become cross-checks of the native map.
* `CSPRNGPoissonSampler` can be dropped if the branch provides a sampler with a
  stated, exact inclusion probability.
