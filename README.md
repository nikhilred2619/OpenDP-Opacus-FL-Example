# Federated learning with Opacus and OpenDP accounting (Stage 1 example)

A minimal, end-to-end example of cross-silo federated learning in which:

* each client trains locally with **Opacus DP-SGD**: per-example clipping and
  Gaussian noise, with Poisson subsampling at a declared rate driven by the OS
  CSPRNG; and
* each client's privacy loss is tracked by an **OpenDP** fully adaptive
  composition odometer. Every local round is an OpenDP `Measurement` over
  `RenyiDivergence`, and it is executed by submitting it to that odometer.

Status: proposed for review by the OpenDP maintainers. Not reviewed, accepted
or merged by OpenDP. A research demonstration on synthetic data, simulated in one
process; not a production DP deployment.

## Why both libraries

Opacus provides the mechanism for PyTorch models: per-example gradients,
clipping and noise. OpenDP provides the accounting framework: each release is a
measurement with a stated privacy map, and composition is done by OpenDP's
combinators. Opacus's own accountant is not used for any reported number; it
appears only as an independent cross-check.

## What is protected

Record-level DP under add/remove-one-record adjacency, within each client's
non-empty local dataset. Because clients hold disjoint records, the federation's
record-level ε is the maximum over clients. Client-level DP is **not** claimed.
The full statement, assumptions A1–A8, the threat model and the limitations are in
[`PRIVACY_ASSUMPTIONS.md`](PRIVACY_ASSUMPTIONS.md).

## How ε is computed

Per-step RDP of the Poisson-subsampled Gaussian (Mironov, Talwar & Zhang 2019),
× K steps per round. Rounds are composed by the OpenDP odometer, and the result is
converted to (ε, δ) with the Canonne–Kamath–Steinke (2020) bound. ε depends only
on q, σ, K, the number of rounds and δ. Every mechanism parameter, including the
gradient normaliser B, is a public constant, not a function of the data.

## Layout

```
./
├── opendp_opacus_fl.ipynb   # the walkthrough (executed, outputs included)
├── src/
│   ├── accounting.py        # OpenDP measurement + odometer, sampled-Gaussian RDP, RDP->(eps, delta)
│   └── fl.py                # data, disjoint partition, Poisson loader, Opacus round, FedAvg loop
├── tests/test_fl_example.py # 91 tests
├── PRIVACY_ASSUMPTIONS.md   # claim, assumptions A1-A8, threat model, randomness, limitations
├── PRIVACY_PROOF_SKETCH.md  # mechanism, sensitivity, RDP, composition, conversion, open obligations
├── MAINTAINER_REVIEW.md     # map for OpenDP reviewers + questions for the maintainers
├── EXECUTION_REPORT.md      # what was run, where, and the results
├── contribution_summary.md
├── LICENSE, NOTICE.md
├── requirements.txt
└── pytest.ini
```

## Reproduce

```bash
# Run from repository root
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                                   # ~30-45 s on a laptop CPU
jupyter nbconvert --to notebook --execute opendp_opacus_fl.ipynb --output opendp_opacus_fl_executed.ipynb   # ~25 s
```

Verified in a fresh virtual environment: Python 3.13.16, opendp 0.16.0,
opacus 1.6.0, torch 2.14.1 (CPU), numpy 2.5.4, scipy 1.18.1 and dp-accounting
0.6.0. All 91 tests passed, and the notebook executed with no errors and no
warnings in its outputs. Details are in `EXECUTION_REPORT.md`.

Expected notebook numbers: ε = 4.1117 per client at δ = 1e-5. That is also the
record-level ε, since clients are disjoint. OpenDP's sequential bound over all
clients is 6.5127. The budget of ε = 5 refuses round 9. These numbers are
deterministic. Opacus's and Google `dp_accounting`'s RDP analyses agree (4.1117).
For reference, a PLD accountant gives 3.5021 for the same mechanism, so the RDP
route is conservative. Test accuracy is about 0.80 and varies between runs, because the
noise is deliberately not seeded.

## Compatibility with the upcoming OpenDP–Opacus branch

The integration point is `accounting.make_dp_sgd_round`, a user measurement
with a hand-written RDP privacy map. Once a native OpenDP constructor for
Opacus DP-SGD exists, the intended change is:

1. replace `make_dp_sgd_round` and `sampled_gaussian_rdp` with the native constructor;
2. if it reports a different measure (for example PRV or approximate DP), switch
   the odometer's output measure and drop `rdp_to_approx_dp`;
3. keep `ClientPrivacyOdometer`, the all-or-nothing budget check and the FL loop;
4. rerun `tests/` (the RDP reference tests become cross-checks of the native map).

## Randomness, in one paragraph

Poisson sampling uses the OS CSPRNG with an exact threshold, so the true rate is
≤ q. Gaussian noise uses a per-client, per-round `torch.Generator` seeded from
`secrets`, with Opacus `secure_mode=True` (floating-point mitigation). That
generator is **not** a CSPRNG: `torchcsprng` has no build for this
Python/PyTorch. This is a research demonstration with idealised noise; see
`PRIVACY_ASSUMPTIONS.md`.

## Attribution and license

* Sampled-Gaussian RDP: Mironov, Talwar & Zhang (2019). It is implemented
  independently here and cross-checked against
  `opacus.accountants.analysis.rdp.compute_rdp` and numerical integration.
* RDP → (ε, δ): Canonne, Kamath & Steinke (2020); Balle et al. (2020), Thm. 21.
* Floating-point mitigation: Opacus `secure_mode`, after Holohan & Braghin (2021).
* Opacus (Apache-2.0) and OpenDP (MIT) are used as libraries. No code is copied
  from either.
* Design lessons, not code, come from the author's FedCRM-DP V2 research
  implementation
  (https://github.com/nikhilred2619/FedCRM-DP-V2-Reproducible-Research):
  sampling-rate drift, persistent per-client accounting, and record-determined
  partitions.
* License: MIT (`LICENSE`), matching OpenDP; it can be changed to whatever
  the maintainers require. Third-party notices are in `NOTICE.md`.

Author: Nikhil Reddy Donapati.
