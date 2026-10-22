# Contribution summary

## What this is

A small, runnable example of federated DP-SGD whose privacy accounting is done
by OpenDP, not by Opacus. It is meant as a first cut for the planned `fl/`
directory: it runs in under a minute on a CPU, and it is easy to move onto the
upcoming OpenDP–Opacus integration.

## Why it may be useful to OpenDP

1. **An end-to-end FL scenario tracked by the OpenDP framework.** Each local
   round is an OpenDP measurement, and a client's rounds run through that
   client's OpenDP odometer.
2. **It exercises `renyi_divergence` with user measurements and fully adaptive
   composition** in a realistic adaptive setting, where the global model depends
   on earlier releases.
3. **It makes the gaps concrete.** These are the pieces it has to supply under
   `honest-but-curious`:

| Gap in OpenDP 0.16.0 | Supplied here |
|---|---|
| RDP of the Poisson-subsampled Gaussian | `accounting.sampled_gaussian_rdp`, checked against Opacus and quadrature |
| `RenyiDivergence` → approximate-DP cast | `accounting.rdp_to_approx_dp` |
| Privacy filter over an RDP odometer | Python pre-check using `make_composition` |
| Parallel composition over disjoint partitions | max over clients, with disjointness enforced in code |
| Opacus DP-SGD as a native mechanism | Opacus called inside `make_user_measurement` |

4. **It avoids two easy mistakes in Opacus-based accounting**, each guarded by a
   regression test:
   * sampling-rate drift: q and K are declared once and feed both the sampler
     and the privacy map, and the step count is asserted;
   * data-dependent normalisation: B is a public constant, not round(q · n_k).
     An earlier draft of this example had this bug.

## Validation performed

| Check | Result |
|---|---|
| Test suite (fresh venv) | 91 passed, 0 failed, 0 skipped |
| Notebook (fresh venv) | executed, 0 errors, no warnings in the outputs |
| Sampled-Gaussian RDP vs Opacus `compute_rdp` | agrees to rel. 1e-9 (4 configurations, α = 2…256) |
| ε vs Google `dp_accounting` 0.6.0 | RDP (same orders) agrees to rel. 1e-6; PLD gives 3.5021 < 4.1117 |
| Sampled-Gaussian RDP vs quadrature of D_α, both directions | forward agrees to rel. 1e-6, reverse dominated (12 configurations) |
| Poisson sampler (Monte Carlo, 4000 steps) | inclusion frequency ≈ q; batch-size variance ≈ nq(1−q) |
| Noise (Monte Carlo, 20,000 coordinates) | std = σC/B within 3%, with and without `secure_mode` |
| Mutation check | 7 deliberate bugs, each caught by at least one test (see EXECUTION_REPORT.md) |

## What still needs mathematical or maintainer review

* The parallel-composition step (federation ε = max_k ε_k) under adaptive,
  interleaved rounds is argued from the literature and enforced structurally,
  but it is not proved here or expressed in OpenDP. Maintainer input wanted:
  is there an OpenDP-native way to express it, for example a partition
  combinator?
* MTZ19's dominance of the remove direction is relied on and checked
  numerically at a few points, not re-proved.
* Noise randomness is idealised (A5): PyTorch's Mersenne Twister with secret
  seeds and Opacus `secure_mode`; it is not a CSPRNG (`torchcsprng` is
  unavailable for Python 3.13 / PyTorch 2.14). Sampling uses the OS CSPRNG.

## Observations worth confirming with the maintainers

**(a) Odometer runs a measurement whose privacy map raises.** In opendp 0.16.0,
`make_fully_adaptive_composition(...)(data)(meas)` executes `meas`'s function and
returns its output even when the privacy map raises. The error appears only when
`privacy_loss(1)` is called. A function that raises is not
charged. This example therefore pre-evaluates the map and charges attempted
rounds (`ClientPrivacyOdometer`). A characterisation test is in
`test_characterise_raw_odometer_runs_function_despite_broken_map`.

**(b) Sampler rate.** `torch.rand(n) < q` includes each record with probability
⌈q·2²⁴⌉/2²⁴, which can exceed q (0.0500000119 for q = 0.05). That is negligible
here, but an accountant using q is then not an upper bound. The default sampler
in this example uses the OS CSPRNG with an exact floor threshold.

**(c) Privacy filter over RDP:**

With opendp 0.16.0, constructing a privacy filter over a `renyi_divergence`
odometer aborts the Python process with exit code 134. It is a Rust panic, not
a Python exception:

```python
import opendp.prelude as dp
dp.enable_features("contrib", "honest-but-curious")
dom = dp.user_domain("D", lambda x: True)
m = dp.m.make_user_measurement(dom, dp.symmetric_distance(), dp.renyi_divergence(),
                               function=lambda x: 0, privacy_map=lambda d: (lambda a: 0.1 * a))
od = dp.c.make_fully_adaptive_composition(dom, dp.symmetric_distance(), dp.renyi_divergence())
budget = dp.c.make_composition([m, m]).map(1)          # an OpenDP Function
dp.c.make_privacy_filter(od, d_in=1, d_out=budget)     # panics: unrecognized generic "Function"
```

This may be an unsupported combination rather than a bug. Either way, a Python
exception would be friendlier than a process abort.

## Proposed next steps

* Swap in the OpenDP–Opacus branch's native constructor when it is available, and
  rerun the tests (see README, "Compatibility").
* Optionally, add a PRV/PLD curve once OpenDP supports it, since RDP is loose.
* Optionally, run the same loop through Flower once the `fl/` repository's
  shape and dependencies are decided.

## Not claimed

No acceptance by OpenDP, no production readiness, no cryptographic randomness,
no client-level DP, and no formal proof. The claim is limited to record-level DP
under the assumptions in `PRIVACY_ASSUMPTIONS.md`.
