# Execution report

Everything below was run on 2026-10-22, in a **fresh virtual environment** created
from this directory. The numbers are copied from the run logs, not edited.

## Environment

| Item | Value |
|---|---|
| OS | Linux 6.18.44 x86_64 (Ubuntu 24.04.5 LTS base), glibc 2.39 |
| Hardware | 2 CPU cores, no GPU (CUDA not available; PyTorch ran on CPU) |
| Python | 3.13.16 |
| opendp | 0.16.0 |
| opacus | 1.6.0 |
| torch | 2.14.1 (default PyPI wheel, `+cu130` build, executed on CPU) |
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| dp-accounting | 0.6.0 |
| pytest | 9.1.1 |
| jupyter / nbconvert / nbclient / ipykernel | 1.1.1 / 7.17.2 / 0.11.0 / 7.4.0 |
| `pip check` | "No broken requirements found." |
| `torchcsprng` | not installable: "No matching distribution found" for this Python/PyTorch |

## Commands

```bash
cd fl
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements.txt        # 2 min 09 s
python -m pip check
python -m pytest -q
jupyter nbconvert --to notebook --execute opendp_opacus_fl.ipynb --output opendp_opacus_fl_executed.ipynb
```

## Tests

`python -m pytest -q -rA`: **91 tests run: 91 passed, 0 failed, 0 skipped, 0 errors**
in 33.5 s. The only log line is Opacus's informational warning about an empty
first batch, which a test triggers on purpose (`sample_rate=0.01` on 40 records).

| Area | Tests (examples) |
|---|---|
| Neighbouring datasets, public parameters, fixed normaliser | `test_neighbouring_datasets_run_identical_mechanisms`, `test_normaliser_is_a_public_constant_not_the_dataset_size` |
| Poisson semantics and exact rates | `test_poisson_sampling_semantics_and_rate[csprng/torch]`, `test_exact_inclusion_probabilities_versus_accounted_rate`, `test_csprng_threshold_rule_is_exact`, `test_torch_rand_grid_assumption_holds_for_installed_torch` |
| Empty batches | `test_local_round_takes_exactly_k_steps_including_empty_batches`, `test_noise_std_on_an_empty_batch_is_still_sigma_c_over_b` |
| Clipping and noise | `test_per_example_clipping_bounds_the_noiseless_update`, `test_empirical_noise_std_is_sigma_c_over_b[True/False]` |
| Sampler/accountant consistency | `test_accountant_uses_the_rate_and_steps_of_the_sampler_that_ran`, `test_reproducible_mode_accounts_with_the_torch_rate` |
| Rényi divergence and conversion | Opacus reference (4 cases), quadrature both directions (12 cases), `test_rdp_to_approx_dp_against_direct_formula`, `test_independent_google_dp_accounting_rdp_and_pld` |
| Composition | `test_step_and_round_composition_is_additive`, `test_opendp_composition_adds_rdp_curves`, `test_parallel_vs_sequential_bound`, `test_a_clients_epsilon_does_not_depend_on_other_clients` |
| Budget, failure handling | `test_budget_exceeded_stops_the_whole_round_before_training`, `test_broken_privacy_map_is_refused_before_any_training`, `test_failed_round_is_charged_and_halts_the_client` |
| Invalid inputs and configurations | invalid mechanism/config parameters (17 cases), misconfigured clients (5 cases), shared records |
| Randomness | `test_private_mode_draws_fresh_secret_seeds_for_every_client_and_round`, `test_csprng_sampler_draws_from_the_os_csprng`, `test_two_clients_with_identical_inputs_get_independent_noise` |
| OpenDP integration | `test_opendp_requires_features_in_a_fresh_interpreter`, `test_opendp_api_surface_used_by_the_example`, odometer agreement, upstream characterisation test |
| End-to-end | `test_end_to_end_training_learns_and_reports_epsilon`, `test_rounds_are_accounted_through_the_odometer` |

### Mutation check

Seven deliberate bugs were injected one at a time. The suite was rerun after
each, and the original code was restored afterwards:

| Injected bug | Tests that failed |
|---|---|
| normaliser B = round(q · n_k) | 2 |
| skip the pre-run budget/map check | 1 |
| failed round not charged | 1 |
| torch-mode accounting uses q instead of the true rate | 2 |
| CSPRNG threshold uses ceil instead of floor | 1 |
| `secure_mode` off by default | 1 |
| sampler takes K + 1 steps | 6 |

## Notebook

`jupyter nbconvert --execute`: all **7 of 7 code cells executed, in order
(1–7)**, with **0 errors** and **0 stderr outputs**, in about 30 s. Outputs:

| Quantity | Value |
|---|---|
| ε per client (δ = 1e-5), q = 0.05, σ = 1.0, K = 20, 5 rounds | 4.1117 (best Rényi order 5) |
| Record-level ε, disjoint clients (max) | 4.1117 |
| Sequential bound over all clients (if records could repeat) | 6.5127 |
| Cross-check: Opacus RDP analysis | 4.1117 |
| Cross-check: Google dp_accounting RDP, same orders | 4.1117 |
| Reference: Google dp_accounting PLD (tighter; not this code's guarantee) | 3.5021 |
| Budget ε = 5 | round 9 refused (would reach 5.1428); 8 rounds recorded, final ε = 4.9145 |
| Test accuracy after 5 rounds | 0.791 in this run (0.447 before training). It varies between runs because the noise is unseeded by design. |

The ε values are deterministic: they depend only on public parameters. The
accuracy and loss values are not, by design.

## Not executed / not verified

* No GPU run; macOS and Windows were not tested.
* No run against the upcoming OpenDP–Opacus branch, which is not yet available.
* No CSPRNG-backed noise generator: `torchcsprng` is unavailable.
* No GitHub Actions run. A CI workflow can be added once the target repository's
  layout is decided.
