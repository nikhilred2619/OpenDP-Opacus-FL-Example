# Privacy assumptions and limitations

This file states what the example claims, under which assumptions, and what has
not been established. It separates four kinds of statement:

* **[lit]** a result from the literature that we rely on but do not re-prove,
* **[impl]** a property enforced by the code,
* **[test]** a property checked empirically or against an independent reference,
* **[assumed]** an assumption that is neither proved nor checked here.

Passing tests do not prove a differential-privacy guarantee, and no formal proof
of this example is supplied. The step-by-step argument is in
[`PRIVACY_PROOF_SKETCH.md`](PRIVACY_PROOF_SKETCH.md), with assumptions H1–H8 and
the open proof obligations.

## The claim

Suppose assumptions A1–A8 hold. Then for each client k, the sequence of model
states that client k releases over all rounds satisfies (ε_k, δ)-DP with respect
to adding or removing one record from client k's non-empty local dataset. Under
A2, the federation's entire output satisfies (max_k ε_k, δ)-DP at the record
level.

Notebook configuration: q = 0.05, σ = 1.0, C = 1.0, B = 50, K = 20, 5 rounds,
δ = 1e-5. Result: ε_k = 4.1117 for every client, so the record-level ε is 4.1117.
ε depends only on (q, σ, K, rounds, δ). The fix to B described below changed
whether the bound is *valid*, not its value.

## Mechanism

For each client, round and step t = 1…K:

1. Sample S_t by including each local record independently with probability
   q̃ = ⌊q·2⁵³⌋/2⁵³ ≤ q (Poisson), using the OS CSPRNG. **[impl]**, **[test]**
   (Monte Carlo inclusion frequency and batch-size variance; the exact rate is ≤ q)
2. Compute per-example gradients and clip each to L2 norm ≤ C. **[impl]** by
   Opacus, **[test]** (update-norm bound)
3. Release internally g̃_t = (Σ_{i∈S_t} clip(g_i) + N(0, σ²C²·I)) / B and take
   an SGD step. **[impl]** by Opacus, **[test]** (empirical noise std = σC/B,
   with and without `secure_mode`)
4. After K steps, release the model state. Nothing else leaves the client: no
   empty-batch count, no loss, no dataset size. **[impl]**, **[test]**

Add/remove changes Σ clip(g_i) by at most C in L2, so each step is the sampled
Gaussian mechanism with noise multiplier σ. Dividing by a **public** B, the SGD
update and the release of the final state are post-processing.

## Accounting

| Step | Statement | Kind |
|---|---|---|
| Per-step RDP | For integer α ≥ 2, ε_step(α) = log(A_α)/(α−1) with A_α = Σ_k C(α,k)(1−q)^{α−k} q^k exp((k²−k)/(2σ²)). This is exactly D_α(P‖Q) for P = (1−q)N(0,σ²) + qN(1,σ²) and Q = N(0,σ²). | [lit] Mironov, Talwar & Zhang 2019; [test] equals Opacus's `compute_rdp` (rel. 1e-9) and quadrature (rel. 1e-6) |
| Both directions | D_α(P‖Q) ≥ D_α(Q‖P), so the same bound covers adding and removing a record. | [lit] MTZ19; [test] quadrature at α = 2, 3, 5 for four (q, σ) pairs |
| Non-integer α | Rényi divergence is non-decreasing in α, so ε(α) ≤ ε(⌈α⌉). Orders above 256 return +∞. | [lit] standard; [impl] |
| Steps and rounds | RDP composes additively under adaptive composition. A round is K × ε_step. A client's rounds are composed by OpenDP's fully adaptive composition odometer. | [lit] Mironov 2017; OpenDP odometer proof (WRRW23); [test] |
| (ε, δ) | ε = min_α ε_RDP(α) + log((α−1)/α) − (log δ + log α)/(α−1), over α = 2…256. | [lit] Canonne, Kamath & Steinke 2020; Balle et al. 2020, Thm. 21; [test] matches Opacus `get_privacy_spent` and a brute-force recomputation |
| Federation | Disjoint clients, and only client k's mechanism reads client k's records. So a record's loss is ε_k, and the federation's record-level ε is max_k ε_k. | [lit] parallel composition under adaptivity (other clients' mechanisms are independent of the record given the public transcript); [impl] disjointness enforced; [test] |

## Neighbouring datasets and input domain

| Item | Choice |
|---|---|
| Privacy unit | One record (record-level DP). **Not** client-level DP. |
| Adjacency | Add or remove one record in one client's dataset (`dp.symmetric_distance()`, `d_in = 1`). The privacy map raises for `d_in > 1`: group privacy is not claimed. |
| Input domain | Non-empty `ClientDataset`s (`client_domain()`). Pairs where one side is empty are outside the guarantee. |
| δ | 1e-5, a public constant chosen below 1/n_k for every client here. |

## Assumptions

| # | Assumption | Status |
|---|---|---|
| A1 | **Mechanism matches the map:** exactly K steps, Poisson rate ≤ the accounted q, clipping C, noise σC. | [impl] step count asserted, the same config object feeds sampler, optimizer and map; [test] the sampler's rate and steps and the optimizer's σ match the curve in the odometer. Opacus internals are trusted, not verified. |
| A2 | **Disjoint, record-determined clients.** Each record is held by one client, assigned by a function of the record alone. | [impl] `make_clients` refuses shared record ids; [test]. With overlap, use the sequential bound (6.5127 in the notebook). |
| A3 | **Public mechanism parameters:** q, σ, C, B, K, number of rounds, δ, architecture and learning rate are fixed in advance and not derived from data. | [impl] all from `DPConfig`; [test] neighbouring datasets yield identical optimizer and sampler parameters; B no longer depends on n_k (regression test; this test fails on the old code). |
| A4 | **Fixed participation:** every client runs every round. The number of rounds is decided by the public budget check, not by data. | [impl] |
| A5 | **Ideal randomness:** Gaussian noise behaves as an ideal random variable unknown to the adversary. Sampling uses the OS CSPRNG. | Sampling: [impl] CSPRNG. Noise: [assumed], not a CSPRNG. See Randomness below. |
| A6 | **Only the model state leaves a client.** | [impl]; [test] |
| A7 | **honest-but-curious contract:** the privacy map is correct for the function, and the function has no side effects other than consuming its own fresh generators. | Required by `make_user_measurement`. OpenDP cannot check it. |
| A8 | **The test set and initial model are public** and independent of client data. | [impl] synthetic, seeded |

## What changed in this revision, and why

* **Normaliser B.** Previously `expected_batch_size = round(q · n_k)`, so the
  released update depended on n_k. Under add/remove adjacency, n_k differs
  between neighbours, which can change B. Then the release is no longer a
  post-processing of the sampled Gaussian output. The previous documentation
  covered this only by assuming n_k public, and an exact dataset size is itself
  sensitive under add/remove. B is now a declared constant
  (`DPConfig.expected_batch_size`). The neighbour-invariance and regression tests
  fail if B is derived from n_k again.
* **Input domain** now excludes empty datasets explicitly. The mechanism
  needs at least one record, and the claim is now stated for that domain.
* **Disjointness** is now enforced: records carry ids, and `make_clients`
  rejects shared ids. Previously it was only argued.
* **All-or-nothing budget check** across clients before each round.
* **`secure_mode=True`** by default (Opacus floating-point mitigation).
* **`debug_seed` without `reproducible_noise`** is now refused in
  `run_federated`, so a known seed cannot be passed by mistake in private mode.

Final validation pass:

* **CSPRNG Poisson sampling** (`CSPRNGPoissonSampler`, default). The integer
  threshold rule makes the true inclusion probability ⌊q·2⁵³⌋/2⁵³ ≤ q, so the
  accountant's q is an upper bound.
* **Sampling-rate finding (torch path).** Opacus's `UniformWithReplacementSampler`
  includes a record iff `torch.rand(...) < q`. `torch.rand` (float32) returns
  multiples of 2⁻²⁴, so the true inclusion probability is ⌈q·2²⁴⌉/2²⁴, which can
  exceed q (0.0500000119 for q = 0.05). An accountant that uses q then
  under-reports, by a negligible amount here, but it is not a valid upper bound.
  The torch sampler is now used only in reproducible (debug) mode, and that mode
  accounts with the corrected rate (`fl.accounted_sample_rate`).
* **Fail-closed accounting.** OpenDP 0.16.0's odometer executes a measurement
  whose privacy map raises, and the error surfaces only at `privacy_loss`. It
  also does not charge a round whose function raised. `ClientPrivacyOdometer`
  now evaluates the map before running, charges every attempted round, and
  halts the client after a failure.
* **Independent accountant.** Results are compared with Google `dp_accounting`
  0.6.0: RDP with the same orders agrees, and PLD gives a smaller ε (3.5021).

## Threat model

* **Untrusted:** the coordinator and any coalition of other clients. They
  observe every model state each client releases, in every round. They may
  choose the global model adaptively, which is covered by A3/A4 and fully
  adaptive composition.
* **Visible metadata:** which clients participate (all of them), the round
  count, all public parameters in `DPConfig`, and the model architecture.
  Client dataset sizes are **not** released by the mechanism. The notebook
  prints them only to illustrate the partition.
* **Trusted:** each client's code, hardware and RNG.
* **Not assumed:** secure aggregation. No claim about collusion resistance is
  needed or made.
* **Out of scope:** side channels such as timing, memory and power; malicious
  clients poisoning the model (an integrity issue, not privacy); attacks on the
  RNG; and protection of a client's whole dataset (client-level DP).
* **Simulation caveat:** all clients run in one Python process. The isolation
  between parties that a real deployment would have is assumed, not provided.

## Randomness

* **Sampling** (default): OS CSPRNG via `secrets.SystemRandom().getrandbits(53)`,
  with one fresh draw per record per step. **[impl]**, **[test]** (draw count; exact
  rate ≤ q; Monte Carlo)
* **Noise**: a `torch.Generator` freshly seeded from `secrets.randbits(63)` for every
  client and every round. No seed is reused, stored or released. **[impl]**,
  **[test]** (all seeds unique; independent noise for identical inputs)
* Opacus `secure_mode=True` applies its floating-point mitigation: four
  Gaussian draws per coordinate (Holohan & Braghin 2021; see the Opacus
  `_generate_noise` docstring). **[impl]**, **[test]** (noise std unchanged)
* **Noise is not cryptographic.** `torchcsprng`, which Opacus's `PrivacyEngine`
  uses for a CSPRNG generator under `secure_mode`, has no distribution for
  Python 3.13 / PyTorch 2.14 (`pip install torchcsprng`: "No matching
  distribution"). `torch.Generator` is a Mersenne Twister, not a CSPRNG.
  Its state could in principle be recovered from enough outputs. Here the outputs
  are not observed directly, but no analysis of that is offered. A production
  deployment would need a CSPRNG-backed sampler (for example OpenDP's own
  samplers, or Opacus with a cryptographic generator). This example is a
  **research demonstration with idealized randomness (A5)**.
* `reproducible_noise=True` seeds the generators with known values for debugging
  and tests. It warns that **no** guarantee holds. Model initialisation and the
  synthetic data are seeded, which is fine because they are not private.

## Known limitations

1. **RDP slack.** The RDP → (ε, δ) conversion is not tight. A PRV/PLD accountant
   would report a smaller ε for the same mechanism. OpenDP 0.16 has no PRV support.
2. **Integer orders 2–256 only.** Valid, possibly slightly loose.
3. **Non-private tuning.** The learning rate was chosen by hand on synthetic data.
4. **Budget enforcement is in Python.** `make_privacy_filter` over an RDP
   odometer aborted the process in OpenDP 0.16.0. The check composes the
   prospective curve with OpenDP's `make_composition` instead. It is not an
   OpenDP-enforced filter: code holding the raw queryable could bypass it.
5. **No clipping schedule.** C is fixed, so Opacus's clipping schedulers, and the
   reported issue with them, are not involved.
6. **Simulation only.** No transport, dropout, stragglers or real data.
