# Privacy proof sketch

This is a **proof sketch**, not a formal proof. Every step is either derived
here or attributed to a published result, and the remaining proof obligations
are listed in §9. The argument is in real arithmetic. The gap between that and
the floating-point implementation is assumption H6.

Notation: m clients; client k holds a local dataset D_k (a multiset of records);
the global dataset is D = D_1 ⊎ … ⊎ D_m. Model parameters θ ∈ ℝ^p. Loss
ℓ(θ; x). Public constants (all from `DPConfig`): sampling rate q ∈ (0, 1], noise
multiplier σ > 0, clipping bound C > 0, normaliser B ∈ ℕ_{≥1}, steps per round K,
rounds R, learning rate η, failure probability δ ∈ (0, 1).

## 1. Adjacency

**Definition 1 (neighbouring global datasets).** D ≃ D′ if D′ is obtained from D
by adding one record x or removing one record x. Records are routed to clients by
a fixed public function a(·) of the record alone (`fl.assign_client`). So D ≃ D′
changes exactly one local dataset, D_{a(x)}, by one record, and leaves every other
D_j unchanged.

**Definition 2 (input domain).** The guarantee is stated for pairs D ≃ D′ in
which every local dataset on both sides is non-empty (`fl.client_domain()`:
`ClientDataset` with n ≥ 1).

In OpenDP terms, a local round is a measurement on `client_domain()` with input
metric `symmetric_distance`. Its privacy map is defined for d_in ∈ {0, 1} and
raises for d_in ≥ 2, so group privacy is not claimed.

## 2. Mechanism

For client k, round r, starting from the public global model θ_{k,r,0} = θ̄_r:

for t = 1, …, K:

1. **Sampling.** S_t = { i ∈ D_k : U_{i,t} < τ }, where the U_{i,t} are fresh,
   independent draws uniform on {0, …, 2⁵³ − 1}, τ = ⌊q·2⁵³⌋, from the OS CSPRNG
   (`CSPRNGPoissonSampler`). Each record is included independently with
   probability q̃ = τ/2⁵³ ≤ q.
2. **Clipping.** g_i = ∇ℓ(θ_{t−1}; x_i) and ĝ_i = g_i · min(1, C/‖g_i‖₂)
   (Opacus flat clipping over all parameters).
3. **Noise.** G_t = Σ_{i∈S_t} ĝ_i + Z_t with Z_t ~ N(0, σ²C² I_p), independent of
   everything else (Opacus `DPOptimizer`).
4. **Update.** θ_t = θ_{t−1} − (η/B) G_t, with B a public constant (Opacus
   `loss_reduction="mean"` divides by `expected_batch_size = B`).

Client k releases θ_{k,r} = θ_K, and nothing else. The coordinator computes
θ̄_{r+1} = (1/m) Σ_k θ_{k,r}, with uniform weights that do not depend on the data.
Empty batches (S_t = ∅) still execute step 3: G_t = Z_t.

## 3. Sensitivity

Fix t and the past (θ_{t−1} and all earlier releases). Let D_k ≃ D′_k differ by
record x*. Write G_t on the larger dataset and G′_t on the smaller. Conditional
on whether x* ∈ S_t:

* if x* ∉ S_t, then Σ ĝ_i is identical on both sides;
* if x* ∈ S_t, the sums differ by ĝ_{x*}, with ‖ĝ_{x*}‖₂ ≤ C.

All other inclusion decisions have the same distribution on both sides, because
the U_{i,t} are independent per record. Dividing by C, G_t/C is the
**Poisson-subsampled Gaussian mechanism** with L2 sensitivity 1, rate q̃ and noise
standard deviation σ. The division by the public constant B, the SGD update, and
releasing θ_K instead of (G_1, …, G_K) are post-processing (H4 below). If B
depended on |D_k| this step would fail. That was the bug fixed in this revision.

## 4. Per-step RDP

Let ε_SG(α; q, σ) denote the RDP of the sampled Gaussian mechanism at order α.

**Fact 1 [MTZ19, §3].** The worst case over sensitivity-1 inputs reduces to one
dimension, P = (1−q)N(0, σ²) + qN(1, σ²) against Q = N(0, σ²). For integer
α ≥ 2,

  A_α = Σ_{k=0}^{α} C(α, k) (1−q)^{α−k} q^k exp((k² − k)/(2σ²)),  D_α(P‖Q) = log A_α / (α − 1),

and D_α(P‖Q) ≥ D_α(Q‖P), so ε_SG(α) = log A_α/(α−1) bounds both the add and
the remove direction. *Implemented:* `accounting.sampled_gaussian_rdp`.
*Checked:* agrees with Opacus `compute_rdp` and Google `dp_accounting`
(rel. ≤ 1e-6), and with numerical quadrature of D_α(P‖Q). Quadrature confirms
D_α(Q‖P) ≤ ε_SG at the points tested.

**Fact 2 (monotonicity in q).** A_α(q) = E_Q[(1 − q + qL)^α] with L = dP₁/dQ
and E_Q[L] = 1. It is convex in q, and A_α(0) = 1 ≤ A_α(q) by Jensen's
inequality. A convex function whose minimum on [0, 1] is at the left endpoint is
non-decreasing. So ε_SG is non-decreasing in q, and accounting with q ≥ q̃ is
valid. *Used for:* q̃ = ⌊q·2⁵³⌋/2⁵³ ≤ q (CSPRNG sampler). In debug mode the torch
sampler includes with probability ⌈q·2²⁴⌉/2²⁴ ≥ q, so the code accounts with that
larger value (`fl.accounted_sample_rate`).

**Fact 3 (orders).** D_α is non-decreasing in α [van Erven & Harremoës 2014], so
for non-integer α, ε(α) ≤ ε(⌈α⌉). The code evaluates α = 2…256 and returns +∞
above 256.

## 5. Sequential composition (steps and rounds of one client)

**Fact 4 (adaptive composition of RDP) [Mironov 2017, Prop. 1].** If M₁ is
(α, ε₁)-RDP and M₂(·, y₁) is (α, ε₂)-RDP for every y₁, then (M₁, M₂) is
(α, ε₁ + ε₂)-RDP.

Within a round, step t depends on the earlier steps only through θ_{t−1}. Across
rounds, the starting model θ̄_r depends on earlier releases by all clients. Each
step is (α, ε_SG(α))-RDP conditional on the past, by §3–4, and its internal
randomness is fresh (H5). So client k's R rounds of K steps are
(α, R·K·ε_SG(α))-RDP for every α.

*In code:* each round is an OpenDP `make_user_measurement` with output measure
`renyi_divergence`, whose map returns K·ε_SG. The rounds are composed by OpenDP:
`make_composition` (reported loss and budget check) and
`make_fully_adaptive_composition` (execution). Because q, σ and K are fixed in
advance, plain adaptive composition suffices. The odometer is used for
execution and bookkeeping, not for adaptively chosen privacy parameters.

**Stopping rule.** The number of rounds is decided by the budget check, which
depends only on public parameters. So it is a fixed, data-independent R, and the
rounds actually run are a prefix of a fixed schedule.

## 6. Parallel composition across clients

**Claim.** For D ≃ D′ differing in client k* = a(x*), the coordinator's whole
view (all θ_{k,r} for all k and r) is (α, R·K·ε_SG(α))-RDP.

*Argument.* Order all mechanisms by execution order. Each mechanism M_{j,r} with
j ≠ k* reads only D_j, which is identical under D and D′, and its other inputs
come from the shared past transcript. Its conditional output distributions given
the past are therefore identical on D and D′: it is (α, 0)-RDP for this pair.
Applying Fact 4 along the execution order, only client k*'s mechanisms
contribute, giving R·K·ε_SG(α). Since k* can be any client, the federation-level
bound is max_k of the per-client bounds. All clients share the same (q, σ, K, R)
here, so the bound is the same for every client.

*Conditions.* (i) Each record maps to exactly one client by a(·) (enforced:
`make_clients` rejects shared record ids). (ii) No mechanism of client j reads
another client's records. This holds by construction in the simulation, and in a
real deployment it is an isolation assumption (H7). (iii) Randomness is
independent across mechanisms (H5).

*If records can repeat across clients,* use sequential composition over all
clients instead (`accounting.sequential_bound`; 6.5127 in the notebook).

## 7. RDP to (ε, δ)

**Fact 5 [Balle et al. 2020, Thm. 21; Canonne, Kamath & Steinke 2020, Prop. 12].**
If M is (α, τ)-RDP with α > 1, then M is (ε, δ)-DP for

  ε = τ + log((α − 1)/α) − (log δ + log α)/(α − 1).

Each α gives a valid bound, so the minimum over α = 2…256 is valid
(`accounting.rdp_to_approx_dp`). A warning is issued if the minimiser is α = 256,
where the bound may be loose. Negative values are clipped to 0, which is still a
valid bound.

**Result.** With q = 0.05, σ = 1, K = 20 and R = 5 (100 steps) at δ = 1e-5:
ε = 4.1117 per client and at the record level. Opacus and Google `dp_accounting`
(RDP, same orders) agree. Google `dp_accounting` with fractional orders gives
4.0389, and its PLD accountant gives 3.5021. Those are tighter bounds for the
same mechanism, and they show that the reported number is conservative.

## 8. Assumptions (numbered)

* **H1** The guarantee is for neighbours as in Definitions 1–2: non-empty client
  datasets, add/remove one record, d_in = 1.
* **H2** Every mechanism parameter (q, σ, C, B, K, R, η, δ, the architecture
  and θ̄_0) is public and fixed independently of the data.
* **H3** Each record is held by exactly one client, assigned by a function of the
  record alone.
* **H4** Opacus computes exact per-example gradients for the model, clips them to
  C in L2 over all parameters, adds N(0, σ²C²) to their sum, and divides by B.
  This is checked by tests (clipping bound, noise standard deviation including
  empty batches, parameters passed); Opacus internals are trusted, not
  verified.
* **H5** Randomness is ideal and independent: U_{i,t} are uniform and
  independent (OS CSPRNG); the Z_t are exact Gaussians, independent across
  coordinates, steps, rounds and clients, and unknown to the adversary.
* **H6** Real-arithmetic idealisation: floating-point effects on sampling and
  noise are ignored (see §9).
* **H7** Each client's code, hardware and RNG are trusted, and clients are
  isolated from each other. In this example they share one Python process.
* **H8** Only θ_{k,r} leaves a client. Failures are handled fail-closed: a round
  that raises is charged, and the client halts. Exceptions in the training path
  are data-independent: step-count assertion, configuration errors.

## 9. Limitations and open proof obligations

1. **Floating-point noise (H6).** The analysis is for real-valued Gaussians.
   Opacus `secure_mode=True` uses the Holohan & Braghin (2021) mitigation against
   known floating-point attacks [Mironov 2012; Jin et al. 2022], but no
   floating-point DP proof is offered.
2. **Noise RNG (H5).** Noise comes from `torch.Generator` (Mersenne Twister),
   seeded per client and round from `secrets.randbits(63)`. This is not a CSPRNG.
   `torchcsprng` is archived and has no build for Python 3.13 / PyTorch 2.14.
   Sampling decisions do use the OS CSPRNG.
3. **Fact 1 is relied on, not re-proved.** The dominance D_α(P‖Q) ≥ D_α(Q‖P) and
   the reduction to one dimension come from MTZ19; quadrature checks it at
   12 (q, σ, α) points only.
4. **The parallel-composition argument in §6 is written out here but not
   machine-checked, and it is not expressed in OpenDP.** An OpenDP-native
   partition combinator would let the framework carry this step.
5. **OpenDP odometer semantics.** The fully adaptive odometer is used only to
   execute rounds; the guarantee in §5 needs only adaptive composition with
   fixed parameters. Two observed OpenDP 0.16.0 behaviours are handled in code,
   not by OpenDP: a measurement with a broken map is executed (and only fails at
   `privacy_loss`), and a failed function is not charged.
6. **Enforcement is in Python.** `make_privacy_filter` over an RDP odometer
   aborted the interpreter in our tests, so the budget is enforced by
   `ClientPrivacyOdometer`, which code holding the raw queryable could bypass.
7. **Not client-level DP.** No guarantee for a client's whole dataset.

## References

* I. Mironov. Rényi Differential Privacy. CSF 2017.
* I. Mironov, K. Talwar, L. Zhang. Rényi Differential Privacy of the Sampled
  Gaussian Mechanism. arXiv:1908.10530, 2019. (MTZ19)
* B. Balle, G. Barthe, M. Gaboardi, J. Hsu, T. Sato. Hypothesis Testing
  Interpretations and Rényi Differential Privacy. AISTATS 2020.
* C. Canonne, G. Kamath, T. Steinke. The Discrete Gaussian for Differential
  Privacy. NeurIPS 2020.
* T. van Erven, P. Harremoës. Rényi Divergence and Kullback–Leibler Divergence.
  IEEE Trans. Inf. Theory, 2014.
* I. Mironov. On Significance of the Least Significant Bits for Differential
  Privacy. CCS 2012.
* N. Holohan, S. Braghin. Secure Random Sampling in Differential Privacy.
  ESORICS 2021 (arXiv:2107.10138).
* J. Jin, E. McMurtry, B. Rubinstein, O. Ohrimenko. Are We There Yet?
  Timing and Floating-Point Attacks on Differential Privacy Systems. IEEE S&P
  2022 (arXiv:2112.05307).
* OpenDP 0.16.0 documentation and docstrings for `make_user_measurement`,
  `renyi_divergence`, `make_composition`, `make_fully_adaptive_composition`.
* Opacus 1.6.0 source: `DPOptimizer`, `_generate_noise` (`secure_mode`),
  `UniformWithReplacementSampler`.
