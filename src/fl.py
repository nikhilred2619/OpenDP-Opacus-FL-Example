"""Minimal cross-silo federated learning with Opacus DP-SGD and OpenDP accounting.

Kept deliberately small: synthetic data, a handful of clients, a two-layer MLP,
plain FedAvg simulated in one process. The privacy-relevant choices are explicit:

* **Disjoint, record-determined partition.** Records are assigned to clients by
  a fixed rule on the record itself (:func:`assign_client`), so client datasets
  are disjoint, and adding or removing a record never moves any other record.
  :func:`make_clients` refuses clients that share a record id.
* **Only public constants parameterise the mechanism.** The Poisson rate ``q``,
  the step count ``K``, the noise multiplier ``sigma``, the clipping bound ``C``
  and the gradient normaliser ``B`` (``expected_batch_size``) all come from
  :class:`DPConfig`. None is derived from a client's data or dataset size, so
  two neighbouring datasets (one record added or removed) run the exact same
  mechanism. An earlier draft set ``B = round(q * n_k)``. That made the
  released update a function of ``n_k`` that the privacy map did not account
  for, unless ``n_k`` was assumed public, so it was removed.
* **The accountant models the sampler that runs.**
  ``UniformWithReplacementSampler(steps=K)`` draws an independent Bernoulli(q)
  for each record at each of exactly ``K`` steps. The same ``(q, sigma, K)`` is
  passed to the OpenDP privacy map, and the step count is asserted.
* Opacus's ``PrivacyEngine`` and its accountant are not used. The clipping
  bound is a fixed constant (no clipping scheduler).
* **Randomness.** Poisson sampling uses the OS CSPRNG (``secrets.SystemRandom``)
  by default. Gaussian noise uses a ``torch.Generator`` freshly seeded from
  ``secrets.randbits`` for every client and every round, with Opacus
  ``secure_mode`` (its floating-point mitigation). That generator is PyTorch's
  Mersenne Twister, **not** a CSPRNG: ``torchcsprng`` has no build for current
  PyTorch/Python, so noise generation is not cryptographically secure (see
  PRIVACY_ASSUMPTIONS.md). ``reproducible_noise=True`` is a debugging switch
  that seeds sampling and noise with known values and warns that the
  guarantee is void.
"""
from __future__ import annotations

import copy
import math
import secrets
import warnings
from fractions import Fraction
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, TensorDataset
from torch.utils.data.dataloader import default_collate

from opacus import GradSampleModule
from opacus.data_loader import wrap_collate_with_empty
from opacus.optimizers import DPOptimizer
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

import opendp.prelude as dp

from . import accounting as acc

#: Public, data-independent thresholds on feature 0 used to route records to clients.
CLIENT_THRESHOLDS = (-0.4, 0.4)


# --------------------------------------------------------------------------- data
def make_synthetic_data(n_train: int = 3000, n_test: int = 1000, d: int = 8, seed: int = 0):
    """Synthetic binary classification data from one fixed generator.

    Returns ``(X_train, y_train, X_test, y_test)``. Non-private and seeded for
    reproducibility; the test set is never seen by any client.
    """
    if n_train < 1 or n_test < 1 or d < 1:
        raise ValueError("n_train, n_test and d must be positive")
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=d)
    logits = X @ w + 0.5 * np.sin(3 * X[:, 0])
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(np.float32)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def assign_client(record: np.ndarray, thresholds: Sequence[float] = CLIENT_THRESHOLDS) -> int:
    """Client id of one record: a fixed function of that record only (non-IID by feature 0)."""
    return int(np.searchsorted(np.asarray(thresholds), record[0], side="right"))


def partition_records(X: np.ndarray, thresholds: Sequence[float] = CLIENT_THRESHOLDS) -> List[np.ndarray]:
    """Disjoint, covering partition of row indices into ``len(thresholds) + 1`` clients."""
    ids = np.array([assign_client(x, thresholds) for x in X])
    return [np.flatnonzero(ids == k) for k in range(len(thresholds) + 1)]


class ClientDataset(TensorDataset):
    """A client's local records, with the global ids of those records.

    ``record_ids`` let :func:`make_clients` verify that no record is held by two
    clients, which the parallel-composition argument requires. The ids are
    used only for that check and never enter training or any release.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, client_id: int, *, record_ids: Sequence[int]):
        X, y = np.ascontiguousarray(X), np.ascontiguousarray(y)
        record_ids = np.asarray(record_ids)
        if not (len(X) == len(y) == len(record_ids)):
            raise ValueError("X, y and record_ids must have the same length")
        if len(np.unique(record_ids)) != len(record_ids):
            raise ValueError(f"client {client_id}: duplicate record ids")
        super().__init__(torch.from_numpy(X), torch.from_numpy(y))
        self.client_id = client_id
        self.record_ids = record_ids


def client_domain() -> dp.Domain:
    """OpenDP input domain: non-empty ClientDatasets.

    The guarantee covers pairs of neighbouring datasets that are both in this
    domain, so both have at least one record. The empty dataset is excluded
    because the sampler and model need at least one record to determine shapes.
    """
    return dp.user_domain("ClientDataset(n>=1)", lambda x: isinstance(x, ClientDataset) and len(x) >= 1)


# -------------------------------------------------------------------------- model
def make_model(d: int, seed: int = 0) -> nn.Module:
    """Two-layer MLP. Seeded initialisation (non-private) without touching the global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1))


def get_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


# ------------------------------------------------------------------------- config
@dataclass(frozen=True)
class DPConfig:
    """Public DP-SGD and federation parameters, identical for all clients.

    Every field is a declared public constant. None may be computed from a
    client's records or dataset size.
    """
    sample_rate: float = 0.05        # Poisson inclusion probability q
    noise_multiplier: float = 1.0    # sigma: noise std on the clipped sum = sigma * C
    max_grad_norm: float = 1.0       # C: fixed per-example clipping bound (L2)
    expected_batch_size: int = 50    # B: public normaliser, update = (clipped sum + noise) / B
    local_steps: int = 20            # K noisy steps per client per round
    rounds: int = 5                  # federated rounds
    lr: float = 0.5
    target_epsilon: float = 5.0      # per-client budget at delta
    delta: float = 1e-5
    secure_noise: bool = True        # Opacus secure_mode (floating-point mitigation); not a CSPRNG
    sampling_rng: str = "csprng"     # "csprng": OS CSPRNG (secrets.SystemRandom) for Poisson sampling; "torch"
    reproducible_noise: bool = False  # debugging only; voids the guarantee (warns)

    def __post_init__(self):
        acc._validate_mechanism(self.sample_rate, self.noise_multiplier, self.local_steps)
        if not (self.max_grad_norm > 0 and math.isfinite(self.max_grad_norm)):
            raise ValueError("max_grad_norm must be positive and finite")
        if not (isinstance(self.expected_batch_size, int) and not isinstance(self.expected_batch_size, bool)
                and self.expected_batch_size >= 1):
            raise ValueError("expected_batch_size must be a positive integer (a public constant)")
        if not (isinstance(self.rounds, int) and self.rounds >= 1):
            raise ValueError("rounds must be a positive integer")
        if not (self.lr > 0 and math.isfinite(self.lr)):
            raise ValueError("lr must be positive and finite")
        if not (0 < self.delta < 1):
            raise ValueError("delta must be in (0, 1)")
        if not (self.target_epsilon > 0 and math.isfinite(self.target_epsilon)):
            raise ValueError("target_epsilon must be positive and finite")
        if self.sampling_rng not in ("csprng", "torch"):
            raise ValueError('sampling_rng must be "csprng" or "torch"')


def _generator(seed: Optional[int]) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(secrets.randbits(63) if seed is None else seed)
    return g


# ------------------------------------------------------------- mechanism pieces
#: Bits of OS randomness per inclusion decision in :class:`CSPRNGPoissonSampler`.
CSPRNG_BITS = 53
#: ``torch.rand`` (float32) returns multiples of 2^-24; Opacus's sampler includes a record iff U < q.
TORCH_RAND_BITS = 24


def csprng_inclusion_probability(q: float) -> Fraction:
    """Exact inclusion probability of :class:`CSPRNGPoissonSampler`: floor(q 2^53) / 2^53 <= q."""
    return Fraction(math.floor(Fraction(q) * 2 ** CSPRNG_BITS), 2 ** CSPRNG_BITS)


def torch_inclusion_probability(q: float) -> Fraction:
    """Exact inclusion probability of ``torch.rand(n) < q`` on the 2^-24 float32 grid: ceil(q 2^24) / 2^24 >= q."""
    return Fraction(math.ceil(Fraction(q) * 2 ** TORCH_RAND_BITS), 2 ** TORCH_RAND_BITS)


def sampler_kind(cfg: "DPConfig") -> str:
    """Which sampler ``local_dp_round`` uses: reproducible (debug) mode always uses the seeded torch sampler."""
    return "torch" if (cfg.sampling_rng == "torch" or cfg.reproducible_noise) else "csprng"


def accounted_sample_rate(cfg: "DPConfig") -> float:
    """The Poisson rate the privacy map must use: an upper bound on the true inclusion probability.

    The sampled-Gaussian RDP is non-decreasing in q, so accounting with any
    q_acc >= q_true is valid.

    * ``csprng``: the true rate is floor(q 2^53)/2^53 <= q, so q itself is a valid bound.
    * ``torch``: ``torch.rand < q`` includes a record with probability
      ceil(q 2^24)/2^24, which can *exceed* q (by 1.2e-8 for q = 0.05).
      Accounting with q would under-report, so the next float at or above the
      exact rate is used.
    """
    if sampler_kind(cfg) == "csprng":
        return cfg.sample_rate
    exact = torch_inclusion_probability(cfg.sample_rate)
    up = float(exact)
    if Fraction(up) < exact:
        up = math.nextafter(up, math.inf)
    return min(up, 1.0)


class CSPRNGPoissonSampler(Sampler):
    """Poisson sampling driven by the OS CSPRNG (``secrets.SystemRandom``).

    Yields exactly ``steps`` batches. In each batch, record i is included iff
    ``getrandbits(53) < floor(q 2^53)``, using a fresh draw for every record and
    every step. So each record is included independently with probability
    floor(q 2^53)/2^53, which is <= q and within 2^-53 of it. The semantics
    match Opacus's ``UniformWithReplacementSampler``, but the randomness comes
    from the operating system instead of a seeded Mersenne Twister.
    """

    def __init__(self, *, num_samples: int, sample_rate: float, steps: int):
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        acc._validate_mechanism(sample_rate, 1.0, steps)
        self.num_samples, self.sample_rate, self.steps = num_samples, float(sample_rate), steps
        self._threshold = math.floor(Fraction(self.sample_rate) * 2 ** CSPRNG_BITS)
        self._rng = secrets.SystemRandom()

    def __len__(self):
        return self.steps

    def __iter__(self):
        bits, t, n = self._rng.getrandbits, self._threshold, self.num_samples
        for _ in range(self.steps):
            yield [i for i in range(n) if bits(CSPRNG_BITS) < t]


def make_poisson_loader(data: ClientDataset, cfg: DPConfig, generator: Optional[torch.Generator],
                        *, use_csprng: Optional[bool] = None) -> DataLoader:
    """Exactly ``K`` batches; each record is included in each batch independently with prob. ``q``.

    Empty batches are produced as zero-length tensors and still count as a
    (noise-only) step. With ``use_csprng`` (default: ``cfg.sampling_rng ==
    "csprng"``) the inclusion decisions come from the OS CSPRNG and
    ``generator`` is ignored.
    """
    use_csprng = (cfg.sampling_rng == "csprng") if use_csprng is None else use_csprng
    if use_csprng:
        sampler = CSPRNGPoissonSampler(num_samples=len(data), sample_rate=cfg.sample_rate,
                                       steps=cfg.local_steps)
    else:
        sampler = UniformWithReplacementSampler(
            num_samples=len(data), sample_rate=cfg.sample_rate, generator=generator, steps=cfg.local_steps)
    x0, y0 = data[0]  # shapes and dtypes only (public)
    return DataLoader(data, batch_sampler=sampler, collate_fn=wrap_collate_with_empty(
        collate_fn=default_collate, sample_empty_shapes=[(0, *x0.shape), (0, *y0.shape)],
        dtypes=[x0.dtype, y0.dtype]))


def make_dp_optimizer(gs_model: GradSampleModule, cfg: DPConfig, generator: torch.Generator) -> DPOptimizer:
    """Per-example clipping to ``C``, then ``N(0, (sigma C)^2)`` noise on the sum, then division by public ``B``."""
    return DPOptimizer(
        torch.optim.SGD(gs_model.parameters(), lr=cfg.lr),
        noise_multiplier=cfg.noise_multiplier, max_grad_norm=cfg.max_grad_norm,
        expected_batch_size=cfg.expected_batch_size, loss_reduction="mean",
        generator=generator, secure_mode=cfg.secure_noise)


# ------------------------------------------------------------------ local training
def local_dp_round(global_state: Dict[str, torch.Tensor], data: ClientDataset, cfg: DPConfig,
                   *, d: int, debug_seed: Optional[int] = None) -> Dict:
    """One client's local round: ``cfg.local_steps`` DP-SGD steps with Opacus.

    Returns the new model state plus client-side diagnostics. Only ``"state"``
    is released by :func:`run_federated`. ``"empty_batches"`` depends on the
    data and must never be released.
    """
    if not (isinstance(data, ClientDataset) and len(data) >= 1):
        raise ValueError("data must be a non-empty ClientDataset")
    if cfg.reproducible_noise:
        if debug_seed is None:
            raise ValueError("reproducible_noise=True requires debug_seed")
        warnings.warn(
            "reproducible_noise=True: DP noise and sampling are seeded with a known value. "
            "Anyone who knows the seed can remove the noise, so NO differential privacy "
            "guarantee holds. Use only for debugging and tests.", UserWarning, stacklevel=2)
        g_noise, g_sample = _generator(debug_seed), _generator(debug_seed + 1)
    else:
        g_noise, g_sample = _generator(None), _generator(None)

    model = make_model(d)
    model.load_state_dict(global_state)
    # loss_reduction="mean" here and in the optimizer: Opacus rescales the
    # per-example gradients of a mean loss back to true per-example gradients.
    gs_model = GradSampleModule(model, loss_reduction="mean")
    optimizer = make_dp_optimizer(gs_model, cfg, g_noise)
    # reproducible (debug) mode must be repeatable, so it uses the seeded torch sampler
    loader = make_poisson_loader(data, cfg, g_sample, use_csprng=(sampler_kind(cfg) == "csprng"))

    loss_fn = nn.BCEWithLogitsLoss()
    steps, empty = 0, 0
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        if len(xb) == 0:
            empty += 1
        loss = loss_fn(gs_model(xb).squeeze(-1), yb)
        loss.backward()
        optimizer.step()  # clip to C, add N(0, (sigma C)^2) to the sum, divide by B
        steps += 1
    if steps != cfg.local_steps:
        raise RuntimeError(f"took {steps} steps but the privacy map assumes {cfg.local_steps}")
    return {"state": get_state(gs_model._module), "steps": steps, "empty_batches": empty}


def fedavg(states: Sequence[Dict[str, torch.Tensor]], weights: Optional[Sequence[float]] = None):
    """Weighted average of model states (post-processing of the clients' DP releases)."""
    if not states:
        raise ValueError("no client states to aggregate")
    w = np.ones(len(states)) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(states) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative, non-zero, one per state")
    w = w / w.sum()
    return {k: sum(float(wi) * s[k] for wi, s in zip(w, states)) for k in states[0]}


@torch.no_grad()
def evaluate(state: Dict[str, torch.Tensor], X: np.ndarray, y: np.ndarray, d: int) -> Dict[str, float]:
    model = make_model(d)
    model.load_state_dict(state)
    logits = model(torch.from_numpy(X)).squeeze(-1)
    yt = torch.from_numpy(y)
    return {"accuracy": float(((logits > 0).float() == yt).float().mean()),
            "loss": float(nn.functional.binary_cross_entropy_with_logits(logits, yt))}


# --------------------------------------------------------------------- federation
@dataclass
class Client:
    data: ClientDataset
    odometer: acc.ClientPrivacyOdometer
    ledger: List[Dict] = field(default_factory=list)


def make_clients(client_sets: Sequence[ClientDataset], cfg: DPConfig) -> List[Client]:
    """One OpenDP odometer per client, held by that client.

    Refuses configurations that would break the privacy argument: empty clients
    (outside the input domain), duplicate client ids, and any record id held by
    more than one client (parallel composition needs disjoint clients).
    """
    if not client_sets:
        raise ValueError("at least one client is required")
    ids = [ds.client_id for ds in client_sets]
    if len(set(ids)) != len(ids):
        raise ValueError("client ids must be unique")
    seen: Dict[int, int] = {}
    for ds in client_sets:
        if len(ds) == 0:
            raise ValueError(f"client {ds.client_id} has no records (outside the input domain)")
        for rid in ds.record_ids.tolist():
            if rid in seen:
                raise ValueError(f"record {rid} is held by clients {seen[rid]} and {ds.client_id}; "
                                 "the per-client (max) privacy bound requires disjoint clients")
            seen[rid] = ds.client_id
    domain = client_domain()
    return [Client(ds, acc.ClientPrivacyOdometer(ds, domain, target_epsilon=cfg.target_epsilon, delta=cfg.delta))
            for ds in client_sets]


def _round_measurement(c: Client, state, cfg: DPConfig, d: int, seed: Optional[int]) -> dp.Measurement:
    frozen_state = copy.deepcopy(state)

    def function(data, _s=frozen_state, _seed=seed):
        # Release ONLY the model state. Diagnostics such as the number of empty
        # Poisson batches depend on the data, are not covered by the privacy
        # map, and never leave the client.
        return local_dp_round(_s, data, cfg, d=d, debug_seed=_seed)["state"]

    # account for the sampler local_dp_round will actually use
    return acc.make_dp_sgd_round(c.odometer.input_domain, function, sample_rate=accounted_sample_rate(cfg),
                                 noise_multiplier=cfg.noise_multiplier, steps=cfg.local_steps)


def run_federated(clients: Sequence[Client], init_state: Dict[str, torch.Tensor], cfg: DPConfig, d: int,
                  *, X_test=None, y_test=None, debug_seed: Optional[int] = None,
                  rounds: Optional[int] = None) -> Dict:
    """Run FedAvg rounds; every local round is executed through the client's OpenDP odometer.

    Before a round starts, every client's budget is checked, so a round either
    runs for all clients or for none.
    """
    if debug_seed is not None and not cfg.reproducible_noise:
        raise ValueError("debug_seed is only meaningful with reproducible_noise=True")
    state = init_state
    history = []
    for r in range(rounds or cfg.rounds):
        seeds = {c.data.client_id: None if debug_seed is None else debug_seed + 1000 * r + 10 * c.data.client_id
                 for c in clients}
        measurements = [_round_measurement(c, state, cfg, d, seeds[c.data.client_id]) for c in clients]
        for c, m in zip(clients, measurements):  # all-or-nothing budget check
            c.odometer.check_budget(m)
        updates = []
        for c, m in zip(clients, measurements):
            updates.append(c.odometer.run_round(m))
            eps, order = c.odometer.epsilon_spent()
            c.ledger.append({"round": len(c.ledger) + 1, "steps": cfg.local_steps,
                             "epsilon": eps, "best_order": order})
        state = fedavg(updates)  # uniform, data-independent weights
        row = {"round": r + 1, **{f"eps_client_{c.data.client_id}": c.ledger[-1]["epsilon"] for c in clients}}
        if X_test is not None:
            row.update(evaluate(state, X_test, y_test, d))
        history.append(row)
    return {"state": state, "history": history}
