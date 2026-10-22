"""Tests for the OpenDP + Opacus federated-learning example.

Passing tests do not prove a differential-privacy guarantee. They check that
the implementation matches the documented assumptions, and they compare the
accounting with references computed independently: Opacus's RDP analysis,
numerical quadrature of the Rényi divergence, and Monte Carlo estimates of the
sampler and the noise. They do not merely re-check values the code produced.
"""
import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import opendp.prelude as dp  # noqa: E402
from opacus import GradSampleModule  # noqa: E402
from src import accounting as acc  # noqa: E402
from src import fl  # noqa: E402

acc.enable_opendp_features()
warnings.filterwarnings("ignore", message="Full backward hook")

D = 8


@pytest.fixture(scope="module")
def data():
    return fl.make_synthetic_data(n_train=900, n_test=300, d=D, seed=0)


def small_cfg(**kw):
    base = dict(sample_rate=0.05, noise_multiplier=1.0, max_grad_norm=1.0, expected_batch_size=15,
                local_steps=5, rounds=2, lr=0.5, target_epsilon=50.0, delta=1e-5)
    base.update(kw)
    return fl.DPConfig(**base)


def client_sets(X, y):
    return [fl.ClientDataset(X[p], y[p], k, record_ids=p) for k, p in enumerate(fl.partition_records(X))]


def build_clients(X, y, cfg):
    return fl.make_clients(client_sets(X, y), cfg)


def ds_from(X, y, cid=0):
    return fl.ClientDataset(X, y, cid, record_ids=np.arange(len(X)))


# ======================================================== 1. partition / neighbours
def test_partition_is_disjoint_and_covering(data):
    X = data[0]
    parts = fl.partition_records(X)
    allidx = np.concatenate(parts)
    assert len(allidx) == len(X) == len(np.unique(allidx))
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            assert np.intersect1d(a, b).size == 0


def test_adding_a_record_does_not_move_other_records(data):
    X = data[0]
    before = [fl.assign_client(x) for x in X]
    X2 = np.vstack([X, np.full((1, D), 10.0, dtype=np.float32)])
    after = [fl.assign_client(x) for x in X2]
    assert after[:-1] == before


def test_partition_is_non_iid(data):
    X = data[0]
    means = [X[p, 0].mean() for p in fl.partition_records(X)]
    assert means == sorted(means) and means[-1] - means[0] > 1.0


class _Spy:
    """Record the keyword arguments of the sampler (either kind) and optimizer actually constructed."""

    def __init__(self, monkeypatch):
        self.sampler, self.optimizer = [], []
        real_u, real_c, real_o = fl.UniformWithReplacementSampler, fl.CSPRNGPoissonSampler, fl.DPOptimizer

        def wrap(real):
            def s(*a, **kw):
                obj = real(*a, **kw)
                self.sampler.append((kw, obj))
                return obj
            return s

        def o(*a, **kw):
            obj = real_o(*a, **kw)
            self.optimizer.append((kw, obj))
            return obj

        monkeypatch.setattr(fl, "UniformWithReplacementSampler", wrap(real_u))
        monkeypatch.setattr(fl, "CSPRNGPoissonSampler", wrap(real_c))
        monkeypatch.setattr(fl, "DPOptimizer", o)


def test_neighbouring_datasets_run_identical_mechanisms(monkeypatch, data):
    """Add/remove one record: every mechanism parameter except the record set must be unchanged."""
    spy = _Spy(monkeypatch)
    # 110 vs 109 records: chosen so that round(q * n) would differ (6 vs 5)
    X, y = data[0][:110], data[1][:110]
    cfg = small_cfg()
    init = fl.get_state(fl.make_model(D))
    fl.local_dp_round(init, ds_from(X, y), cfg, d=D)          # D
    fl.local_dp_round(init, ds_from(X[1:], y[1:]), cfg, d=D)  # D' = D minus one record
    (s1, _), (s2, _) = spy.sampler
    (o1, _), (o2, _) = spy.optimizer
    for k in ("sample_rate", "steps"):
        assert s1[k] == s2[k]
    assert (s1["num_samples"], s2["num_samples"]) == (110, 109)  # only the record set differs
    for k in ("noise_multiplier", "max_grad_norm", "expected_batch_size", "secure_mode", "loss_reduction"):
        assert o1[k] == o2[k]


def test_normaliser_is_a_public_constant_not_the_dataset_size(monkeypatch, data):
    """Regression: B used to be round(q * n_k), a data-dependent quantity."""
    spy = _Spy(monkeypatch)
    X, y = data[0], data[1]
    cfg = small_cfg(rounds=1, expected_batch_size=17)
    clients = build_clients(X, y, cfg)
    assert len({len(c.data) for c in clients}) > 1  # clients really have different sizes
    fl.run_federated(clients, fl.get_state(fl.make_model(D)), cfg, D)
    bs = [kw["expected_batch_size"] for kw, _ in spy.optimizer]
    assert bs == [17] * len(clients)
    assert all(opt.expected_batch_size == 17 for _, opt in spy.optimizer)


def test_empty_dataset_is_outside_the_input_domain():
    empty = fl.ClientDataset(np.zeros((0, D), np.float32), np.zeros(0, np.float32), 0, record_ids=[])
    one = fl.ClientDataset(np.zeros((1, D), np.float32), np.zeros(1, np.float32), 0, record_ids=[0])
    dom = fl.client_domain()
    assert dom.member(one) and not dom.member(empty)
    with pytest.raises(ValueError):
        fl.local_dp_round(fl.get_state(fl.make_model(D)), empty, small_cfg(), d=D)


# ===================================================== 2. sampling semantics
@pytest.mark.parametrize("rng", ["csprng", "torch"])
def test_poisson_sampling_semantics_and_rate(rng):
    """Monte Carlo: per-record inclusion frequency ~ q; batch sizes vary like Binomial(n, q)."""
    n, q, steps = 200, 0.05, 4000
    cfg = small_cfg(sample_rate=q, local_steps=steps, sampling_rng=rng)
    ds = ds_from(np.zeros((n, D), np.float32), np.zeros(n, np.float32))
    g = torch.Generator().manual_seed(1)
    counts, sizes = np.zeros(n), []
    for idx in fl.make_poisson_loader(ds, cfg, g).batch_sampler:
        counts[idx] += 1
        sizes.append(len(idx))
    assert len(sizes) == steps
    freq = counts / steps
    assert abs(freq.mean() - q) < 0.002
    assert freq.min() > q - 6 * math.sqrt(q * (1 - q) / steps)  # no record systematically excluded
    assert np.var(sizes) == pytest.approx(n * q * (1 - q), rel=0.15)  # not a fixed batch size
    assert 0 in sizes or min(sizes) < n * q / 2


def test_accountant_uses_the_rate_and_steps_of_the_sampler_that_ran(monkeypatch, data):
    spy = _Spy(monkeypatch)
    X, y = data[0], data[1]
    cfg = small_cfg(rounds=2, sample_rate=0.03, local_steps=7, noise_multiplier=1.4)
    clients = build_clients(X, y, cfg)
    fl.run_federated(clients, fl.get_state(fl.make_model(D)), cfg, D)
    _, sampler = spy.sampler[0]
    _, opt = spy.optimizer[0]
    total_steps = sum(s.steps for _, s in spy.sampler[::len(clients)])  # client 0's samplers
    assert total_steps == cfg.local_steps * cfg.rounds
    reference = acc.rdp_curve(sampler.sample_rate, opt.noise_multiplier, total_steps)
    for a in (2.0, 7.0, 40.0):
        assert clients[0].odometer.spent_curve()(a) == pytest.approx(reference(a))


def test_local_round_takes_exactly_k_steps_including_empty_batches(data):
    X, y = data[0][:40], data[1][:40]
    cfg = small_cfg(sample_rate=0.01, local_steps=30, expected_batch_size=1, reproducible_noise=True)
    with pytest.warns(UserWarning, match="NO differential privacy"):
        out = fl.local_dp_round(fl.get_state(fl.make_model(D)), ds_from(X, y), cfg, d=D, debug_seed=3)
    assert out["steps"] == 30
    assert out["empty_batches"] > 0


def test_empty_batches_keep_weights_finite(data):
    cfg = small_cfg(sample_rate=0.01, local_steps=30, expected_batch_size=1)
    out = fl.local_dp_round(fl.get_state(fl.make_model(D)), ds_from(data[0][:40], data[1][:40]), cfg, d=D)
    assert all(torch.isfinite(v).all() for v in out["state"].values())


# ===================================================== 3. RDP math (independent references)
@pytest.mark.parametrize("q,sigma,steps", [(0.05, 1.0, 100), (0.01, 0.8, 1000), (0.2, 2.0, 50), (1.0, 1.5, 10)])
def test_rdp_matches_independent_opacus_analysis(q, sigma, steps):
    from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
    orders = list(range(2, acc.MAX_ORDER + 1))
    ours = [acc.rdp_curve(q, sigma, steps)(a) for a in orders]
    ref = compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=orders)
    for a, b in zip(ours, ref):
        if math.isfinite(b):
            assert a == pytest.approx(b, rel=1e-9)
    eps, _ = acc.rdp_to_approx_dp(acc.rdp_curve(q, sigma, steps), 1e-5)
    ref_eps, _ = get_privacy_spent(orders=orders, rdp=ref, delta=1e-5)
    assert eps == pytest.approx(float(ref_eps), rel=1e-9)


@pytest.mark.parametrize("q,sigma", [(0.05, 1.0), (0.2, 0.8), (0.5, 2.0), (0.01, 0.7)])
@pytest.mark.parametrize("alpha", [2, 3, 5])
def test_rdp_matches_numerical_renyi_divergence_both_directions(q, sigma, alpha):
    """Quadrature of the Rényi divergence between N(0, s^2) and the q-mixture, both directions."""
    from scipy import integrate, stats

    def renyi(logp, logq):
        f = lambda x: math.exp(alpha * logp(x) + (1 - alpha) * logq(x))  # noqa: E731
        v, _ = integrate.quad(f, -12 * sigma, 1 + 12 * sigma, points=[0, 1], limit=500, epsabs=0, epsrel=1e-11)
        return math.log(v) / (alpha - 1)

    lq = lambda x: stats.norm.logpdf(x, 0, sigma)  # noqa: E731
    lp = lambda x: np.logaddexp(math.log1p(-q) + stats.norm.logpdf(x, 0, sigma),  # noqa: E731
                                math.log(q) + stats.norm.logpdf(x, 1, sigma))
    ours = acc.sampled_gaussian_rdp(q, sigma, alpha)
    assert ours == pytest.approx(renyi(lp, lq), rel=1e-6)     # D(P||Q): exact
    assert renyi(lq, lp) <= ours * (1 + 1e-9)                 # D(Q||P): dominated


def test_full_batch_gaussian_closed_form():
    for a in (2, 7, 32):
        assert acc.sampled_gaussian_rdp(1.0, 2.0, a) == pytest.approx(a / 8.0)


def test_subsampling_reduces_privacy_loss():
    assert acc.sampled_gaussian_rdp(0.05, 1.0, 8) < acc.sampled_gaussian_rdp(1.0, 1.0, 8)


def test_curve_is_monotone_and_handles_fractional_orders():
    c = acc.rdp_curve(0.05, 1.0, 10)
    vals = [c(a) for a in (1.5, 2, 2.5, 3, 10, 100)]
    assert vals == sorted(vals)
    assert c(2.5) == c(3)
    assert c(acc.MAX_ORDER + 1) == math.inf
    with pytest.raises(ValueError):
        c(1.0)


def test_rdp_to_approx_dp_against_direct_formula():
    """Recompute the CKS20/BBGHS20 bound by brute force over orders, independently."""
    c = acc.rdp_curve(0.05, 1.0, 100)
    delta = 1e-5
    direct = min(c(a) + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1) for a in range(2, 257))
    assert acc.rdp_to_approx_dp(c, delta)[0] == pytest.approx(direct)
    assert acc.rdp_to_approx_dp(c, 1e-3)[0] < acc.rdp_to_approx_dp(c, 1e-7)[0]  # larger delta, smaller eps


def test_step_and_round_composition_is_additive():
    per_step = [acc.sampled_gaussian_rdp(0.05, 1.0, a) for a in (2, 5, 9)]
    K, R = 20, 5
    dom = fl.client_domain()
    rounds = [acc.make_dp_sgd_round(dom, lambda x: 0, sample_rate=0.05, noise_multiplier=1.0, steps=K)
              for _ in range(R)]
    composed = dp.c.make_composition(rounds).map(1)
    for a, ps in zip((2.0, 5.0, 9.0), per_step):
        assert acc.rdp_curve(0.05, 1.0, K)(a) == pytest.approx(K * ps)
        assert composed(a) == pytest.approx(R * K * ps)


@pytest.mark.parametrize("bad", [dict(sample_rate=0.0), dict(sample_rate=1.5), dict(noise_multiplier=0.0),
                                 dict(noise_multiplier=float("inf")), dict(steps=0), dict(steps=2.0)])
def test_invalid_mechanism_parameters_rejected(bad):
    args = dict(sample_rate=0.05, noise_multiplier=1.0, steps=10)
    args.update(bad)
    with pytest.raises(ValueError):
        acc.rdp_curve(**args)


@pytest.mark.parametrize("bad", [dict(sample_rate=0), dict(noise_multiplier=-1), dict(max_grad_norm=0),
                                 dict(local_steps=0), dict(rounds=0), dict(delta=1.0), dict(target_epsilon=0),
                                 dict(target_epsilon=float("inf")), dict(expected_batch_size=0),
                                 dict(expected_batch_size=12.5), dict(lr=float("nan"))])
def test_invalid_config_rejected(bad):
    with pytest.raises(ValueError):
        small_cfg(**bad)


def test_invalid_delta_rejected_by_conversion():
    with pytest.raises(ValueError):
        acc.rdp_to_approx_dp(acc.rdp_curve(0.05, 1.0, 10), 0.0)


# ===================================================== 4. OpenDP wiring
def test_measurement_uses_opendp_renyi_divergence_and_rejects_d_in_above_one():
    m = acc.make_dp_sgd_round(fl.client_domain(), lambda x: 0, sample_rate=0.05, noise_multiplier=1.0, steps=10)
    assert "RenyiDivergence" in str(m.output_measure)
    assert "SymmetricDistance" in str(m.input_metric)
    assert m.map(1)(4.0) == pytest.approx(acc.rdp_curve(0.05, 1.0, 10)(4.0))
    assert m.map(0)(4.0) == 0.0
    with pytest.raises(Exception):
        m.map(2)


def test_opendp_composition_adds_rdp_curves():
    dom = fl.client_domain()
    ms = [acc.make_dp_sgd_round(dom, lambda x: 0, sample_rate=0.05, noise_multiplier=1.0, steps=10) for _ in range(3)]
    composed = dp.c.make_composition(ms).map(1)
    for a in (2.0, 5.0, 20.0):
        assert composed(a) == pytest.approx(3 * acc.rdp_curve(0.05, 1.0, 10)(a))


def test_opendp_requires_features_in_a_fresh_interpreter():
    """Without contrib + honest-but-curious, OpenDP must refuse the user measurement."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from src import accounting as acc, fl\n"
        "try:\n"
        "    acc.make_dp_sgd_round(fl.client_domain(), lambda x: 0, sample_rate=0.05, noise_multiplier=1.0, steps=1)\n"
        "    print('ACCEPTED')\n"
        "except Exception as e:\n"
        "    print('REFUSED', type(e).__name__)\n"
    ) % str(ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "REFUSED" in out.stdout, out.stdout + out.stderr


def test_opendp_api_surface_used_by_the_example():
    from importlib.metadata import version
    assert version("opendp").startswith("0.16"), "example is validated against opendp 0.16.x only"
    for name in ("renyi_divergence", "symmetric_distance", "user_domain"):
        assert hasattr(dp, name)
    assert hasattr(dp.m, "make_user_measurement")
    assert hasattr(dp.c, "make_fully_adaptive_composition") and hasattr(dp.c, "make_composition")
    assert hasattr(dp.OdometerQueryable, "privacy_loss")


# ===================================================== 5. Opacus mechanism
def test_per_example_clipping_bounds_the_noiseless_update(data):
    X, y = data[0][:200] * 50, data[1][:200]  # large inputs -> large raw gradients
    cfg = fl.DPConfig(sample_rate=1.0, noise_multiplier=1e-9, max_grad_norm=0.1, expected_batch_size=200,
                      local_steps=1, rounds=1, lr=1.0, target_epsilon=1e12, delta=1e-5, reproducible_noise=True)
    init = fl.get_state(fl.make_model(D))
    with pytest.warns(UserWarning):
        out = fl.local_dp_round(init, ds_from(X.astype(np.float32), y), cfg, d=D, debug_seed=0)
    delta_norm = torch.sqrt(sum(((out["state"][k] - init[k]) ** 2).sum() for k in init))
    assert float(delta_norm) <= 0.1 * 1.0 + 1e-6  # 200 * C * lr / B with B = 200


@pytest.mark.parametrize("secure", [True, False])
def test_empirical_noise_std_is_sigma_c_over_b(secure):
    """Zero gradients: the update is pure noise with std lr * sigma * C / B (Monte Carlo)."""
    sigma, C, B, lr = 1.3, 0.7, 10, 1.0
    cfg = small_cfg(noise_multiplier=sigma, max_grad_norm=C, expected_batch_size=B, lr=lr, secure_noise=secure)
    model = nn.Linear(20000, 1, bias=False)
    nn.init.zeros_(model.weight)
    gs = GradSampleModule(model, loss_reduction="mean")
    opt = fl.make_dp_optimizer(gs, cfg, torch.Generator().manual_seed(0))
    xb = torch.zeros(4, 20000)  # zero inputs -> zero per-example gradients
    opt.zero_grad(set_to_none=True)
    gs(xb).sum().backward()
    opt.step()
    std = float(model.weight.detach().std())
    assert std == pytest.approx(lr * sigma * C / B, rel=0.03)


def test_noise_clipping_and_secure_mode_are_passed_to_opacus(monkeypatch, data):
    spy = _Spy(monkeypatch)
    cfg = small_cfg(noise_multiplier=1.3, max_grad_norm=0.7, expected_batch_size=5)
    fl.local_dp_round(fl.get_state(fl.make_model(D)), ds_from(data[0][:100], data[1][:100]), cfg, d=D)
    kw, opt = spy.optimizer[0]
    assert kw["noise_multiplier"] == 1.3 and kw["max_grad_norm"] == 0.7
    assert kw["expected_batch_size"] == 5 and kw["secure_mode"] is True and kw["loss_reduction"] == "mean"
    assert opt.max_grad_norm == 0.7  # fixed: no clipping scheduler attached


def test_reproducible_mode_is_deterministic_and_default_mode_is_not(data):
    ds = ds_from(data[0][:200], data[1][:200])
    init = fl.get_state(fl.make_model(D))
    rep = small_cfg(reproducible_noise=True)
    with pytest.warns(UserWarning):
        a = fl.local_dp_round(init, ds, rep, d=D, debug_seed=7)["state"]
        b = fl.local_dp_round(init, ds, rep, d=D, debug_seed=7)["state"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    c = fl.local_dp_round(init, ds, small_cfg(), d=D, debug_seed=7)["state"]  # seed ignored in private mode
    e = fl.local_dp_round(init, ds, small_cfg(), d=D, debug_seed=7)["state"]
    assert not all(torch.equal(c[k], e[k]) for k in c)


def test_reproducible_mode_requires_explicit_seed(data):
    with pytest.raises(ValueError):
        fl.local_dp_round(fl.get_state(fl.make_model(D)), ds_from(data[0][:50], data[1][:50]),
                          small_cfg(reproducible_noise=True), d=D)


def test_debug_seed_refused_in_private_federated_mode(data):
    cfg = small_cfg(rounds=1)
    with pytest.raises(ValueError):
        fl.run_federated(build_clients(data[0], data[1], cfg), fl.get_state(fl.make_model(D)), cfg, D,
                         debug_seed=1)


def test_private_mode_draws_fresh_secret_seeds_for_every_client_and_round(monkeypatch, data):
    drawn = []
    real = fl.secrets.randbits

    def spy(k):
        v = real(k)
        drawn.append(v)
        return v

    monkeypatch.setattr(fl.secrets, "randbits", spy)
    cfg = small_cfg(rounds=2)
    clients = build_clients(data[0], data[1], cfg)
    fl.run_federated(clients, fl.get_state(fl.make_model(D)), cfg, D)
    assert len(drawn) == 2 * len(clients) * cfg.rounds  # noise + sampling, per client per round
    assert len(set(drawn)) == len(drawn)                 # no reuse


def test_model_construction_does_not_reseed_global_rng():
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    fl.make_model(D, seed=0)
    assert torch.equal(torch.rand(3), expected)


# ===================================================== 6. federation, budget, clients
def test_rounds_are_accounted_through_the_odometer(data):
    cfg = small_cfg(rounds=3)
    clients = build_clients(data[0], data[1], cfg)
    fl.run_federated(clients, fl.get_state(fl.make_model(D)), cfg, D)
    expected = acc.rdp_curve(cfg.sample_rate, cfg.noise_multiplier, cfg.local_steps * cfg.rounds)
    for c in clients:
        assert c.odometer.rounds_run == 3
        assert c.odometer.spent_curve()(6.0) == pytest.approx(expected(6.0))
        assert [row["steps"] for row in c.ledger] == [cfg.local_steps] * 3
        eps = [row["epsilon"] for row in c.ledger]
        assert eps == sorted(eps)
        assert eps[-1] == pytest.approx(acc.rdp_to_approx_dp(expected, cfg.delta)[0])


def test_budget_exceeded_stops_the_whole_round_before_training(data, monkeypatch):
    one_round = acc.rdp_to_approx_dp(acc.rdp_curve(0.05, 1.0, 5), 1e-5)[0]
    two_rounds = acc.rdp_to_approx_dp(acc.rdp_curve(0.05, 1.0, 10), 1e-5)[0]
    cfg = small_cfg(target_epsilon=(one_round + two_rounds) / 2, rounds=2)
    clients = build_clients(data[0], data[1], cfg)
    calls = {"n": 0}
    real = fl.local_dp_round

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(fl, "local_dp_round", counting)
    with pytest.raises(acc.BudgetExceeded):
        fl.run_federated(clients, fl.get_state(fl.make_model(D)), cfg, D)
    assert calls["n"] == len(clients)  # round 1 ran for every client; round 2 for none
    assert [c.odometer.rounds_run for c in clients] == [1] * len(clients)
    assert all(c.odometer.epsilon_spent()[0] <= cfg.target_epsilon for c in clients)


def test_parallel_vs_sequential_bound(data):
    cfg = small_cfg(rounds=1)
    clients = build_clients(data[0], data[1], cfg)
    fl.run_federated(clients, fl.get_state(fl.make_model(D)), cfg, D)
    per_client = [c.odometer.epsilon_spent()[0] for c in clients]
    seq = acc.sequential_bound([m for c in clients for m in c.odometer.measurements], cfg.delta)
    assert max(per_client) < seq


def test_a_clients_epsilon_does_not_depend_on_other_clients(data):
    cfg = small_cfg(rounds=2)
    sets = client_sets(data[0], data[1])
    alone = fl.make_clients(sets[:1], cfg)
    together = fl.make_clients(sets, cfg)
    fl.run_federated(alone, fl.get_state(fl.make_model(D)), cfg, D)
    fl.run_federated(together, fl.get_state(fl.make_model(D)), cfg, D)
    assert alone[0].odometer.epsilon_spent()[0] == pytest.approx(together[0].odometer.epsilon_spent()[0])


def test_released_value_is_only_the_model_state(data):
    cfg = small_cfg(rounds=1)
    clients = build_clients(data[0], data[1], cfg)
    seen = []
    real = clients[0].odometer.run_round

    def spy(meas):
        out = real(meas)
        seen.append(out)
        return out

    clients[0].odometer.run_round = spy
    fl.run_federated(clients, fl.get_state(fl.make_model(D)), cfg, D)
    assert set(seen[0].keys()) == set(fl.get_state(fl.make_model(D)).keys())


def test_shared_record_across_clients_is_refused(data):
    X, y = data[0], data[1]
    a = fl.ClientDataset(X[:10], y[:10], 0, record_ids=np.arange(10))
    b = fl.ClientDataset(X[5:15], y[5:15], 1, record_ids=np.arange(5, 15))
    with pytest.raises(ValueError, match="disjoint"):
        fl.make_clients([a, b], small_cfg())


@pytest.mark.parametrize("case", ["empty", "dup_client", "dup_record", "len_mismatch", "none"])
def test_misconfigured_clients_rejected(case, data):
    X, y = data[0][:10], data[1][:10]
    cfg = small_cfg()
    with pytest.raises(ValueError):
        if case == "empty":
            fl.make_clients([fl.ClientDataset(X[:0], y[:0], 0, record_ids=[])], cfg)
        elif case == "dup_client":
            fl.make_clients([fl.ClientDataset(X[:5], y[:5], 0, record_ids=range(5)),
                             fl.ClientDataset(X[5:], y[5:], 0, record_ids=range(5, 10))], cfg)
        elif case == "dup_record":
            fl.ClientDataset(X[:2], y[:2], 0, record_ids=[1, 1])
        elif case == "len_mismatch":
            fl.ClientDataset(X, y, 0, record_ids=range(3))
        else:
            fl.make_clients([], cfg)


def test_fedavg_validation():
    s = {"w": torch.ones(2)}
    with pytest.raises(ValueError):
        fl.fedavg([])
    with pytest.raises(ValueError):
        fl.fedavg([s, s], weights=[1.0])
    assert torch.allclose(fl.fedavg([s, {"w": torch.zeros(2)}])["w"], torch.full((2,), 0.5))


def test_end_to_end_training_learns_and_reports_epsilon(data):
    X, y, Xt, yt = fl.make_synthetic_data(n_train=1500, n_test=500, d=D, seed=0)
    cfg = fl.DPConfig(rounds=3, local_steps=20, expected_batch_size=25)
    clients = build_clients(X, y, cfg)
    init = fl.get_state(fl.make_model(D))
    before = fl.evaluate(init, Xt, yt, D)["accuracy"]
    out = fl.run_federated(clients, init, cfg, D, X_test=Xt, y_test=yt)
    assert out["history"][-1]["accuracy"] > before + 0.1
    expected = acc.rdp_to_approx_dp(acc.rdp_curve(cfg.sample_rate, cfg.noise_multiplier, 60), cfg.delta)[0]
    assert all(c.odometer.epsilon_spent()[0] == pytest.approx(expected) for c in clients)


# ===================================================== 7. added in the final-validation pass
def test_independent_google_dp_accounting_rdp_and_pld():
    """Compare with Google's dp_accounting (independent of Opacus and of this code)."""
    dpa = pytest.importorskip("dp_accounting")
    q, sigma, T, delta = 0.05, 1.0, 100, 1e-5
    event = dpa.SelfComposedDpEvent(dpa.PoissonSampledDpEvent(q, dpa.GaussianDpEvent(sigma)), T)
    rdp = dpa.rdp.RdpAccountant(orders=[float(a) for a in range(2, acc.MAX_ORDER + 1)])
    rdp.compose(event)
    ours = acc.rdp_to_approx_dp(acc.rdp_curve(q, sigma, T), delta)[0]
    assert ours == pytest.approx(rdp.get_epsilon(delta), rel=1e-6)   # same orders, independent code
    pld = dpa.pld.PLDAccountant()
    pld.compose(event)
    assert pld.get_epsilon(delta) < ours  # tighter accountant gives smaller epsilon: RDP is conservative


def test_broken_privacy_map_is_refused_before_any_training():
    """OpenDP 0.16's odometer runs the function before the map is evaluated; we must not."""
    import numpy as np  # noqa: F811
    dom = fl.client_domain()
    ds = ds_from(np.zeros((3, D), np.float32), np.zeros(3, np.float32))
    odo = acc.ClientPrivacyOdometer(ds, dom, target_epsilon=10.0, delta=1e-5)
    ran = {"n": 0}

    def function(x):
        ran["n"] += 1
        return 0

    def broken_map(d_in):
        raise ValueError("map cannot be evaluated")

    meas = dp.m.make_user_measurement(dom, dp.symmetric_distance(), dp.renyi_divergence(),
                                      function=function, privacy_map=broken_map)
    with pytest.raises(Exception):
        odo.run_round(meas)
    assert ran["n"] == 0 and odo.rounds_run == 0


def test_characterise_raw_odometer_runs_function_despite_broken_map():
    """Documents upstream behaviour that motivates the pre-check (not a property of this code)."""
    dom = dp.user_domain("D", lambda x: True)
    ran = {"n": 0}

    def function(x):
        ran["n"] += 1
        return 1.0

    meas = dp.m.make_user_measurement(dom, dp.symmetric_distance(), dp.renyi_divergence(), function=function,
                                      privacy_map=lambda d: (_ for _ in ()).throw(ValueError("broken")))
    q = dp.c.make_fully_adaptive_composition(dom, dp.symmetric_distance(), dp.renyi_divergence())("x")
    q(meas)
    assert ran["n"] == 1  # the release happened
    with pytest.raises(Exception):
        q.privacy_loss(1)


def test_failed_round_is_charged_and_halts_the_client(data, monkeypatch):
    cfg = small_cfg(rounds=3)
    clients = build_clients(data[0], data[1], cfg)
    init = fl.get_state(fl.make_model(D))
    fl.run_federated(clients, init, cfg, D, rounds=1)
    real = fl.local_dp_round
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:  # second client of round 2 fails
            raise RuntimeError("simulated crash")
        return real(*a, **kw)

    monkeypatch.setattr(fl, "local_dp_round", flaky)
    with pytest.raises(acc.RoundFailed):
        fl.run_federated(clients, init, cfg, D, rounds=1)
    c1 = clients[1]
    assert c1.odometer.halted and c1.odometer.rounds_run == 2 and c1.odometer.rounds_completed == 1
    # charged conservatively: reported loss covers both attempted rounds ...
    two = acc.rdp_curve(cfg.sample_rate, cfg.noise_multiplier, 2 * cfg.local_steps)
    assert c1.odometer.spent_curve()(5.0) == pytest.approx(two(5.0))
    # ... while the raw odometer only charged the completed one
    assert c1.odometer.odometer_curve()(5.0) < c1.odometer.spent_curve()(5.0)
    monkeypatch.setattr(fl, "local_dp_round", real)
    with pytest.raises(acc.RoundFailed):
        fl.run_federated(clients, init, cfg, D, rounds=1)  # halted client refuses further rounds


def test_odometer_and_reported_loss_agree_without_failures(data):
    cfg = small_cfg(rounds=2)
    clients = build_clients(data[0], data[1], cfg)
    fl.run_federated(clients, fl.get_state(fl.make_model(D)), cfg, D)
    for c in clients:
        for a in (2.0, 6.0, 30.0):
            assert c.odometer.odometer_curve()(a) == pytest.approx(c.odometer.spent_curve()(a))


def test_csprng_sampler_draws_from_the_os_csprng(monkeypatch):
    calls = {"n": 0}
    real_cls = fl.secrets.SystemRandom

    class Counting(real_cls):
        def getrandbits(self, k):
            assert k == fl.CSPRNG_BITS
            calls["n"] += 1
            return super().getrandbits(k)

    monkeypatch.setattr(fl.secrets, "SystemRandom", Counting)
    s = fl.CSPRNGPoissonSampler(num_samples=50, sample_rate=0.1, steps=7)
    batches = list(s)
    assert len(batches) == 7 and calls["n"] == 50 * 7  # one fresh OS-CSPRNG draw per record per step
    assert all(len(set(b)) == len(b) and all(0 <= i < 50 for i in b) for b in batches)


def test_reproducible_mode_uses_seeded_sampler_and_private_mode_uses_csprng(monkeypatch, data):
    spy = _Spy(monkeypatch)
    ds = ds_from(data[0][:60], data[1][:60])
    init = fl.get_state(fl.make_model(D))
    fl.local_dp_round(init, ds, small_cfg(), d=D)
    with pytest.warns(UserWarning):
        fl.local_dp_round(init, ds, small_cfg(reproducible_noise=True), d=D, debug_seed=1)
    kinds = [type(obj).__name__ for _, obj in spy.sampler]
    assert kinds == ["CSPRNGPoissonSampler", "UniformWithReplacementSampler"]


def test_noise_std_on_an_empty_batch_is_still_sigma_c_over_b():
    """An empty Poisson batch must still add N(0, (sigma C)^2) / B, not skip or divide by zero."""
    sigma, C, B, lr = 0.9, 1.1, 7, 1.0
    cfg = small_cfg(noise_multiplier=sigma, max_grad_norm=C, expected_batch_size=B, lr=lr)
    model = nn.Linear(20000, 1, bias=False)
    nn.init.zeros_(model.weight)
    gs = GradSampleModule(model, loss_reduction="mean")
    opt = fl.make_dp_optimizer(gs, cfg, torch.Generator().manual_seed(0))
    opt.zero_grad(set_to_none=True)
    gs(torch.zeros(0, 20000)).sum().backward()  # empty batch
    opt.step()
    assert float(model.weight.detach().std()) == pytest.approx(lr * sigma * C / B, rel=0.03)


def test_two_clients_with_identical_inputs_get_independent_noise(data):
    ds = ds_from(data[0][:100], data[1][:100])
    init = fl.get_state(fl.make_model(D))
    a = fl.local_dp_round(init, ds, small_cfg(), d=D)["state"]
    b = fl.local_dp_round(init, ds, small_cfg(), d=D)["state"]
    diff = torch.cat([(a[k] - b[k]).flatten() for k in a])
    assert float(diff.abs().max()) > 0  # no shared noise stream


def test_boundary_renyi_order_warns():
    with pytest.warns(RuntimeWarning, match="largest one evaluated"):
        acc.rdp_to_approx_dp(acc.rdp_curve(0.001, 50.0, 1), 1e-5)


def test_invalid_sampling_rng_rejected():
    with pytest.raises(ValueError):
        small_cfg(sampling_rng="mt19937")


def test_exact_inclusion_probabilities_versus_accounted_rate():
    """The accountant's q must upper-bound the sampler's true inclusion probability."""
    from fractions import Fraction
    for q in (0.05, 0.01, 0.3, 1 / 3, 1.0):
        assert fl.csprng_inclusion_probability(q) <= Fraction(q)          # CSPRNG rule never exceeds q
        assert Fraction(q) - fl.csprng_inclusion_probability(q) < Fraction(1, 2 ** 53)
        t = fl.accounted_sample_rate(small_cfg(sample_rate=q, sampling_rng="torch"))
        assert Fraction(t) >= fl.torch_inclusion_probability(q) >= Fraction(q)
    # concrete case: torch.rand < 0.05 includes with prob 838861 / 2^24 > 0.05
    assert fl.torch_inclusion_probability(0.05) == Fraction(838861, 2 ** 24)
    assert fl.accounted_sample_rate(small_cfg(sampling_rng="torch")) > 0.05
    assert fl.accounted_sample_rate(small_cfg()) == 0.05


def test_torch_rand_grid_assumption_holds_for_installed_torch():
    """torch.rand(float32) values are multiples of 2^-24; the torch-mode rate correction relies on this."""
    u = torch.rand(200000, generator=torch.Generator().manual_seed(0)).double() * 2 ** fl.TORCH_RAND_BITS
    assert torch.all(u == u.round())


def test_csprng_threshold_rule_is_exact():
    s = fl.CSPRNGPoissonSampler(num_samples=5, sample_rate=0.05, steps=1)
    from fractions import Fraction
    assert Fraction(s._threshold, 2 ** fl.CSPRNG_BITS) == fl.csprng_inclusion_probability(0.05)
    assert fl.CSPRNGPoissonSampler(num_samples=5, sample_rate=1.0, steps=3).__iter__().__next__() == list(range(5))


def test_reproducible_mode_accounts_with_the_torch_rate(data):
    cfg = small_cfg(rounds=1, reproducible_noise=True)
    clients = build_clients(data[0], data[1], cfg)
    with pytest.warns(UserWarning):
        fl.run_federated(clients, fl.get_state(fl.make_model(D)), cfg, D, debug_seed=5)
    q_acc = fl.accounted_sample_rate(cfg)
    assert q_acc > cfg.sample_rate
    expected = acc.rdp_curve(q_acc, cfg.noise_multiplier, cfg.local_steps)
    assert clients[0].odometer.spent_curve()(4.0) == pytest.approx(expected(4.0))
