"""Privacy accounting for the OpenDP + Opacus federated-learning example.

Division of responsibilities
----------------------------
* **Opacus** is the *mechanism*: it clips per-example gradients and adds
  Gaussian noise (see ``fl.local_dp_round``). Opacus's own accountant is NOT
  used for any reported guarantee; it appears only in tests as an independent
  numerical cross-check.
* **OpenDP** does the *accounting*: every local training round is wrapped as an
  OpenDP ``Measurement`` (``make_user_measurement``, output measure
  ``renyi_divergence``). Each client keeps its own OpenDP fully adaptive
  composition odometer (``make_fully_adaptive_composition``), and the training
  itself is executed *by submitting the measurement to that odometer*, so a
  round cannot run without being recorded.

What OpenDP 0.16 does not provide (and this module therefore supplies, under the
``honest-but-curious`` trust model):

1. The Rényi-DP curve of the Poisson-subsampled Gaussian mechanism
   (Mironov, Talwar & Zhang 2019, integer orders). ``sampled_gaussian_rdp``.
2. A RenyiDivergence -> approximate-DP measure cast. ``rdp_to_approx_dp``
   implements the conversion of Canonne, Kamath & Steinke (2020, Prop. 12),
   the same bound Opacus uses in ``get_privacy_spent``.
3. Budget enforcement on an RDP odometer. With OpenDP 0.16.0,
   ``make_privacy_filter(odometer, d_in=1, d_out=curve)`` over a
   ``renyi_divergence`` odometer aborted the interpreter (Rust panic
   ``unrecognized generic "Function"`` in ``src/data/ffi/mod.rs``), both when
   ``curve`` was a Python callable and when it was an OpenDP ``Function``
   returned by ``make_composition(...).map(1)``. The budget is therefore
   checked in Python *before* a round is submitted, using OpenDP's own
   ``make_composition`` to compose the prospective curve. This is enforcement
   by the calling code, not an OpenDP-enforced filter.
4. Fail-closed handling. In OpenDP 0.16.0 the fully adaptive odometer runs a
   measurement's function even if its privacy map raises; the error appears
   only when ``privacy_loss`` is queried. It also does not charge a round
   whose function raised. :class:`ClientPrivacyOdometer` therefore evaluates
   the prospective map before running, charges every attempted round, and
   halts the client after a failure.

Adjacency: add/remove one record within one client's local dataset
(``symmetric_distance``, ``d_in = 1``). Larger ``d_in`` is rejected rather than
handled through group privacy, to keep the example small and the claims narrow.
"""
from __future__ import annotations

import math
import warnings
from functools import lru_cache
from typing import Callable, List, Optional, Sequence, Tuple

import opendp.prelude as dp

#: Largest integer Rényi order evaluated. Orders above this return +inf, which
#: is always a valid (if useless) upper bound.
MAX_ORDER = 256
DEFAULT_ORDERS: Tuple[int, ...] = tuple(range(2, MAX_ORDER + 1))


def enable_opendp_features() -> None:
    """``make_user_measurement`` requires both features (OpenDP 0.16 docstring).

    ``honest-but-curious`` is required because OpenDP cannot verify that a
    user-supplied privacy map is correct for the user-supplied function; the
    correctness of :func:`sampled_gaussian_rdp` and of the Opacus training loop
    is our responsibility, not OpenDP's.
    """
    dp.enable_features("contrib", "honest-but-curious")


def _validate_mechanism(sample_rate: float, noise_multiplier: float, steps: int) -> None:
    if not (isinstance(sample_rate, (int, float)) and 0.0 < sample_rate <= 1.0):
        raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate!r}")
    if not (isinstance(noise_multiplier, (int, float)) and noise_multiplier > 0.0
            and math.isfinite(noise_multiplier)):
        raise ValueError(f"noise_multiplier must be a positive finite number, got {noise_multiplier!r}")
    if not (isinstance(steps, int) and not isinstance(steps, bool) and steps >= 1):
        raise ValueError(f"steps must be a positive integer, got {steps!r}")


@lru_cache(maxsize=65536)
def sampled_gaussian_rdp(sample_rate: float, noise_multiplier: float, order: int) -> float:
    """Per-step RDP of the Poisson-subsampled Gaussian mechanism at integer ``order``.

    Mironov, Talwar & Zhang (2019), "Rényi Differential Privacy of the Sampled
    Gaussian Mechanism", Sec. 3.3: for integer alpha >= 2 and sensitivity 1
    (clipping norm C, noise std sigma * C),

        A_alpha = sum_{k=0}^{alpha} C(alpha, k) (1-q)^(alpha-k) q^k exp((k^2 - k) / (2 sigma^2))
        RDP(alpha) = log(A_alpha) / (alpha - 1)

    This is exactly D_alpha(P || Q) for P = (1-q) N(0, sigma^2) + q N(1, sigma^2)
    and Q = N(0, sigma^2). MTZ19 show that this ordering dominates the reverse
    one, D_alpha(Q || P), so it bounds both the add and the remove direction.
    We rely on that result from the literature and do not re-prove it here. The
    tests check both directions numerically by quadrature at a few points.
    Computed in log space for numerical stability.
    """
    _validate_mechanism(sample_rate, noise_multiplier, 1)
    if not (isinstance(order, int) and order >= 2):
        raise ValueError(f"order must be an integer >= 2, got {order!r}")
    q, s2 = float(sample_rate), float(noise_multiplier) ** 2
    if q == 1.0:
        return order / (2.0 * s2)
    log_q, log_1mq = math.log(q), math.log1p(-q)
    log_a = -math.inf
    for k in range(order + 1):
        log_binom = math.lgamma(order + 1) - math.lgamma(k + 1) - math.lgamma(order - k + 1)
        term = log_binom + k * log_q + (order - k) * log_1mq + (k * k - k) / (2.0 * s2)
        log_a = _log_add(log_a, term)
    return log_a / (order - 1)


def _log_add(a: float, b: float) -> float:
    """log(exp(a) + exp(b)) without overflow."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def rdp_curve(sample_rate: float, noise_multiplier: float, steps: int) -> Callable[[float], float]:
    """RDP curve alpha -> epsilon(alpha) for ``steps`` compositions of the sampled Gaussian.

    Valid for every real alpha > 1: Rényi divergence is non-decreasing in alpha,
    so a non-integer alpha is bounded by the value at ceil(alpha); alpha above
    :data:`MAX_ORDER` returns +inf.
    """
    _validate_mechanism(sample_rate, noise_multiplier, steps)

    def curve(alpha: float) -> float:
        if not alpha > 1.0:
            raise ValueError(f"Rényi order must be > 1, got {alpha!r}")
        order = max(2, math.ceil(alpha))
        if order > MAX_ORDER:
            return math.inf
        return steps * sampled_gaussian_rdp(float(sample_rate), float(noise_multiplier), order)

    return curve


def rdp_to_approx_dp(curve: Callable[[float], float], delta: float,
                     orders: Sequence[int] = DEFAULT_ORDERS) -> Tuple[float, int]:
    """Convert an RDP curve to (epsilon, delta)-DP; returns ``(epsilon, best_order)``.

    Canonne, Kamath & Steinke (2020), Prop. 12 (also Balle et al. 2020, Thm. 21):
        eps = rdp(alpha) + log((alpha - 1) / alpha) - (log(delta) + log(alpha)) / (alpha - 1)
    minimised over the given integer orders.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta!r}")
    best_eps, best_order = math.inf, -1
    for a in orders:
        r = curve(float(a))
        if not math.isfinite(r):
            continue
        eps = r + math.log1p(-1.0 / a) - (math.log(delta) + math.log(a)) / (a - 1)
        if eps < best_eps:
            best_eps, best_order = eps, int(a)
    if best_order == max(orders):
        warnings.warn(f"best Rényi order is the largest one evaluated ({best_order}); "
                      "epsilon is valid but may be loose", RuntimeWarning, stacklevel=2)
    return max(best_eps, 0.0), best_order


def make_dp_sgd_round(input_domain: dp.Domain, function: Callable, *, sample_rate: float,
                      noise_multiplier: float, steps: int) -> dp.Measurement:
    """Wrap one client's local DP-SGD round as an OpenDP Measurement.

    ``function`` runs the Opacus training and returns the model update. The
    privacy map returns the RDP curve of ``steps`` sampled-Gaussian steps for
    ``d_in = 1`` (add/remove one record). It is only correct if ``function``
    really performs at most ``steps`` noisy steps with Poisson rate
    ``sample_rate`` and noise multiplier ``noise_multiplier``; the training code
    asserts this at run time.
    """
    curve = rdp_curve(sample_rate, noise_multiplier, steps)

    def privacy_map(d_in: int):
        if d_in == 0:
            return lambda alpha: 0.0
        if d_in != 1:
            raise ValueError("this example only supports d_in = 1 (add/remove one record)")
        return curve

    return dp.m.make_user_measurement(
        input_domain, dp.symmetric_distance(), dp.renyi_divergence(),
        function=function, privacy_map=privacy_map,
    )


class BudgetExceeded(RuntimeError):
    """Raised when running a round would exceed the client's (epsilon, delta) budget."""


class RoundFailed(RuntimeError):
    """Raised when a round's mechanism fails; the client is halted (fail closed)."""


class ClientPrivacyOdometer:
    """One client's privacy budget, held by the client (the data owner).

    Wraps an OpenDP fully adaptive composition odometer bound to the client's
    local dataset. Every training round must go through :meth:`run_round`.

    Fail-closed rules, motivated by observed OpenDP 0.16.0 behaviour:

    * The odometer runs a measurement's function even if its privacy map later
      raises; the error only surfaces when ``privacy_loss`` is queried. So the
      prospective privacy map and curve are evaluated *before* the round runs,
      and any error refuses the round.
    * The odometer does not charge a round whose function raised. This class
      charges every *attempted* round (a conservative over-count), halts the
      client, and refuses further rounds.

    The reported loss is the OpenDP ``make_composition`` of all attempted
    rounds. When no round failed, it equals the odometer's own
    ``privacy_loss``; :meth:`odometer_curve` exposes that for comparison.
    This is enforcement in Python by the calling code, not an OpenDP privacy
    filter: code holding the raw queryable could bypass it.
    """

    def __init__(self, client_data, input_domain: dp.Domain, *, target_epsilon: float, delta: float):
        if not (target_epsilon > 0 and math.isfinite(target_epsilon)):
            raise ValueError(f"target_epsilon must be positive and finite, got {target_epsilon!r}")
        if not (0.0 < delta < 1.0):
            raise ValueError(f"delta must be in (0, 1), got {delta!r}")
        self.input_domain = input_domain
        self.target_epsilon = float(target_epsilon)
        self.delta = float(delta)
        odometer = dp.c.make_fully_adaptive_composition(
            input_domain, dp.symmetric_distance(), dp.renyi_divergence())
        self._queryable = odometer(client_data)
        self._attempted: List[dp.Measurement] = []
        self._completed = 0
        self.halted = False

    @property
    def rounds_run(self) -> int:
        """Rounds charged to this client (attempted, including any that failed)."""
        return len(self._attempted)

    @property
    def rounds_completed(self) -> int:
        return self._completed

    @property
    def measurements(self) -> List[dp.Measurement]:
        """All attempted measurements (a copy)."""
        return list(self._attempted)

    def spent_curve(self) -> Callable[[float], float]:
        """RDP curve of all attempted rounds, composed by OpenDP ``make_composition``."""
        if not self._attempted:
            return lambda alpha: 0.0
        return dp.c.make_composition(list(self._attempted)).map(1)

    def odometer_curve(self) -> Callable[[float], float]:
        """The OpenDP odometer's own record (excludes rounds whose function raised)."""
        if not self._completed:
            return lambda alpha: 0.0
        return self._queryable.privacy_loss(1)

    def epsilon_spent(self) -> Tuple[float, int]:
        if not self._attempted:
            return 0.0, -1
        return rdp_to_approx_dp(self.spent_curve(), self.delta)

    def prospective_epsilon(self, measurement: dp.Measurement) -> float:
        """Epsilon if ``measurement`` were run next, composed natively by OpenDP.

        Evaluates the new privacy map and its curve at every order used, so a
        broken map raises here, before any data is touched.
        """
        composed = dp.c.make_composition(self._attempted + [measurement])
        return rdp_to_approx_dp(composed.map(1), self.delta)[0]

    def check_budget(self, measurement: dp.Measurement) -> float:
        """Raise if ``measurement`` would exceed the budget or cannot be accounted.

        Depends only on public parameters (the privacy maps), never on data, so
        refusing a round does not leak anything.
        """
        if self.halted:
            raise RoundFailed("client halted after a failed round; no further rounds allowed")
        eps_next = self.prospective_epsilon(measurement)
        if not math.isfinite(eps_next) or eps_next > self.target_epsilon:
            raise BudgetExceeded(
                f"round {self.rounds_run + 1} would raise epsilon to {eps_next:.4f} "
                f"> target {self.target_epsilon} at delta={self.delta}")
        return eps_next

    def run_round(self, measurement: dp.Measurement):
        """Check the budget, charge the round, then execute it through the OpenDP odometer."""
        self.check_budget(measurement)
        self._attempted.append(measurement)  # charged before running: failures are counted
        try:
            release = self._queryable(measurement)
        except Exception as exc:
            self.halted = True
            raise RoundFailed(f"round {self.rounds_run} failed and was charged; client halted") from exc
        self._completed += 1
        return release


def sequential_bound(measurements: Sequence[dp.Measurement], delta: float) -> float:
    """Epsilon if every measurement could touch the same record (OpenDP sequential composition).

    Valid without the disjointness assumption; conservative when clients hold
    disjoint records.
    """
    return rdp_to_approx_dp(dp.c.make_composition(list(measurements)).map(1), delta)[0]
