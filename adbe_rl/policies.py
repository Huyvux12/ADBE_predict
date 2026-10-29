"""Policies: CEM target-exposure policies and a rule-based baseline.

A CEM policy outputs a logit for the desired stock exposure. `decide` turns it
into one of the 7 discrete actions: pick the legal order whose post-trade
exposure is closest to the target, skip trades whose improvement is below a
learned hysteresis, and before the idle penalty place the cheapest order if its
fee is below a learned fraction of the penalty.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .broker import order_quantities
from .config import Rules
from .env import Account
from .market import N_FEATURES, Market


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -35, 35)))


def decide(logits, controls, cash, shares, idle, price, rules: Rules, mask=None) -> np.ndarray:
    """Vectorized action choice for n portfolios; controls[:, 0:2] = hysteresis, threshold."""
    target = sigmoid(logits)
    hysteresis = .005 + .20 * sigmoid(controls[:, 0])
    threshold = .25 + 2.75 * sigmoid(controls[:, 1])
    q = order_quantities(cash, shares, price, rules)
    fee = np.abs(q) * price * rules.fee_rate
    cash_after = cash[:, None] - q * price - fee
    shares_after = shares[:, None] + q
    assets_after = np.maximum(cash_after + shares_after * price, 1e-8)
    distance = np.abs(shares_after * price / assets_after - target[:, None])

    invalid = q == 0
    invalid[:, 0] = False
    if mask is not None:
        invalid |= ~np.broadcast_to(mask, invalid.shape)
    distance = np.where(invalid, np.inf, distance)
    rows = np.arange(len(distance))
    best = np.argmin(distance, axis=1)
    action = np.where(distance[:, 0] - distance[rows, best] > hysteresis, best, 0)

    cost = np.where(invalid, np.inf, fee)
    cost[:, 0] = np.inf
    cheapest = np.argmin(cost, axis=1)
    dodge = ((idle >= rules.idle_limit - 1) & (action == 0)
             & (cost[rows, cheapest] < rules.idle_penalty * threshold))
    return np.where(dodge, cheapest, action)


class CEMPolicy:
    """Base for policies parameterized by one vector theta = [logit params..., 2 controls]."""
    kind = ''

    def __init__(self, theta, rules: Rules):
        self.theta = np.asarray(theta, float)
        self.rules = rules

    def logits(self, thetas: np.ndarray, features: np.ndarray, year: float) -> np.ndarray:
        """Exposure logits, shape (n,), from yesterday's features and the current date."""
        raise NotImplementedError

    def predict(self, obs, account: Account, action_mask=None) -> int:
        thetas = self.theta[None, :]
        logit = self.logits(thetas, obs[None, :N_FEATURES], float(obs[-1]))
        action = decide(logit, thetas[:, -2:], np.array([account.cash]), np.array([account.shares]),
                        np.array([account.idle]), account.price, self.rules, action_mask)
        return int(action[0])

    def population(self, thetas: np.ndarray, market: Market) -> Population:
        return Population(self, np.atleast_2d(thetas), market)

    def to_dict(self) -> dict:
        return {'policy': self.kind, 'theta': self.theta.tolist()}


class IndicatorPolicy(CEMPolicy):
    """Exposure logit = w · features + b (12 weights, bias, 2 controls = 15 parameters)."""
    kind = 'indicator'
    size = N_FEATURES + 3

    def logits(self, thetas, features, year):
        return np.sum(features * thetas[:, :N_FEATURES], axis=1) + thetas[:, N_FEATURES]


class TemporalPolicy(CEMPolicy):
    """Exposure logit interpolated from `knots` values spread over the training horizon.

    After the horizon (e.g. during test) it holds `continuation_logit`, which is
    estimated from the final training state only.
    """
    kind = 'temporal'

    def __init__(self, theta, rules: Rules, knots: int = 33, horizon_years: float = 8.,
                 continuation_logit: float | None = None):
        super().__init__(theta, rules)
        self.knots, self.horizon_years, self.continuation_logit = knots, horizon_years, continuation_logit

    def with_continuation(self, logit: float) -> TemporalPolicy:
        return TemporalPolicy(self.theta, self.rules, self.knots, self.horizon_years, logit)

    def logits(self, thetas, features, year):
        if self.continuation_logit is not None and year >= self.horizon_years:
            return np.full(len(thetas), self.continuation_logit)
        z = np.clip(year / self.horizon_years * (self.knots - 1), 0, self.knots - 1)
        left = min(int(z), self.knots - 2)
        frac = z - left
        return (1 - frac) * thetas[:, left] + frac * thetas[:, left + 1]

    def to_dict(self) -> dict:
        return {**super().to_dict(), 'knots': self.knots, 'horizon_years': self.horizon_years,
                'continuation_logit': self.continuation_logit}


class Population:
    """Many thetas of one policy class evaluated side by side on a market."""

    def __init__(self, policy: CEMPolicy, thetas: np.ndarray, market: Market):
        self.policy, self.thetas, self.market = policy, thetas, market
        self.controls = thetas[:, -2:]

    def __len__(self) -> int:
        return len(self.thetas)

    def logits(self, t: int) -> np.ndarray:
        return self.policy.logits(self.thetas, self.market.features[t - 1], self.market.years[t])


class PenaltyAwareHold:
    """Baseline: buy the maximum while under-invested, else trade minimally to dodge penalties."""
    kind = 'penalty_aware_hold'

    def predict(self, obs, account: Account, action_mask) -> int:
        exposure, idle_fraction = obs[N_FEATURES], obs[N_FEATURES + 1]
        if exposure < .98 and action_mask[3]:
            return 3
        if idle_fraction >= 20 / 21:
            return 1 if action_mask[1] else 4 if action_mask[4] else 0
        return 0


POLICY_TYPES = {cls.kind: cls for cls in (IndicatorPolicy, TemporalPolicy)}


def save_policy(policy: CEMPolicy, path: Path) -> None:
    Path(path).write_text(json.dumps(policy.to_dict(), indent=2), encoding='utf-8')


def load_policy(path: Path, rules: Rules) -> CEMPolicy:
    spec = json.loads(Path(path).read_text(encoding='utf-8'))
    cls = POLICY_TYPES[spec.pop('policy')]
    return cls(rules=rules, **spec)
