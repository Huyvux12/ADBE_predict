"""Order sizing and settlement, vectorized over independent portfolios.

The single-episode environment and the population simulator share this code,
so both apply exactly the same fees, penalties and win/loss checks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Rules

ACTIONS = ('hold', 'buy_min', 'buy_mid', 'buy_max', 'sell_min', 'sell_mid', 'sell_max')
N_ACTIONS = len(ACTIONS)
_ROUNDING = 1e-10


def order_bounds(cash, shares, price, rules: Rules):
    """Whole-share limits (buy_lo, buy_hi, sell_lo, sell_hi) allowed by the rules."""
    cash = np.maximum(np.asarray(cash), 0)
    shares = np.asarray(shares)
    buy_lo = rules.buy_min * cash / price
    buy_hi = np.minimum(rules.buy_max * cash / price, cash / (price * (1 + rules.fee_rate)))
    sell_lo = rules.sell_min * shares
    sell_hi = rules.sell_max * shares
    return (np.ceil(buy_lo - _ROUNDING), np.floor(buy_hi + _ROUNDING),
            np.ceil(sell_lo - _ROUNDING), np.floor(sell_hi + _ROUNDING))


def order_quantities(cash, shares, price, rules: Rules) -> np.ndarray:
    """Signed share quantity of every action, shape (..., 7); 0 marks an infeasible order."""
    cash = np.asarray(cash)
    shares = np.asarray(shares)
    buy_lo, buy_hi, sell_lo, sell_hi = order_bounds(cash, shares, price, rules)
    buy_fractions = np.asarray([rules.buy_min, (rules.buy_min + rules.buy_max) / 2, rules.buy_max])
    sell_fractions = np.asarray([rules.sell_min, (rules.sell_min + rules.sell_max) / 2, rules.sell_max])
    buys = np.floor(np.maximum(cash, 0)[..., None] * buy_fractions / price + _ROUNDING)
    sells = np.floor(shares[..., None] * sell_fractions + _ROUNDING)
    buys = np.clip(buys, buy_lo[..., None], buy_hi[..., None])
    sells = np.clip(sells, sell_lo[..., None], sell_hi[..., None])
    buys = np.where(((buy_hi >= buy_lo) & (buy_hi > 0))[..., None], buys, 0)
    sells = np.where(((sell_hi >= sell_lo) & (sell_hi > 0))[..., None], sells, 0)
    return np.concatenate([np.zeros_like(cash)[..., None], buys, -sells], axis=-1)


@dataclass
class Fill:
    quantity: np.ndarray
    price: np.ndarray
    notional: np.ndarray
    fee: np.ndarray
    penalty: np.ndarray
    assets: np.ndarray
    outcome: np.ndarray      # '', 'win', 'loss_assets' or 'loss_cash'


class Book:
    """Cash and share ledgers of n portfolios; finished portfolios stay frozen."""

    def __init__(self, n: int, rules: Rules):
        self.rules = rules
        self.cash = np.full(n, rules.initial_cash)
        self.shares = np.zeros(n)
        self.idle = np.zeros(n, dtype=int)
        self.peak = self.cash.copy()
        self.final = self.cash.copy()
        self.alive = np.ones(n, dtype=bool)
        self.lost = np.zeros(n, dtype=bool)   # ended by a loss condition

    def step(self, quantity, buy_price, sell_price, mark_price) -> Fill:
        r = self.rules
        quantity = np.where(self.alive, quantity, 0.)
        price = np.where(quantity > 0, buy_price, sell_price)
        notional = np.abs(quantity) * price
        fee = notional * r.fee_rate
        self.cash -= quantity * price + fee
        self.shares += quantity

        self.idle = np.where(quantity != 0, 0, self.idle + self.alive)
        penalized = self.alive & (self.idle >= r.idle_limit)
        penalty = penalized * r.idle_penalty
        self.cash -= penalty
        self.idle = np.where(penalized, 0, self.idle)

        assets = self.cash + self.shares * mark_price
        self.peak = np.where(self.alive, np.maximum(self.peak, assets), self.peak)
        self.final = np.where(self.alive, assets, self.final)
        outcome = np.select([assets >= r.win_assets, assets < r.loss_assets, self.cash <= r.loss_cash],
                            ['win', 'loss_assets', 'loss_cash'], '')
        outcome = np.where(self.alive, outcome, '')
        self.lost |= (outcome == 'loss_assets') | (outcome == 'loss_cash')
        self.alive &= outcome == ''
        return Fill(quantity, price, notional, fee, penalty, assets, outcome)
