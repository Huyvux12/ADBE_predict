"""Single-stock trading environment with a gym-style reset/step API."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .broker import N_ACTIONS, Book, order_quantities
from .config import Rules
from .market import N_FEATURES, Market

OBSERVATION_NAMES = ['exposure', 'idle_fraction', 'log_assets', 'assets_drawdown',
                     'log_previous_high', 'years']
OBSERVATION_SIZE = N_FEATURES + len(OBSERVATION_NAMES)


@dataclass(frozen=True)
class Account:
    """Exact portfolio state a broker would report; used to size whole-share orders."""
    cash: float
    shares: float
    price: float      # previous High = buy price of the current session
    idle: int


class TradingEnv:
    """One episode over market sessions [start, end).

    Observation (18 float32): 12 market features of the previous day, then
    exposure, idle/limit, log(assets/initial), assets/peak - 1,
    log(previous High / 100) and years since the first market row.
    Reward: log(assets_t / assets_{t-1}); fees and penalties lower it directly.
    """

    def __init__(self, market: Market, start: int, end: int, rules: Rules):
        if not 1 <= start < end <= len(market):
            raise ValueError('Need a previous session and at least one day')
        self.market, self.start, self.end, self.rules = market, int(start), int(end), rules
        self.reset()

    def reset(self):
        self.t = self.start
        self.book = Book(1, self.rules)
        self.history: list[dict] = []
        self.reward_sum = 0.
        self.done = False
        return self.observation(), {}

    @property
    def account(self) -> Account:
        return Account(float(self.book.cash[0]), float(self.book.shares[0]),
                       float(self.market.high[self.t - 1]), int(self.book.idle[0]))

    def observation(self) -> np.ndarray:
        cash, shares, peak = self.book.cash[0], self.book.shares[0], self.book.peak[0]
        price = self.market.high[self.t - 1]
        assets = cash + shares * price
        account = np.array([shares * price / max(assets, 1e-6),
                            self.book.idle[0] / self.rules.idle_limit,
                            np.log(max(assets, 1) / self.rules.initial_cash),
                            assets / max(peak, 1) - 1,
                            np.log(price / 100),
                            self.market.years[min(self.t, len(self.market) - 1)]], np.float32)
        return np.concatenate([self.market.features[self.t - 1], account]).astype(np.float32)

    def action_mask(self) -> np.ndarray:
        """Hold is always legal; other actions need a feasible whole-share order."""
        q = order_quantities(self.book.cash, self.book.shares, self.market.high[self.t - 1], self.rules)[0]
        return np.r_[True, q[1:] != 0]

    def step(self, action: int):
        if self.done:
            raise RuntimeError('Call reset after done')
        action = int(action)
        if not 0 <= action < N_ACTIONS:
            raise ValueError('Bad action ID')
        t, book = self.t, self.book
        buy, sell, mark = self.market.quotes(t, self.rules.sell_price)
        cash_before, shares_before = float(book.cash[0]), float(book.shares[0])
        assets_before = cash_before + shares_before * buy
        quantity = order_quantities(book.cash, book.shares, buy, self.rules)[0, action]
        fill = book.step(np.array([quantity]), buy, sell, mark)

        assets = float(fill.assets[0])
        reward = float(np.log(max(assets, 1e-8) / max(assets_before, 1e-8)))
        self.reward_sum += reward
        reason = str(fill.outcome[0])
        terminated = bool(reason)
        self.t += 1
        truncated = self.t >= self.end and not terminated
        self.done = terminated or truncated
        traded = quantity != 0
        info = {'Date': str(self.market.dates.iloc[t].date()), 'action': action,
                'trade': 'buy' if quantity > 0 else 'sell' if quantity < 0 else 'hold',
                'cash_before': cash_before, 'shares_before': shares_before,
                'cash': float(book.cash[0]), 'shares': float(book.shares[0]),
                'quantity': abs(float(quantity)),
                'execution_price': float(fill.price[0]) if traded else 0.,
                'notional': float(fill.notional[0]), 'fee': float(fill.fee[0]),
                'penalty': float(fill.penalty[0]), 'idle': int(book.idle[0]),
                'mark_price': float(mark), 'assets': assets, 'reward': reward,
                'cumulative_reward': self.reward_sum, 'terminated': terminated, 'truncated': truncated,
                'reason': reason or ('end_of_data' if truncated else '')}
        self.history.append(info)
        return self.observation(), reward, terminated, truncated, info


def run_episode(policy, env: TradingEnv) -> tuple[pd.DataFrame, dict]:
    """Play one episode with `policy.predict` and return (daily log, metrics)."""
    obs, _ = env.reset()
    while not env.done:
        action = policy.predict(obs, env.account, env.action_mask())
        obs, *_ = env.step(action)
    history = pd.DataFrame(env.history)
    return history, episode_metrics(history, env.rules)


def episode_metrics(history: pd.DataFrame, rules: Rules) -> dict:
    assets = history.assets.to_numpy()
    running_peak = np.maximum.accumulate(np.r_[rules.initial_cash, assets])[1:]
    k = int(np.argmax(assets))
    peak = max(float(assets[k]), rules.initial_cash)
    return {'final_assets': float(assets[-1]), 'peak_assets': peak,
            'peak_date': str(history.Date.iloc[k]) if assets[k] >= rules.initial_cash else 'initial',
            'score_ratio_peak': peak / rules.win_assets,
            'cumulative_reward': float(history.reward.sum()),
            'trades': int((history.quantity > 0).sum()),
            'total_fees': float(history.fee.sum()), 'total_penalty': float(history.penalty.sum()),
            'max_drawdown': float(np.min(assets / running_peak - 1)),
            'sessions': len(history), 'end_reason': str(history.reason.iloc[-1])}
