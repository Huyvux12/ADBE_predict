"""Audit saved results: rules, accounting, splits, causality and frozen-model replay."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SCENARIOS, Config, Rules, track_dir
from .env import TradingEnv, run_episode
from .evaluation import load_models, model_hashes
from .market import Market, load_prices, load_track
from .policies import CEMPolicy
from .simulate import simulate

CHECKS = [
    '8-year train and 2-year test calendar windows, zero overlap, train before test',
    'buy orders within 35-73% of cash, sell orders within 13-82% of shares, whole shares',
    '4% fee on both sides; cash, shares and assets reconcile every session',
    'penalty exactly after 21 idle sessions; failed orders do not reset the counter',
    'buy fills at previous High, sell fills at today High/Close',
    'cumulative reward equals the sum of rewards; only the last row ends the episode',
    'observations unaffected by current or future prices',
    'single-episode env equals the vectorized simulator',
    'load/predict replay reproduces stored summaries',
    'model files unchanged by evaluation',
]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision='round_trip')


def audit_history(h: pd.DataFrame, rules: Rules, market: Market) -> None:
    buy, sell = h.trade == 'buy', h.trade == 'sell'
    sign = np.where(buy, 1, np.where(sell, -1, 0))
    assert h.cash_before.iloc[0] == rules.initial_cash and h.shares_before.iloc[0] == 0
    assert np.allclose(h.fee, h.notional * rules.fee_rate)
    assert np.allclose(h.assets, h.cash + h.shares * h.mark_price)
    assert np.allclose(h.cash, h.cash_before - sign * h.notional - h.fee - h.penalty)
    assert np.allclose(h.shares, h.shares_before + sign * h.quantity)
    assert (h.shares >= 0).all() and np.array_equal(h.quantity, np.round(h.quantity))
    if buy.any():
        ratio = h.notional[buy] / h.cash_before[buy]
        assert ratio.between(rules.buy_min - 1e-7, rules.buy_max + 1e-7).all()
        assert (h.cash[buy] >= -1e-7).all()
    if sell.any():
        ratio = h.quantity[sell] / h.shares_before[sell]
        assert ratio.between(rules.sell_min - 1e-7, rules.sell_max + 1e-7).all()

    idle = 0
    for row in h.itertuples():
        idle = 0 if row.quantity > 0 else idle + 1
        penalty = rules.idle_penalty if idle >= rules.idle_limit else 0
        idle = 0 if penalty else idle
        assert row.penalty == penalty and row.idle == idle

    t = np.searchsorted(market.dates.values, pd.to_datetime(h.Date).values)
    sell_column = {'high': 1, 'close': 3}[rules.sell_price]
    is_buy, is_sell = buy.to_numpy(), sell.to_numpy()
    assert np.allclose(h.execution_price[buy], market.ohlc[t[is_buy] - 1, 1])
    assert np.allclose(h.execution_price[sell], market.ohlc[t[is_sell], sell_column])
    assert np.allclose(h.mark_price, market.ohlc[t, 1])

    assert np.allclose(h.cumulative_reward, h.reward.cumsum())
    assert not (h.terminated | h.truncated).iloc[:-1].any()
    assert h.terminated.iloc[-1] or h.truncated.iloc[-1]


def scenario_of(path: Path) -> str:
    return next(s for s in SCENARIOS if path.stem.endswith(s))


def audit_folder(folder: Path, rules: Rules, market: Market) -> list[str]:
    audited = []
    for path in sorted(folder.glob('*.csv')):
        if path.name == 'summary.csv':
            continue
        audit_history(read_csv(path), rules.with_sell_price(SCENARIOS[scenario_of(path)]), market)
        audited.append(path.as_posix())
    return audited


def check_parity(models: dict, market: Market, ranges: dict, rules: Rules) -> None:
    for name, model in models.items():
        if not isinstance(model, CEMPolicy):
            continue
        for period in ['train', 'test']:
            for sell_price in SCENARIOS.values():
                r = rules.with_sell_price(sell_price)
                _, metrics = run_episode(model, TradingEnv(market, *ranges[period], r))
                peak, final, _ = simulate(model.population(model.theta, market), *ranges[period], r)
                assert np.isclose(metrics['peak_assets'], peak[0], rtol=1e-12), (name, period, sell_price)
                assert np.isclose(metrics['final_assets'], final[0], rtol=1e-12), (name, period, sell_price)


def check_replay(models: dict, market: Market, ranges: dict, rules: Rules, summary: pd.DataFrame) -> None:
    for row in summary.itertuples():
        env = TradingEnv(market, *ranges[row.period], rules.with_sell_price(SCENARIOS[row.scenario]))
        _, metrics = run_episode(models[row.model], env)
        assert metrics['final_assets'] == row.final_assets and metrics['peak_assets'] == row.peak_assets
        assert metrics['end_reason'] == row.end_reason


def check_causality(config: Config, name: str) -> None:
    """Scaling every price from the first test day on must not change that day's observation."""
    track = load_track(config, name)
    start, end = track.ranges['test']
    observed = TradingEnv(track.market, start, end, config.rules).observation()
    prices = load_prices(config.data_file)
    first_test = prices.Date >= track.market.dates.iloc[start]
    prices.loc[first_test, ['Open', 'High', 'Low', 'Close']] *= 1.5
    shocked = load_track(config, name, prices)
    assert np.array_equal(observed, TradingEnv(shocked.market, start, end, config.rules).observation())


def verify(config: Config, output: Path, tracks: list[str] | None = None) -> dict:
    audited = []
    for name in tracks or list(config.tracks):
        track = load_track(config, name)
        folder = track_dir(output, name)
        stored = json.loads((folder / 'model_hashes.json').read_text(encoding='utf-8'))
        assert model_hashes(folder / 'models') == stored, f'{name}: models differ from evaluated ones'
        models = load_models(folder / 'models', config.rules)
        audited += audit_folder(folder / 'histories', config.rules, track.market)
        check_parity(models, track.market, track.ranges, config.rules)
        check_replay(models, track.market, track.ranges, config.rules, read_csv(folder / 'summary.csv'))
        check_causality(config, name)
        print(f'[{name}] verified', flush=True)
    replay_dir = Path(output) / 'replay'
    if replay_dir.exists():
        audited += audit_folder(replay_dir, config.rules, Market.from_prices(load_prices(config.data_file)))
    result = {'status': 'PASS', 'checks': CHECKS, 'audited_histories': len(audited)}
    (Path(output) / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result
