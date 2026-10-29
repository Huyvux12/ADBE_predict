"""Frozen-model evaluation: load saved policies and predict; never trains."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import SCENARIOS, Config, track_dir
from .env import TradingEnv, run_episode
from .market import Market, load_prices, load_track
from .policies import PenaltyAwareHold, load_policy

MODEL_NAMES = ['penalty_aware_hold', 'cem_validation', 'cem_train_peak', 'cem_temporal',
               'cem_temporal_continuation']
LABELS = {'penalty_aware_hold': 'Giữ + tránh phạt', 'cem_validation': 'CEM chọn validation',
          'cem_train_peak': 'CEM chỉ báo, đỉnh train', 'cem_temporal': 'CEM theo thời gian',
          'cem_temporal_continuation': 'CEM thời gian + tiếp tục'}


def load_models(models_dir: Path, rules) -> dict:
    return {name: PenaltyAwareHold() if name == 'penalty_aware_hold'
            else load_policy(models_dir / f'{name}.json', rules) for name in MODEL_NAMES}


def model_hashes(models_dir: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(models_dir.glob('*.json'))}


def history_name(model: str, period: str, scenario: str) -> str:
    return f'{model}_{period}_{scenario}.csv'


def replay(models: dict, market: Market, start: int, end: int, config: Config, out: Path, prefix: str,
           tags: dict) -> list[dict]:
    """Every model x sell-price scenario from 100,000 USD cash; writes daily logs.

    Sell prices only change the fills, not the policies' order-sizing rules.
    """
    rows = []
    for scenario, sell_price in SCENARIOS.items():
        rules = config.rules.with_sell_price(sell_price)
        for name, model in models.items():
            history, metrics = run_episode(model, TradingEnv(market, start, end, rules))
            history.to_csv(out / history_name(name, prefix, scenario), index=False)
            rows.append({**tags, 'model': name, 'scenario': scenario, **metrics})
    return rows


def evaluate_track(config: Config, name: str, output: Path) -> pd.DataFrame:
    track = load_track(config, name)
    folder = track_dir(output, name)
    (folder / 'histories').mkdir(parents=True, exist_ok=True)
    (folder / 'splits.json').write_text(json.dumps(track.describe(), indent=2), encoding='utf-8')
    before = model_hashes(folder / 'models')
    models = load_models(folder / 'models', config.rules)
    rows = []
    for period in ['train', 'test']:
        rows += replay(models, track.market, *track.ranges[period], config, folder / 'histories', period,
                       {'track': name, 'period': period})
    assert model_hashes(folder / 'models') == before, 'model files changed during evaluation'
    (folder / 'model_hashes.json').write_text(json.dumps(before, indent=2), encoding='utf-8')
    summary = pd.DataFrame(rows)
    summary.to_csv(folder / 'summary.csv', index=False)
    print(summary[summary.scenario == 'sell_high'][['track', 'model', 'period', 'peak_assets', 'final_assets',
                                                   'end_reason']].to_string(index=False), flush=True)
    return summary


def replay_windows(config: Config, output: Path) -> pd.DataFrame:
    """Main-track checkpoints on later 2-year windows; uses the full price history for features."""
    folder = Path(output) / 'replay'
    folder.mkdir(parents=True, exist_ok=True)
    models_dir = track_dir(output, config.main_track) / 'models'
    before = model_hashes(models_dir)
    models = load_models(models_dir, config.rules)
    market = Market.from_prices(load_prices(config.data_file))
    train_end = pd.Timestamp(config.tracks[config.main_track].train_end)
    rows = []
    for first, stop in config.replay_windows:
        assert pd.Timestamp(stop) == pd.Timestamp(first) + pd.DateOffset(years=2)
        assert pd.Timestamp(first) >= train_end, 'replay windows must not overlap training'
        start, end = market.index(first), market.index(stop)
        rows += replay(models, market, start, end, config, folder, f'{first}_{stop}',
                       {'window_start': first, 'window_end': stop})
    assert model_hashes(models_dir) == before, 'model files changed during replay'
    summary = pd.DataFrame(rows)
    summary['return_pct'] = (summary.final_assets / config.rules.initial_cash - 1) * 100
    summary.to_csv(folder / 'summary.csv', index=False)
    return summary
