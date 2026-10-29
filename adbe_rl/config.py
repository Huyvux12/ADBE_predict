"""Project configuration: trading rules, data splits, training budget and paths."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / 'config.json'
OUTPUT_DIR = ROOT / 'outputs'

# Sell-price scenarios evaluated for every model: name -> OHLC column used for sells.
SCENARIOS = {'sell_high': 'high', 'sell_close': 'close'}


@dataclass(frozen=True)
class Rules:
    """Assignment rules for ADBE. Fractions are 0-1, money is USD."""
    initial_cash: float
    buy_min: float
    buy_max: float
    sell_min: float
    sell_max: float
    fee_rate: float
    idle_limit: int          # TRANSACTION_SESSION
    idle_penalty: float      # TRANSACTION_PENALTY
    loss_assets: float       # TOTAL_ASSETS
    loss_cash: float
    win_assets: float
    sell_price: str = 'high'

    def with_sell_price(self, sell_price: str) -> Rules:
        return replace(self, sell_price=sell_price)


@dataclass(frozen=True)
class Split:
    """Calendar dates; every *_end is exclusive."""
    train_start: str
    validation_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class Budget:
    population: int
    generations: int


@dataclass(frozen=True)
class Config:
    symbol: str
    data_file: Path
    rules: Rules
    main_track: str
    tracks: dict[str, Split]
    seeds: tuple[int, ...]
    indicator: Budget
    temporal: Budget
    replay_windows: tuple[tuple[str, str], ...]

    def quick(self) -> Config:
        """Small budget for smoke tests; results are not comparable to the full run."""
        small = Budget(population=128, generations=12)
        return replace(self, seeds=(42,), indicator=small, temporal=small)


def load_config(path: Path = CONFIG_PATH) -> Config:
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    training = raw['training']
    return Config(
        symbol=raw['symbol'],
        data_file=ROOT / raw['data_file'],
        rules=Rules(**raw['rules']),
        main_track=raw['main_track'],
        tracks={name: Split(**split) for name, split in raw['tracks'].items()},
        seeds=tuple(training['seeds']),
        indicator=Budget(**training['indicator']),
        temporal=Budget(**training['temporal']),
        replay_windows=tuple(tuple(window) for window in raw['replay_windows']),
    )


def track_dir(output: Path, track: str) -> Path:
    return Path(output) / track
