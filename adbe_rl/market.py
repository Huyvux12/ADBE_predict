"""Daily OHLC data, causal market features and train/test splits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config, Split

PRICE_COLUMNS = ['Open', 'High', 'Low', 'Close']
FEATURE_NAMES = ['return_1', 'return_5', 'return_20', 'return_60', 'return_252',
                 'ma_20', 'ma_60', 'ma_200', 'rsi_14', 'volatility_20',
                 'relative_price_ma252', 'drawdown_252']
N_FEATURES = len(FEATURE_NAMES)


def load_prices(path) -> pd.DataFrame:
    """Read and validate a daily OHLC csv sorted by date."""
    df = pd.read_csv(path, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    prices = df[PRICE_COLUMNS]
    if df.Date.duplicated().any() or prices.isna().any().any():
        raise ValueError('Duplicate dates or missing OHLC')
    if (prices <= 0).any().any():
        raise ValueError('Non-positive price')
    high_ok = df.High >= df[['Open', 'Close', 'Low']].max(axis=1) - 1e-4
    low_ok = df.Low <= df[['Open', 'Close', 'High']].min(axis=1) + 1e-4
    if not (high_ok & low_ok).all():
        raise ValueError('Invalid OHLC')
    return df


def market_features(close: pd.Series) -> np.ndarray:
    """Scaled indicators from closes; row i uses closes up to day i only."""
    columns = []
    for days, scale in [(1, 30), (5, 15), (20, 8), (60, 4), (252, 2)]:
        columns.append(close.pct_change(days).fillna(0).to_numpy() * scale)
    for days in [20, 60, 200]:
        columns.append((close / close.rolling(days, min_periods=1).mean() - 1).to_numpy() * 8)
    delta = close.diff().fillna(0)
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = (up / (up + down).replace(0, np.nan)).fillna(0.5)
    volatility = close.pct_change().rolling(20, min_periods=2).std().fillna(0)
    c = close.to_numpy()
    columns.extend([
        (rsi.to_numpy() - .5) * 2,
        volatility.to_numpy() * 30,
        np.log(c / close.rolling(252, min_periods=1).mean().to_numpy()) * 3,
        (c / close.rolling(252, min_periods=1).max().to_numpy() - 1) * 5,
    ])
    return np.clip(np.column_stack(columns), -5, 5).astype(np.float32)


@dataclass(frozen=True)
class Market:
    """Price arrays for one data window. Features of row i become known on day i+1."""
    dates: pd.Series
    ohlc: np.ndarray
    features: np.ndarray
    years: np.ndarray        # calendar years since the first row

    @classmethod
    def from_prices(cls, df: pd.DataFrame) -> Market:
        df = df.reset_index(drop=True)
        years = (df.Date - df.Date.iloc[0]).dt.days.to_numpy() / 365.25
        return cls(df.Date, df[PRICE_COLUMNS].to_numpy(), market_features(df.Close), years)

    def __len__(self) -> int:
        return len(self.ohlc)

    @property
    def high(self) -> np.ndarray:
        return self.ohlc[:, 1]

    def index(self, date) -> int:
        """First row on or after `date`."""
        return int(np.searchsorted(self.dates.values, np.datetime64(pd.Timestamp(date))))

    def quotes(self, t: int, sell_price: str) -> tuple[float, float, float]:
        """(buy, sell, mark) prices for a decision on day t.

        Buys fill at the previous High (known before the order); sells fill at
        today's High or Close; holdings are marked at today's High.
        """
        _, high, _, close = self.ohlc[t]
        sell = {'high': high, 'close': close}[sell_price]
        return self.ohlc[t - 1, 1], sell, high


@dataclass(frozen=True)
class Track:
    """A market window and its fit/validation/train/test session ranges."""
    name: str
    split: Split
    market: Market
    ranges: dict[str, tuple[int, int]]

    def dates(self, period: str) -> pd.Series:
        start, end = self.ranges[period]
        return self.market.dates.iloc[start:end]

    def describe(self) -> dict:
        info = {period: {'first_session': str(d.iloc[0].date()), 'last_session': str(d.iloc[-1].date()),
                         'sessions': len(d)}
                for period in self.ranges for d in [self.dates(period)]}
        gap = pd.Timestamp(self.split.test_start) - pd.Timestamp(self.split.train_end)
        info['gap_calendar_days'] = gap.days
        return info


def load_track(config: Config, name: str, prices: pd.DataFrame | None = None) -> Track:
    """Market starts one session before training so day 1 has a previous High."""
    split = config.tracks[name]
    prices = load_prices(config.data_file) if prices is None else prices
    first_train = max(1, int(np.searchsorted(prices.Date.values, np.datetime64(split.train_start))))
    market = Market.from_prices(prices.iloc[first_train - 1:])
    ranges = {'fit': (1, market.index(split.validation_start)),
              'validation': (market.index(split.validation_start), market.index(split.train_end)),
              'train': (1, market.index(split.train_end)),
              'test': (market.index(split.test_start), market.index(split.test_end))}
    check_split(split, market, ranges)
    return Track(name, split, market, ranges)


def check_split(split: Split, market: Market, ranges: dict[str, tuple[int, int]]) -> None:
    """8 calendar years of training strictly before 2 calendar years of test."""
    assert pd.Timestamp(split.train_end) == pd.Timestamp(split.train_start) + pd.DateOffset(years=8)
    assert pd.Timestamp(split.test_end) == pd.Timestamp(split.test_start) + pd.DateOffset(years=2)
    assert ranges['train'][1] <= ranges['test'][0]
    train = set(market.dates.iloc[slice(*ranges['train'])])
    test = set(market.dates.iloc[slice(*ranges['test'])])
    assert train and test and not train & test and max(train) < min(test)
