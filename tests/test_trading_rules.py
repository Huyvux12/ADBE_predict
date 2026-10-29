"""Unit tests for the assignment rules on a synthetic flat market (price 100)."""
import numpy as np
import pandas as pd
import pytest

from adbe_rl.broker import order_quantities
from adbe_rl.config import load_config
from adbe_rl.env import OBSERVATION_SIZE, TradingEnv
from adbe_rl.market import Market

RULES = load_config().rules
HOLD, BUY_MIN, BUY_MAX, SELL_MIN = 0, 1, 3, 4


def flat_market(days: int = 120, price: float = 100., shock_from: int | None = None) -> Market:
    """Constant prices; optionally scaled by 1.5 from row `shock_from` on."""
    prices = np.full(days, price)
    if shock_from is not None:
        prices[shock_from:] *= 1.5
    dates = pd.bdate_range('2000-01-03', periods=days)
    return Market.from_prices(pd.DataFrame({'Date': dates, 'Open': prices, 'High': prices,
                                            'Low': prices, 'Close': prices}))


@pytest.fixture
def env():
    return TradingEnv(flat_market(), 1, 100, RULES)


def test_buy_max_matches_report_example(env):
    *_, info = env.step(BUY_MAX)
    assert info['quantity'] == 730 and info['notional'] == 73_000
    assert info['fee'] == pytest.approx(2_920) and info['cash'] == pytest.approx(24_080)


def test_order_sizes_stay_within_limits():
    for cash, shares, price in [(100_000, 0, 3.21875), (5_799.63, 28_088, 4.02), (10, 7, 950.)]:
        q = order_quantities(np.array(cash), np.array(shares), price, RULES)
        buys, sells = q[1:4][q[1:4] != 0], -q[4:][q[4:] != 0]
        assert np.all(buys * price >= RULES.buy_min * cash - 1e-9)
        assert np.all(buys * price * (1 + RULES.fee_rate) <= cash + 1e-9)
        assert np.all((sells >= RULES.sell_min * shares) & (sells <= RULES.sell_max * shares))
        assert np.array_equal(q, np.round(q))


def test_penalty_after_21_idle_sessions(env):
    infos = [env.step(HOLD)[-1] for _ in range(21)]
    assert [i['penalty'] for i in infos] == [0.] * 20 + [RULES.idle_penalty]
    assert infos[-1]['idle'] == 0 and infos[-1]['cash'] == RULES.initial_cash - RULES.idle_penalty


def test_infeasible_order_does_not_reset_idle(env):
    env.step(HOLD)
    *_, info = env.step(SELL_MIN)          # no shares to sell
    assert info['trade'] == 'hold' and info['idle'] == 2
    assert not env.action_mask()[SELL_MIN]


def test_trade_resets_idle(env):
    for _ in range(20):
        env.step(HOLD)
    *_, info = env.step(BUY_MIN)
    assert info['idle'] == 0 and info['penalty'] == 0


@pytest.mark.parametrize('cash, shares, reason', [(1_000_000, 0, 'win'), (999, 0, 'loss_assets'),
                                                  (-5_000, 1_000, 'loss_cash')])
def test_terminal_conditions(env, cash, shares, reason):
    env.book.cash[:], env.book.shares[:] = cash, shares
    _, _, terminated, truncated, info = env.step(HOLD)
    assert terminated and not truncated and info['reason'] == reason and env.done


def test_truncated_at_end_of_data():
    env = TradingEnv(flat_market(), 1, 2, RULES)
    _, _, terminated, truncated, info = env.step(HOLD)
    assert truncated and not terminated and info['reason'] == 'end_of_data'


def test_reward_sums_to_log_return(env):
    for action in [BUY_MAX, HOLD, SELL_MIN, HOLD, BUY_MIN]:
        env.step(action)
    assets = env.history[-1]['assets']
    assert env.reward_sum == pytest.approx(np.log(assets / RULES.initial_cash))


def test_observation_shape_and_causality():
    market = flat_market()
    obs = TradingEnv(market, 50, 100, RULES).observation()
    assert obs.shape == (OBSERVATION_SIZE,) and obs.dtype == np.float32
    shocked = flat_market(shock_from=50)
    assert np.array_equal(obs, TradingEnv(shocked, 50, 100, RULES).observation())
