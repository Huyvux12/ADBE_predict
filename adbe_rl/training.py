"""Training pipeline for one track. Only fit/validation/train sessions are used."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cem import CEMSettings, cem_search, fitness
from .config import Config, track_dir
from .env import TradingEnv, run_episode
from .market import N_FEATURES, Track, load_track
from .policies import CEMPolicy, IndicatorPolicy, TemporalPolicy, save_policy
from .simulate import simulate

KNOTS = 33
HORIZON_YEARS = 8.


def indicator_settings(config: Config) -> CEMSettings:
    mean = np.zeros(IndicatorPolicy.size)
    mean[N_FEATURES] = 3        # start mostly invested
    mean[N_FEATURES + 1] = -2   # small hysteresis
    return CEMSettings(config.indicator.population, config.indicator.generations, tuple(mean),
                       init_std=2., keep=.25, std_floor=.20, weights=(.90, .10), restart_every=5)


def temporal_settings(config: Config) -> CEMSettings:
    mean = np.full(KNOTS + 2, 4.)
    mean[-2:] = [-2, 0]
    return CEMSettings(config.temporal.population, config.temporal.generations, tuple(mean),
                       init_std=3., keep=.2, std_floor=.35, weights=(.95, .05))


def fully_invested_candidate(x: np.ndarray) -> None:
    """Seed generation 0 with an all-in schedule so buy-and-hold is always explored."""
    x[1, :KNOTS] = 8


def train_track(config: Config, name: str, output: Path) -> None:
    track = load_track(config, name)
    folder = track_dir(output, name)
    for sub in ['models', 'learning']:
        (folder / sub).mkdir(parents=True, exist_ok=True)
    (folder / 'splits.json').write_text(json.dumps(track.describe(), indent=2), encoding='utf-8')
    rules = config.rules
    indicator = IndicatorPolicy(np.zeros(IndicatorPolicy.size), rules)
    temporal = TemporalPolicy(np.zeros(KNOTS + 2), rules, KNOTS, HORIZON_YEARS)

    def evaluator(template: CEMPolicy, period: str):
        return lambda thetas: simulate(template.population(thetas, track.market), *track.ranges[period], rules)

    def search(template, settings, period, warm_start=None):
        results = []
        for seed in config.seeds:
            print(f'[{name}] {template.kind} CEM on {period}, seed {seed}', flush=True)
            results.append(cem_search(evaluator(template, period), settings, seed,
                                      rules.initial_cash, warm_start))
        return results

    # 1. Indicator policy fitted on 7 years, checkpoint chosen on the last training year.
    fitted = search(indicator, indicator_settings(config), 'fit')
    candidates = np.asarray([theta for r in fitted for theta in [*r.archive, r.best, r.best_peak_theta]])
    peaks, finals, lost = simulate(indicator.population(candidates, track.market),
                                   *track.ranges['validation'], rules)
    k = int(np.argmax(np.where(lost, -np.inf, finals)))   # never pick a checkpoint that loses
    save_policy(IndicatorPolicy(candidates[k], rules), folder / 'models/cem_validation.json')
    pd.DataFrame({'candidate': np.arange(len(finals)), 'validation_peak': peaks,
                  'validation_final': finals, 'validation_lost': lost}).to_csv(folder / 'learning/validation_selection.csv', index=False)

    # 2. Indicator policy fitted on all 8 years, highest training peak.
    trained = search(indicator, indicator_settings(config), 'train')
    champion = max(trained, key=lambda r: r.best_peak)
    save_policy(IndicatorPolicy(champion.best_peak_theta, rules), folder / 'models/cem_train_peak.json')

    # 3. Time-conditioned allocation schedule, highest training peak.
    schedules = search(temporal, temporal_settings(config), 'train', fully_invested_candidate)
    champion = max(schedules, key=lambda r: r.best_peak)
    schedule = TemporalPolicy(champion.best_peak_theta, rules, KNOTS, HORIZON_YEARS)
    save_policy(schedule, folder / 'models/cem_temporal.json')

    # 4. Continuation: after the horizon, hold the constant exposure that did best on training.
    continuation = fit_continuation(schedule, track, temporal_settings(config).weights)
    save_policy(schedule.with_continuation(continuation['continuation_logit']),
                folder / 'models/cem_temporal_continuation.json')
    (folder / 'learning/continuation.json').write_text(json.dumps(continuation, indent=2), encoding='utf-8')

    for label, results in [('indicator_fit', fitted), ('indicator_train', trained), ('temporal_train', schedules)]:
        pd.DataFrame([row for r in results for row in r.log]).to_csv(folder / f'learning/{label}.csv', index=False)


CONTINUATION_GRID = np.round(np.arange(0, 1.0001, .05), 2)


def fit_continuation(schedule: TemporalPolicy, track: Track, weights: tuple[float, float]) -> dict:
    """Constant exposure used after the schedule horizon (e.g. during test).

    Every exposure on a grid is run as a flat schedule (same trading controls)
    over the whole training period from 100,000 USD cash under the real rules,
    and the best one by the CEM fitness is kept. Test data is never read.

    Also records how far the training replay of the schedule got: the episode
    stops at the win condition, so knots after that date never affect training.
    """
    rules = schedule.rules
    exposures = np.clip(CONTINUATION_GRID, 1e-3, 1 - 1e-5)
    logits = np.log(exposures / (1 - exposures))
    thetas = np.repeat(schedule.theta[None, :], len(logits), axis=0)
    thetas[:, :schedule.knots] = logits[:, None]
    peaks, finals, lost = simulate(schedule.population(thetas, track.market), *track.ranges['train'], rules)
    scores = fitness(peaks, finals, lost, weights, rules.initial_cash)
    k = int(np.argmax(scores))

    env = TradingEnv(track.market, *track.ranges['train'], rules)
    history, _ = run_episode(schedule, env)
    last_year = float(track.market.years[track.ranges['train'][0] + len(history) - 1])
    z = min(last_year / schedule.horizon_years * (schedule.knots - 1), schedule.knots - 1)
    last_knot = min(int(np.ceil(z)), schedule.knots - 1)
    return {'source': 'constant-exposure grid on the training period only',
            'continuation_exposure': float(exposures[k]),
            'continuation_logit': float(logits[k]),
            'grid': [{'exposure': float(e), 'peak': float(p), 'final': float(f), 'lost': bool(l),
                      'fitness': float(s)} for e, p, f, l, s in zip(CONTINUATION_GRID, peaks, finals, lost, scores)],
            'schedule_last_training_date': str(history.Date.iloc[-1]),
            'schedule_end_reason': str(history.reason.iloc[-1]),
            'schedule_knots_trained': last_knot + 1,
            'schedule_knots_total': schedule.knots}
