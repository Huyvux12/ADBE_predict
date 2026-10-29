"""Training pipeline for one track. Only fit/validation/train sessions are used."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cem import CEMSettings, cem_search
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
    peaks, finals = simulate(indicator.population(candidates, track.market), *track.ranges['validation'], rules)
    k = int(np.argmax(finals))
    save_policy(IndicatorPolicy(candidates[k], rules), folder / 'models/cem_validation.json')
    pd.DataFrame({'candidate': np.arange(len(finals)), 'validation_peak': peaks,
                  'validation_final': finals}).to_csv(folder / 'learning/validation_selection.csv', index=False)

    # 2. Indicator policy fitted on all 8 years, highest training peak.
    trained = search(indicator, indicator_settings(config), 'train')
    champion = max(trained, key=lambda r: r.best_peak)
    save_policy(IndicatorPolicy(champion.best_peak_theta, rules), folder / 'models/cem_train_peak.json')

    # 3. Time-conditioned allocation schedule, highest training peak.
    schedules = search(temporal, temporal_settings(config), 'train', fully_invested_candidate)
    champion = max(schedules, key=lambda r: r.best_peak)
    schedule = TemporalPolicy(champion.best_peak_theta, rules, KNOTS, HORIZON_YEARS)
    save_policy(schedule, folder / 'models/cem_temporal.json')

    # 4. Continuation: keep the exposure of the final training state after the horizon.
    continuation = fit_continuation(schedule, track)
    save_policy(schedule.with_continuation(continuation['continuation_logit']),
                folder / 'models/cem_temporal_continuation.json')
    (folder / 'learning/continuation.json').write_text(json.dumps(continuation, indent=2), encoding='utf-8')

    for label, results in [('indicator_fit', fitted), ('indicator_train', trained), ('temporal_train', schedules)]:
        pd.DataFrame([row for r in results for row in r.log]).to_csv(folder / f'learning/{label}.csv', index=False)


def fit_continuation(schedule: TemporalPolicy, track: Track) -> dict:
    """Logit of the stock exposure at the end of the training replay; test data is never read."""
    env = TradingEnv(track.market, *track.ranges['train'], schedule.rules)
    history, _ = run_episode(schedule, env)
    last = history.iloc[-1]
    exposure = float(np.clip(last.shares * last.mark_price / max(last.assets, 1e-8), 1e-5, 1 - 1e-5))
    return {'source': 'training replay only', 'last_training_date': last.Date,
            'last_training_exposure': exposure, 'continuation_logit': float(np.log(exposure / (1 - exposure)))}
