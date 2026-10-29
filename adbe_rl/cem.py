"""Cross-Entropy Method policy search with smoothing, elitism and restarts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Evaluate = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class CEMSettings:
    population: int
    generations: int
    init_mean: tuple[float, ...]
    init_std: float
    keep: float                       # weight of the old mean/std in the smoothed update
    std_floor: float                  # lower bound on std, against premature collapse
    weights: tuple[float, float]      # fitness = w_peak*log(peak/E0) + w_final*log(final/E0)
    restart_every: int = 0            # every k generations, resample 20% of the population widely
    restart_std: float = 4.

    @property
    def n_elite(self) -> int:
        return max(12, self.population // 10)


@dataclass
class SearchResult:
    best: np.ndarray            # best fitness seen
    best_peak_theta: np.ndarray  # highest peak assets seen
    best_peak: float
    archive: np.ndarray         # best member of every generation
    log: list[dict]


def cem_search(evaluate: Evaluate, settings: CEMSettings, seed: int, initial_cash: float,
               warm_start: Callable[[np.ndarray], None] | None = None) -> SearchResult:
    """Maximize episode fitness of the thetas sampled from N(mean, std^2)."""
    rng = np.random.default_rng(seed)
    mean = np.asarray(settings.init_mean, float)
    std = np.full(len(mean), settings.init_std)
    best, best_score = mean.copy(), -np.inf
    best_peak_theta, best_peak = mean.copy(), -np.inf
    archive, log = [], []
    w_peak, w_final = settings.weights
    for generation in range(settings.generations):
        x = rng.normal(mean, std, (settings.population, len(mean)))
        x[0] = best
        if settings.restart_every and generation % settings.restart_every == 0:
            k = settings.population // 5
            x[-k:] = rng.normal(0, settings.restart_std, (k, len(mean)))
        if warm_start is not None and generation == 0:
            warm_start(x)

        peaks, finals = evaluate(x)
        score = (w_peak * np.log(np.maximum(peaks, 1) / initial_cash)
                 + w_final * np.log(np.maximum(finals, 1) / initial_cash))
        elite = x[np.argsort(score)[-settings.n_elite:]]
        mean = settings.keep * mean + (1 - settings.keep) * elite.mean(axis=0)
        std = np.maximum(settings.std_floor, settings.keep * std + (1 - settings.keep) * elite.std(axis=0))

        k = int(np.argmax(score))
        if score[k] > best_score:
            best, best_score = x[k].copy(), float(score[k])
        k_peak = int(np.argmax(peaks))
        if peaks[k_peak] > best_peak:
            best_peak_theta, best_peak = x[k_peak].copy(), float(peaks[k_peak])
        archive.append(x[k].copy())
        log.append({'seed': seed, 'generation': generation, 'best_fitness': best_score,
                    'best_peak': best_peak, 'median_peak': float(np.median(peaks))})
    return SearchResult(best, best_peak_theta, best_peak, np.asarray(archive), log)
