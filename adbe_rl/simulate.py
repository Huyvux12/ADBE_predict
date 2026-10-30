"""Vectorized episodes: a whole CEM population trades the same prices at once."""
from __future__ import annotations

import numpy as np

from .broker import Book, order_quantities
from .config import Rules
from .policies import Population, decide


def simulate(population: Population, start: int, end: int, rules: Rules
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run every member from 100% cash over sessions [start, end).

    Returns (peak, final, lost): peak and final assets, and whether the episode
    ended by a loss condition (assets < loss limit or cash <= cash limit).
    """
    market = population.market
    book = Book(len(population), rules)
    rows = np.arange(len(population))
    for t in range(start, end):
        buy, sell, mark = market.quotes(t, rules.sell_price)
        actions = decide(population.logits(t), population.controls,
                         book.cash, book.shares, book.idle, buy, rules)
        quantity = order_quantities(book.cash, book.shares, buy, rules)[rows, actions]
        book.step(quantity, buy, sell, mark)
        if not book.alive.any():
            break
    return book.peak, book.final, book.lost
