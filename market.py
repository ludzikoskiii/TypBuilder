"""Kursy -> prawdopodobieństwa (bez marży), value i stawki.

Wszystkie funkcje przyjmują kursy dziesiętne. Funkcje "fair_*" działają na tablicach
o kształcie (..., k), gdzie ostatnia oś to wyniki jednego rynku (np. 1/X/2).
"""
from __future__ import annotations

import numpy as np


def implied(odds: np.ndarray) -> np.ndarray:
    """Prawdopodobieństwa implikowane 1/kurs (z marżą)."""
    return 1.0 / np.asarray(odds, dtype=float)


def overround(odds: np.ndarray) -> np.ndarray:
    """Suma prawdopodobieństw implikowanych (1.05 = marża 5%)."""
    return implied(odds).sum(axis=-1)


def fair_proportional(odds: np.ndarray) -> np.ndarray:
    """Usunięcie marży metodą proporcjonalną: p_i = (1/o_i) / sum(1/o_j)."""
    pi = implied(odds)
    return pi / pi.sum(axis=-1, keepdims=True)


def _shin_probs(pi: np.ndarray, beta: np.ndarray, z: np.ndarray) -> np.ndarray:
    z = z[..., None]
    return (np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / beta[..., None]) - z) / (2 * (1 - z))


def fair_shin(odds: np.ndarray, return_z: bool = False, iters: int = 80):
    """Usunięcie marży metodą Shina (1992/93).

    p_i(z) = (sqrt(z^2 + 4(1-z) pi_i^2 / beta) - z) / (2(1-z)), gdzie pi_i = 1/o_i,
    beta = sum(pi_i), a z (udział "insiderów") wyznaczamy tak, by sum(p_i) = 1
    (bisekcja, wektorowo). Bez marży (beta <= 1) zwracamy metodę proporcjonalną.
    """
    pi = implied(odds)
    beta = pi.sum(axis=-1)
    lo = np.zeros_like(beta)
    hi = np.full_like(beta, 0.999)
    for _ in range(iters):
        mid = (lo + hi) / 2
        s = _shin_probs(pi, beta, mid).sum(axis=-1)
        too_big = s > 1.0
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
    z = (lo + hi) / 2
    p = _shin_probs(pi, beta, z)
    p = p / p.sum(axis=-1, keepdims=True)  # usuwa resztkowy błąd bisekcji
    no_margin = beta <= 1.0
    if np.any(no_margin):
        p = np.where(no_margin[..., None], pi / beta[..., None], p)
        z = np.where(no_margin, 0.0, z)
    return (p, z) if return_z else p


def fair_probs(odds: np.ndarray, method: str = "shin") -> np.ndarray:
    """Prawdopodobieństwa bez marży wybraną metodą ('shin' lub 'proportional')."""
    if method == "shin":
        return fair_shin(odds)
    if method == "proportional":
        return fair_proportional(odds)
    raise ValueError(f"Nieznana metoda usuwania marży: {method}")


def value(p_model: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Oczekiwany zysk na jednostkę stawki: p_model * kurs - 1."""
    return np.asarray(p_model, dtype=float) * np.asarray(odds, dtype=float) - 1.0


def is_value(p_model: np.ndarray, odds: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Czy typ przekracza próg value (np. 0.05 = 5%)."""
    return value(p_model, odds) > threshold


def kelly_fraction(p_model: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Pełny Kelly: f* = (p*o - 1) / (o - 1), obcięty do 0 dla ujemnego value."""
    o = np.asarray(odds, dtype=float)
    return np.clip(value(p_model, o) / (o - 1.0), 0.0, None)


def kelly_stake(p_model: np.ndarray, odds: np.ndarray, bankroll: float,
                fraction: float = 0.25, max_pct: float = 0.02) -> np.ndarray:
    """Stawka ułamkowym Kellym: bankroll * min(fraction * f*, max_pct)."""
    f = np.minimum(fraction * kelly_fraction(p_model, odds), max_pct)
    return bankroll * f


def flat_stake(p_model: np.ndarray, odds: np.ndarray, stake: float = 1.0,
               threshold: float = 0.05) -> np.ndarray:
    """Stawka płaska dla typów value, 0 dla pozostałych."""
    return np.where(is_value(p_model, odds, threshold), stake, 0.0)
