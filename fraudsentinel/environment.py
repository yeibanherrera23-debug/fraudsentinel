"""Entorno simulado: flujo de transacciones, clientes, defraudadores, sensores y analista.

Las columnas V1..V28, Amount y Class provienen del conjunto de datos. El identificador de
tarjeta, el comercio, el dispositivo y el país NO existen en ese conjunto; el simulador los
genera con aleatoriedad (ambiente estocástico) y modela un defraudador que actúa de forma
distinta a un cliente legítimo (ambiente multiagente).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from .config import EnvironmentConfig

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
COUNTRIES = ["FR", "DE", "ES", "IT", "BE", "NL", "PT", "AT"]
CATEGORIES = [
    "supermercado", "combustible", "restaurante", "viajes",
    "electronica", "salud", "ropa", "entretenimiento",
]


@dataclass
class Card:
    card_id: int
    tier: int
    home_country: str
    devices: list
    categories: list


@dataclass
class RawTransaction:
    """Lo que llega a los sensores del agente (sin la etiqueta real)."""

    tx_id: int
    time_s: float
    card_id: int
    amount: float
    v: np.ndarray
    merchant_category: Optional[str]
    country: Optional[str]
    device_id: Optional[str]


class CardPool:
    """Conjunto de tarjetas simuladas, con un nivel de gasto habitual (tier) cada una."""

    def __init__(self, amounts: np.ndarray, config: EnvironmentConfig, rng: np.random.Generator):
        quantiles = np.linspace(0, 1, config.n_amount_tiers + 1)[1:-1]
        self.bounds = np.quantile(np.log1p(amounts), quantiles)
        self.cards: list[Card] = []
        for cid in range(config.n_cards):
            n_devices = int(rng.integers(1, 4))
            self.cards.append(
                Card(
                    card_id=cid,
                    tier=cid % config.n_amount_tiers,
                    home_country=COUNTRIES[int(rng.integers(len(COUNTRIES)))],
                    devices=[f"D{cid}-{k}" for k in range(n_devices)],
                    categories=[CATEGORIES[i] for i in rng.choice(len(CATEGORIES), 3, replace=False)],
                )
            )
        self.by_tier = {t: [c for c in self.cards if c.tier == t] for t in range(config.n_amount_tiers)}

    def tier_of(self, amount: float) -> int:
        return int(np.searchsorted(self.bounds, np.log1p(amount)))


class TransactionEnvironment:
    """Genera el flujo de transacciones y guarda la verdad (oculta para el agente)."""

    def __init__(self, df: pd.DataFrame, pool: CardPool, config: EnvironmentConfig,
                 seed: int, id_offset: int = 0):
        df = df.reset_index(drop=True)
        self._time = df["Time"].to_numpy(dtype=float)
        self._amount = df["Amount"].to_numpy(dtype=float)
        self._label = df["Class"].to_numpy(dtype=int)
        self._v = df[V_COLUMNS].to_numpy(dtype=float)
        self.pool = pool
        self.config = config
        self._rng = np.random.default_rng(seed)
        self._analyst_rng = np.random.default_rng(seed + 10_000)
        self._offset = id_offset
        self._truth: dict[int, int] = {}
        self._amounts: dict[int, float] = {}
        self._fraud_devices = [f"F{k:03d}" for k in range(config.fraud_device_pool)]
        self._fresh_device = 0

    def __len__(self) -> int:
        return len(self._label)

    def __iter__(self) -> Iterator[RawTransaction]:
        for i in range(len(self._label)):
            yield self._build(i)

    # ---- verdad y retroalimentación (no accesibles por los sensores) ----
    def reveal_outcome(self, tx_id: int) -> int:
        """Confirmación real posterior (reclamo del cliente / equipo antifraude)."""
        return self._truth[tx_id]

    def true_amount(self, tx_id: int) -> float:
        """Monto real de la transacción (para calcular costos)."""
        return self._amounts[tx_id]

    def analyst_review(self, tx_id: int) -> int:
        """Veredicto del analista humano; acierta con probabilidad analyst_accuracy."""
        truth = self._truth[tx_id]
        if self._analyst_rng.random() < self.config.analyst_accuracy:
            return truth
        return 1 - truth

    # ---- construcción de una transacción ----
    def _foreign_country(self, home: str) -> str:
        options = [c for c in COUNTRIES if c != home]
        return options[int(self._rng.integers(len(options)))]

    def _build(self, i: int) -> RawTransaction:
        cfg, rng, pool = self.config, self._rng, self.pool
        tx_id = self._offset + i
        amount = float(self._amount[i])
        is_fraud = int(self._label[i]) == 1
        self._truth[tx_id] = int(is_fraud)
        self._amounts[tx_id] = amount

        if is_fraud:
            if cfg.p_fraud_same_tier > 0 and rng.random() < cfg.p_fraud_same_tier:
                candidates = pool.by_tier[pool.tier_of(amount)]
            else:
                candidates = pool.cards
            card = candidates[int(rng.integers(len(candidates)))]
            if rng.random() < cfg.p_fraud_new_device:
                device = self._fraud_devices[int(rng.integers(len(self._fraud_devices)))]
            else:
                device = card.devices[int(rng.integers(len(card.devices)))]
            country = (self._foreign_country(card.home_country)
                       if rng.random() < cfg.p_fraud_foreign_country else card.home_country)
            if cfg.p_fraud_preferred_category > 0 and rng.random() < cfg.p_fraud_preferred_category:
                category = card.categories[int(rng.integers(len(card.categories)))]
            else:
                category = CATEGORIES[int(rng.integers(len(CATEGORIES)))]
        else:
            if rng.random() < cfg.p_legit_same_tier:
                candidates = pool.by_tier[pool.tier_of(amount)]
            else:
                candidates = pool.cards
            card = candidates[int(rng.integers(len(candidates)))]
            if rng.random() < cfg.p_legit_new_device:
                self._fresh_device += 1
                device = f"N{self._offset}-{self._fresh_device}"
            else:
                device = card.devices[int(rng.integers(len(card.devices)))]
            country = (self._foreign_country(card.home_country)
                       if rng.random() < cfg.p_legit_foreign_country else card.home_country)
            if rng.random() < cfg.p_legit_preferred_category:
                category = card.categories[int(rng.integers(len(card.categories)))]
            else:
                category = CATEGORIES[int(rng.integers(len(CATEGORIES)))]

        # Falla de sensores: un dato puede llegar vacío
        if rng.random() < cfg.p_sensor_dropout:
            device = None
        if rng.random() < cfg.p_sensor_dropout:
            country = None
        if rng.random() < cfg.p_sensor_dropout:
            category = None

        return RawTransaction(
            tx_id=tx_id, time_s=float(self._time[i]), card_id=card.card_id, amount=amount,
            v=self._v[i], merchant_category=category, country=country, device_id=device,
        )
