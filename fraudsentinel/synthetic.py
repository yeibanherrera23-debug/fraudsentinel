"""Generador de datos sintéticos con el mismo esquema del conjunto de datos de referencia.

Solo sirve para verificar que el código corre de punta a punta; sus resultados no se
reportan como evidencia del desempeño del agente.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .environment import V_COLUMNS


def make_synthetic_creditcard(n_rows: int = 20_000, fraud_rate: float = 0.02, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n_rows) < fraud_rate
    v = rng.normal(0.0, 1.0, size=(n_rows, len(V_COLUMNS)))
    shift = np.zeros(len(V_COLUMNS))
    shift[[0, 2, 3, 9, 13]] = [-2.0, -2.5, 1.5, -2.0, -2.5]
    v[is_fraud] += shift
    amount = np.where(
        is_fraud,
        rng.lognormal(mean=4.0, sigma=1.5, size=n_rows),
        rng.lognormal(mean=3.4, sigma=1.3, size=n_rows),
    ).round(2)
    time_s = np.sort(rng.uniform(0, 172_792, size=n_rows))
    df = pd.DataFrame(v, columns=V_COLUMNS)
    df.insert(0, "Time", time_s)
    df["Amount"] = amount
    df["Class"] = is_fraud.astype(int)
    return df
