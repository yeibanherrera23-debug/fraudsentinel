"""Modelo de riesgo: estima la probabilidad de fraude de una transacción."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class RiskModel:
    """Clasificador entrenado con submuestreo de la clase legítima.

    El submuestreo distorsiona las probabilidades; se corrigen con el ajuste de prior
    p = ps / (ps + (1 - ps) / beta), donde beta es la fracción de legítimas conservadas.
    """

    def __init__(self, negatives_per_positive: int = 20, seed: int = 0):
        self.negatives_per_positive = negatives_per_positive
        self.seed = seed
        self.beta = 1.0
        self.fitted = False
        self._keep: np.ndarray | None = None  # columnas con información (None = todas)
        self._model = self._new_estimator()

    def _new_estimator(self):
        return HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=self.seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        positives = np.flatnonzero(y == 1)
        negatives = np.flatnonzero(y == 0)
        if len(positives) == 0 or len(negatives) == 0:
            raise ValueError("El entrenamiento requiere ejemplos de ambas clases.")
        kept = min(len(negatives), self.negatives_per_positive * len(positives))
        negatives_kept = rng.choice(negatives, size=kept, replace=False)
        idx = np.concatenate([positives, negatives_kept])
        rng.shuffle(idx)
        self.beta = kept / len(negatives)
        X_train = X[idx]
        keep = self._informative_columns(X_train)
        self._keep = None if keep.all() else np.flatnonzero(keep)
        if self._keep is not None:
            X_train = X_train[:, self._keep]
        self._model = self._new_estimator()
        self._model.fit(X_train, y[idx])
        self.fitted = True

    @staticmethod
    def _informative_columns(X: np.ndarray) -> np.ndarray:
        """Columnas con al menos dos valores distintos (sin contar los vacíos).

        Una columna que llega siempre vacía o constante (por ejemplo, un sensor caído) no
        aporta información y algunas versiones de scikit-learn fallan al procesarla.
        """
        return np.array([np.unique(col[~np.isnan(col)]).size >= 2 for col in X.T])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilidad de fraude (corregida por el submuestreo) para cada fila de X."""
        if self._keep is not None:
            X = X[:, self._keep]
        ps = self._model.predict_proba(X)[:, 1]
        return ps / (ps + (1.0 - ps) / self.beta)
