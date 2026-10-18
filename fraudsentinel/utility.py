"""Función de utilidad: elige la acción con mayor utilidad esperada."""
from __future__ import annotations

from .config import CostConfig

APPROVE = "APPROVE"
BLOCK = "BLOCK"
ESCALATE = "ESCALATE"


class UtilityFunction:
    """Utilidad = -(costo esperado) de cada acción.

    - APPROVE:  p * M          (se pierde el monto M si resulta fraude)
    - BLOCK:    (1 - p) * F    (fricción F si la transacción era legítima)
    - ESCALATE: R              (costo de la revisión humana, que se asume acertada)
    """

    def __init__(self, costs: CostConfig):
        self.costs = costs

    def expected_costs(self, p_fraud: float, amount: float) -> dict[str, float]:
        return {
            APPROVE: p_fraud * amount,
            ESCALATE: self.costs.review_cost,
            BLOCK: (1.0 - p_fraud) * self.costs.friction_cost,
        }

    def utilities(self, p_fraud: float, amount: float) -> dict[str, float]:
        return {a: -c for a, c in self.expected_costs(p_fraud, amount).items()}

    def choose(self, p_fraud: float, amount: float) -> tuple[str, dict[str, float]]:
        utilities = self.utilities(p_fraud, amount)
        best = max(utilities, key=utilities.get)
        return best, utilities
