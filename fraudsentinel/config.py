"""Parámetros configurables del agente y del entorno simulado."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CostConfig:
    """Costos (EUR) que alimentan la función de utilidad del agente."""

    friction_cost: float = 20.0  # F: costo de bloquear una transacción legítima
    review_cost: float = 5.0  # R: costo de una revisión humana


@dataclass(frozen=True)
class EnvironmentConfig:
    """Parámetros del simulador del entorno (clientes, defraudadores, sensores, analista)."""

    seed: int = 42
    n_cards: int = 500
    n_amount_tiers: int = 5
    fraud_device_pool: int = 100
    # Cliente legítimo
    p_legit_same_tier: float = 0.90
    p_legit_new_device: float = 0.03
    p_legit_foreign_country: float = 0.05
    p_legit_preferred_category: float = 0.80
    # Defraudador (agente adversario)
    p_fraud_new_device: float = 0.70
    p_fraud_foreign_country: float = 0.50
    p_fraud_same_tier: float = 0.0  # prob. de que el fraude use una tarjeta acorde a su monto
    p_fraud_preferred_category: float = 0.0  # prob. de que use una categoría habitual de la tarjeta
    # Fallas de sensores y analista humano
    p_sensor_dropout: float = 0.0
    analyst_accuracy: float = 1.0


@dataclass(frozen=True)
class AgentConfig:
    """Parámetros internos del agente."""

    min_history: int = 5  # transacciones mínimas para confiar en el perfil de una tarjeta
    negatives_per_positive: int = 20  # submuestreo de la clase legítima en el entrenamiento
    feedback_delay: int = 200  # transacciones que tarda en llegar la confirmación real
    retrain_every: int = 3000  # confirmaciones acumuladas antes de reentrenar el modelo
