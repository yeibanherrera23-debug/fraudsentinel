"""Agente FraudSentinel: ciclo percepción -> decisión -> acción.

Tipo de agente: basado en utilidad, con estado interno (perfil por tarjeta) y componente
de aprendizaje (reentrenamiento con resultados confirmados).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import AgentConfig
from .environment import V_COLUMNS, RawTransaction, TransactionEnvironment
from .event_log import EventLog
from .risk_model import RiskModel
from .services import ExternalServices
from .utility import APPROVE, BLOCK, ESCALATE, UtilityFunction

CONTEXT_NAMES = ["amount_log", "amount_z", "new_device", "foreign_country", "new_category", "history_len"]
NEW_DEVICE_IDX = len(V_COLUMNS) + CONTEXT_NAMES.index("new_device")


class CardProfile:
    """Estado interno del agente para una tarjeta (memoria de largo plazo)."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.devices: set = set()
        self.countries: set = set()
        self.categories: set = set()

    def update(self, amount, device, country, category) -> None:
        x = math.log1p(amount)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)
        if device is not None:
            self.devices.add(device)
        if country is not None:
            self.countries.add(country)
        if category is not None:
            self.categories.add(category)

    def amount_z(self, amount: float, min_history: int) -> float:
        if self.n < min_history:
            return float("nan")
        std = max(math.sqrt(self.m2 / (self.n - 1)), 0.25)
        return (math.log1p(amount) - self.mean) / std


@dataclass
class Percept:
    tx_id: int
    time_s: float
    card_id: int
    amount: float
    device_id: Optional[str]
    country: Optional[str]
    merchant_category: Optional[str]
    features: np.ndarray
    blacklist_hit: bool


@dataclass
class Decision:
    action: str
    p_fraud: Optional[float]
    utilities: dict
    reason: str


@dataclass
class ActionResult:
    tx_id: int
    first_action: str
    final_status: str  # APPROVED | DECLINED
    case_verdict: Optional[int] = None


class FraudSentinelAgent:
    def __init__(self, model: RiskModel, utility: UtilityFunction, services: ExternalServices,
                 log: EventLog, config: AgentConfig = AgentConfig()):
        self.model = model
        self.utility = utility
        self.services = services
        self.log = log
        self.config = config
        self.profiles: dict[int, CardProfile] = defaultdict(CardProfile)
        self._train_X: Optional[np.ndarray] = None
        self._train_y: Optional[np.ndarray] = None
        self._exp_X: list = []
        self._exp_y: list = []
        self._pending: dict[int, tuple] = {}
        self._since_retrain = 0
        self.retrain_count = 0

    # ------------------------------------------------------------------ ciclo del agente
    def step(self, raw: RawTransaction) -> ActionResult:
        """Un ciclo completo: percepción -> decisión -> acción."""
        percept = self.perceive(raw)
        decision = self.decide(percept)
        return self.act(percept, decision)

    # ------------------------------------------------------------------ percepción
    def _context_features(self, raw: RawTransaction, profile: CardProfile) -> np.ndarray:
        nan = float("nan")
        reliable = profile.n >= self.config.min_history

        def flag(value, known):
            if value is None or not reliable:
                return nan
            return float(value not in known)

        return np.array([
            math.log1p(raw.amount),
            profile.amount_z(raw.amount, self.config.min_history),
            flag(raw.device_id, profile.devices),
            flag(raw.country, profile.countries),
            flag(raw.merchant_category, profile.categories),
            float(profile.n),
        ])

    def perceive(self, raw: RawTransaction) -> Percept:
        profile = self.profiles[raw.card_id]
        context = self._context_features(raw, profile)
        features = np.concatenate([raw.v, context])
        hit = raw.device_id is not None and self.services.blacklist.contains(raw.device_id)
        percept = Percept(raw.tx_id, raw.time_s, raw.card_id, raw.amount, raw.device_id,
                          raw.country, raw.merchant_category, features, hit)
        self.log.event(
            "percept", raw.tx_id, raw.time_s, card_id=raw.card_id, amount=raw.amount,
            device_id=raw.device_id, country=raw.country, merchant_category=raw.merchant_category,
            context=dict(zip(CONTEXT_NAMES, context)), blacklist_hit=hit,
        )
        return percept

    # ------------------------------------------------------------------ decisión
    def decide(self, percept: Percept) -> Decision:
        if percept.blacklist_hit:
            decision = Decision(BLOCK, None, {}, "dispositivo en lista negra")
        else:
            p = float(self.model.predict_proba(percept.features.reshape(1, -1))[0])
            action, utilities = self.utility.choose(p, percept.amount)
            decision = Decision(action, p, utilities, "máxima utilidad esperada")
        self.log.event(
            "decision", percept.tx_id, percept.time_s, action=decision.action,
            p_fraud=decision.p_fraud, utilities=decision.utilities, reason=decision.reason,
        )
        return decision

    # ------------------------------------------------------------------ acción
    def act(self, percept: Percept, decision: Decision) -> ActionResult:
        svc, tx = self.services, percept.tx_id
        verdict = None
        if decision.action == APPROVE:
            svc.gateway.authorize(tx)
            final = "APPROVED"
        elif decision.action == BLOCK:
            svc.gateway.decline(tx)
            svc.notifier.send("antifraude", f"Transacción {tx} bloqueada: {decision.reason}")
            final = "DECLINED"
        else:  # ESCALATE
            svc.gateway.hold(tx)
            summary = {"tx_id": tx, "amount": percept.amount, "p_fraud": decision.p_fraud,
                       "utilities": decision.utilities}
            verdict = svc.cases.open_case(tx, summary)
            if verdict == 1:
                svc.gateway.decline(tx)
                final = "DECLINED"
            else:
                svc.gateway.authorize(tx)
                final = "APPROVED"

        if final == "APPROVED":
            self.profiles[percept.card_id].update(
                percept.amount, percept.device_id, percept.country, percept.merchant_category)

        self._pending[tx] = (percept.features, percept.device_id)
        self.log.event("action", tx, percept.time_s, first_action=decision.action,
                       final_status=final, case_verdict=verdict)
        return ActionResult(tx, decision.action, final, verdict)

    # ------------------------------------------------------------------ aprendizaje
    def warm_up(self, env: TransactionEnvironment) -> None:
        """Entrena el modelo con historial etiquetado y construye los perfiles iniciales."""
        X, y = [], []
        for raw in env:
            label = env.reveal_outcome(raw.tx_id)
            profile = self.profiles[raw.card_id]
            X.append(np.concatenate([raw.v, self._context_features(raw, profile)]))
            y.append(label)
            if label == 0:
                profile.update(raw.amount, raw.device_id, raw.country, raw.merchant_category)
        self._train_X, self._train_y = np.asarray(X), np.asarray(y)
        self.model.fit(self._train_X, self._train_y)
        self.log.event("warm_up", None, n_samples=len(y), n_fraud=int(self._train_y.sum()),
                       beta=self.model.beta)

    def observe_outcome(self, tx_id: int, label: int) -> None:
        """Recibe el resultado real de una transacción ya decidida y aprende de él."""
        features, device = self._pending.pop(tx_id)
        self._exp_X.append(features)
        self._exp_y.append(label)
        if label == 1 and device is not None and features[NEW_DEVICE_IDX] == 1.0:
            self.services.blacklist.add(device)
        self.log.event("feedback", tx_id, label=label)
        self._since_retrain += 1
        if self._since_retrain >= self.config.retrain_every:
            self._retrain()

    def _retrain(self) -> None:
        X = np.vstack([self._train_X, np.asarray(self._exp_X)])
        y = np.concatenate([self._train_y, np.asarray(self._exp_y)])
        self.model.fit(X, y)
        self._since_retrain = 0
        self.retrain_count += 1
        self.log.event("retrain", None, n_samples=len(y), n_fraud=int(y.sum()), beta=self.model.beta)
