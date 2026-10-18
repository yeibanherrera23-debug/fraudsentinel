"""Servicios externos que el agente invoca como herramientas (tool calling)."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable


class Blacklist:
    """Lista negra de dispositivos asociados a fraude confirmado."""

    def __init__(self):
        self._devices: set[str] = set()

    def contains(self, device_id) -> bool:
        return device_id in self._devices

    def add(self, device_id) -> None:
        self._devices.add(device_id)

    def __len__(self) -> int:
        return len(self._devices)


class PaymentGateway:
    """Pasarela de pagos: autoriza, retiene o rechaza una transacción."""

    def __init__(self):
        self.status: dict[int, str] = {}

    def authorize(self, tx_id: int) -> None:
        self.status[tx_id] = "APPROVED"

    def decline(self, tx_id: int) -> None:
        self.status[tx_id] = "DECLINED"

    def hold(self, tx_id: int) -> None:
        self.status[tx_id] = "ON_HOLD"


class CaseManagement:
    """Sistema de casos del equipo antifraude; el analista humano emite el veredicto."""

    def __init__(self, analyst: Callable[[int], int]):
        self._analyst = analyst
        self.cases_opened = 0

    def open_case(self, tx_id: int, summary: dict) -> int:
        """Abre un caso y devuelve el veredicto del analista (1 = fraude, 0 = legítima)."""
        self.cases_opened += 1
        return self._analyst(tx_id)


class Notifier:
    """Canal de notificaciones internas del banco."""

    def __init__(self):
        self.count_by_channel: Counter = Counter()
        self.recent: deque = deque(maxlen=1000)

    def send(self, channel: str, message: str) -> None:
        self.count_by_channel[channel] += 1
        self.recent.append((channel, message))


@dataclass
class ExternalServices:
    blacklist: Blacklist
    gateway: PaymentGateway
    cases: CaseManagement
    notifier: Notifier
