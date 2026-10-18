"""Registro (log) de ejecución: percepciones, decisiones, acciones y retroalimentación."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _sanitize(value):
    """Convierte tipos de numpy y NaN en valores serializables a JSON."""
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return _sanitize(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


class EventLog:
    """Escribe un evento por línea (JSON Lines) con marca de tiempo UTC.

    Modos:
    - ``full``: registra todos los eventos de todas las transacciones.
    - ``light``: registra el ciclo completo solo de las transacciones que el agente no aprobó,
      y la confirmación real solo de los fraudes y de las transacciones marcadas. Reduce el
      tamaño del archivo en varios órdenes de magnitud para ejecuciones repetidas.
    """

    CYCLE_EVENTS = ("percept", "decision", "action")

    def __init__(self, path, mode: str = "full"):
        if mode not in ("full", "light"):
            raise ValueError("mode debe ser 'full' o 'light'")
        self.mode = mode
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._buffer: list[dict] = []
        self._flagged: set = set()

    def _write(self, record: dict) -> None:
        self._fh.write(json.dumps(_sanitize(record), ensure_ascii=False) + "\n")

    def event(self, kind: str, tx_id, sim_time=None, **data):
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": kind,
            "tx_id": tx_id,
            "sim_time_s": sim_time,
            **data,
        }
        if self.mode == "full":
            self._write(record)
            return
        if kind in self.CYCLE_EVENTS:
            self._buffer.append(record)
            if kind == "action":
                if record["first_action"] != "APPROVE":
                    self._flagged.add(tx_id)
                    for buffered in self._buffer:
                        self._write(buffered)
                self._buffer.clear()
        elif kind == "feedback":
            if data.get("label") == 1 or tx_id in self._flagged:
                self._write(record)
            self._flagged.discard(tx_id)
        else:
            self._write(record)

    def close(self):
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
