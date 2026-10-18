"""Desglosa los resultados de una ejecución a partir de su registro de eventos.

Uso:
    python tools/analyze_run.py logs/run_<fecha>

Clasifica cada transacción según cómo la trató el agente (aprobada, escalada, bloqueada por
lista negra o bloqueada por el modelo) y la cruza con el resultado real confirmado.
Solo se consideran las transacciones con confirmación recibida (feedback).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

BLACKLIST_REASON = "dispositivo en lista negra"
CATEGORIES = ["APPROVE", "ESCALATE", "BLOCK_MODEL", "BLOCK_BLACKLIST"]


def load_events(run_dir: Path):
    decisions, labels = {}, {}
    with (run_dir / "events.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            event = json.loads(line)
            kind = event["event"]
            if kind == "decision":
                decisions[event["tx_id"]] = (event["action"], event.get("reason"))
            elif kind == "feedback":
                labels[event["tx_id"]] = event["label"]
    return decisions, labels


def categorize(action: str, reason: str | None) -> str:
    if action == "BLOCK":
        return "BLOCK_BLACKLIST" if reason == BLACKLIST_REASON else "BLOCK_MODEL"
    return action


def analyze(run_dir: Path) -> None:
    decisions, labels = load_events(run_dir)
    counts = {0: Counter(), 1: Counter()}
    for tx_id, label in labels.items():
        if tx_id in decisions:
            counts[label][categorize(*decisions[tx_id])] += 1

    fraud_total = sum(counts[1].values())
    legit_total = sum(counts[0].values())
    print(f"Transacciones con confirmación: {fraud_total + legit_total} "
          f"({fraud_total} fraudes, {legit_total} legítimas)\n")
    print(f"{'Tratamiento del agente':<26}{'Fraudes':>10}{'Legítimas':>12}")
    for category in CATEGORIES:
        print(f"{category:<26}{counts[1][category]:>10}{counts[0][category]:>12}")

    if fraud_total:
        print("\nDe los fraudes confirmados:")
        for category in CATEGORIES:
            share = counts[1][category] / fraud_total
            print(f"  {category:<18}{counts[1][category]:>5}  ({share:.1%})")
        flagged = fraud_total - counts[1]["APPROVE"]
        print(f"  Recall (marcados):  {flagged / fraud_total:.4f}")
        without_blacklist = flagged - counts[1]["BLOCK_BLACKLIST"]
        print(f"  Recall sin contar los bloqueos por lista negra: {without_blacklist / fraud_total:.4f}")
    if legit_total:
        flagged_legit = legit_total - counts[0]["APPROVE"]
        print(f"\nFPR (marcadas): {flagged_legit / legit_total:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="Carpeta de la ejecución, por ejemplo logs/run_20261017T003631Z")
    analyze(Path(parser.parse_args().run_dir))
