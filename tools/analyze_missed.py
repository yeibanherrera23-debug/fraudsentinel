"""Lista los fraudes que el agente no marcó (aprobó) en una ejecución con registro completo.

Uso:
    python tools/analyze_missed.py logs/run_<fecha>

Muestra, por cada fraude aprobado, su monto y la probabilidad de fraude estimada por el
modelo, para analizar por qué no fue detectado.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def analyze(run_dir: Path) -> None:
    events = []
    with (run_dir / "events.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            events.append(json.loads(line))

    decisions = {e["tx_id"]: e for e in events if e["event"] == "decision"}
    percepts = {e["tx_id"]: e for e in events if e["event"] == "percept"}
    labels = {e["tx_id"]: e["label"] for e in events if e["event"] == "feedback"}
    frauds = [t for t, y in labels.items() if y == 1 and t in decisions]
    missed = [t for t in frauds if decisions[t]["action"] == "APPROVE"]

    print(f"Fraudes confirmados: {len(frauds)}; aprobados por el agente: {len(missed)}\n")
    for tx_id in missed:
        d, p = decisions[tx_id], percepts.get(tx_id, {})
        p_fraud = d.get("p_fraud")
        p_txt = f"{p_fraud:.4f}" if p_fraud is not None else "n/d"
        print(f"  {tx_id}: monto={p.get('amount', 'n/d')}  p_fraude={p_txt}  "
              f"nuevo_dispositivo={p.get('context', {}).get('new_device', 'n/d')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir")
    analyze(Path(parser.parse_args().run_dir))
