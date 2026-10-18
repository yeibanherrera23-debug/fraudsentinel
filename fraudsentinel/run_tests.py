"""Ejecuta el plan de pruebas contra las medidas de desempeño.

Uso:
    python -m fraudsentinel.run_tests --data data/creditcard.csv --reps 5 --workers 2
"""
from __future__ import annotations

import argparse

from .experiments import MEASURES, SCENARIOS, run_test_plan


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plan de pruebas de FraudSentinel")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--data", help="Ruta al archivo creditcard.csv")
    src.add_argument("--synthetic", action="store_true", help="Datos sintéticos (solo verificación técnica)")
    p.add_argument("--rows", type=int, default=20_000, help="Filas sintéticas")
    p.add_argument("--reps", type=int, default=5, help="Repeticiones por escenario (mínimo 5)")
    p.add_argument("--seed-base", type=int, default=42, help="Primera semilla; las demás son consecutivas")
    p.add_argument("--workers", type=int, default=1, help="Ejecuciones en paralelo")
    p.add_argument("--max-test", type=int, default=None, help="Máximo de transacciones de operación")
    p.add_argument("--scenarios", default=None, help="Subconjunto de escenarios, por ejemplo N01,L01")
    p.add_argument("--out-dir", default="results/pruebas", help="Carpeta de resultados (reanudable)")
    p.add_argument("--list", action="store_true", help="Lista los escenarios y las medidas, y termina")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.list:
        print("Medidas de desempeño")
        for m in MEASURES:
            kind = "medida del diseño" if m.official else "complementaria"
            print(f"  {m.key} [{kind}]: {m.label}  {m.op} {m.threshold}  ({m.formula})")
        print("\nEscenarios")
        for s in SCENARIOS:
            print(f"  {s.id} [{s.kind}] {s.name}: {s.description}")
        return
    if not args.data and not args.synthetic:
        raise SystemExit("Indica --data RUTA o --synthetic.")
    source = ({"kind": "synthetic", "rows": args.rows} if args.synthetic
              else {"kind": "csv", "path": args.data})
    ids = [s.strip() for s in args.scenarios.split(",")] if args.scenarios else None
    run_test_plan(source, args.out_dir, reps=args.reps, seed_base=args.seed_base, workers=args.workers,
                  max_test=args.max_test, scenario_ids=ids)


if __name__ == "__main__":
    main()
