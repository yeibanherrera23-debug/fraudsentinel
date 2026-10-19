"""Crea la muestra liviana que usa la aplicación web (data/demo_sample.csv.gz).

Uso:
    python tools/make_demo_sample.py data/creditcard.csv

Conserva todos los fraudes del conjunto original y una muestra aleatoria de transacciones
legítimas, ordenadas por tiempo. La aplicación publicada usa esta muestra porque el archivo
original (unos 144 MB) supera el límite de GitHub. Como la proporción de fraudes en la muestra
es mayor que en el conjunto completo, sus resultados no reemplazan los de las pruebas oficiales.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LEGIT_ROWS = 40_000
SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", help="Ruta de creditcard.csv")
    parser.add_argument("--out", default="data/demo_sample.csv.gz")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(n=LEGIT_ROWS, random_state=SEED)
    sample = pd.concat([fraud, legit]).sort_values("Time").reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False, compression="gzip")
    print(f"{len(sample)} filas ({len(fraud)} fraudes) -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
