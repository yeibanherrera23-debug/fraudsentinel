"""Plan de pruebas: escenarios, medidas de desempeño con umbrales y ejecución repetida."""
from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AgentConfig, CostConfig, EnvironmentConfig
from .run_simulation import load_data, run_experiment
from .synthetic import make_synthetic_creditcard


@dataclass(frozen=True)
class Measure:
    key: str
    label: str
    formula: str
    op: str  # ">=" o "<"
    threshold: float
    official: bool = True  # False: medida complementaria (mismo umbral, solo como referencia)

    def passes(self, value: float) -> bool:
        return value >= self.threshold if self.op == ">=" else value < self.threshold


# Umbrales fijados en el diseño, antes de ejecutar las pruebas. Las dos primeras son las medidas
# del diseño (el agente "detecta" cuando marca la transacción); las otras dos son complementarias
# y miden el resultado final del sistema, es decir, después de la revisión del analista humano.
MEASURES = [
    Measure("recall_flagged", "Recall (sensibilidad)",
            "fraudes marcados (bloqueados o escalados) / fraudes totales", ">=", 0.80),
    Measure("fpr_flagged", "Tasa de falsos positivos",
            "legítimas marcadas (bloqueadas o escaladas) / legítimas totales", "<", 0.05),
    Measure("recall_declined_final", "Recall final (complementaria)",
            "fraudes finalmente rechazados / fraudes totales", ">=", 0.80, official=False),
    Measure("fpr_declined_final", "Falsos positivos finales (complementaria)",
            "legítimas finalmente rechazadas / legítimas totales", "<", 0.05, official=False),
]


@dataclass(frozen=True)
class Scenario:
    id: str
    kind: str  # normal | limite | falla
    name: str
    description: str
    env: dict = field(default_factory=dict)
    agent: dict = field(default_factory=dict)
    costs: dict = field(default_factory=dict)


SCENARIOS = [
    Scenario("N01", "normal", "Línea base", "Parámetros por defecto del agente y del simulador."),
    Scenario("N02", "normal", "Analista con 90 % de acierto",
             "El analista humano se equivoca en 1 de cada 10 revisiones.",
             env={"analyst_accuracy": 0.9}),
    Scenario("N03", "normal", "Sensores con 10 % de datos faltantes",
             "Cada sensor (dispositivo, país, comercio) deja de reportar en el 10 % de las transacciones.",
             env={"p_sensor_dropout": 0.10}),
    Scenario("L01", "limite", "Defraudador sin señales de contexto",
             "El defraudador se comporta como un cliente legítimo en dispositivo, país, comercio y nivel "
             "de gasto; solo las variables V1 a V28 lo distinguen.",
             env={"p_fraud_new_device": 0.03, "p_fraud_foreign_country": 0.05,
                  "p_fraud_same_tier": 0.90, "p_fraud_preferred_category": 0.80}),
    Scenario("L02", "limite", "Defraudador con señales débiles",
             "El defraudador imita parcialmente el comportamiento habitual del cliente.",
             env={"p_fraud_new_device": 0.15, "p_fraud_foreign_country": 0.15,
                  "p_fraud_same_tier": 0.50, "p_fraud_preferred_category": 0.40}),
    Scenario("L03", "limite", "Costo de fricción alto (F = 60)",
             "Bloquear a un cliente legítimo cuesta el triple que en la línea base.",
             costs={"friction_cost": 60.0}),
    Scenario("L04", "limite", "Costo de revisión bajo (R = 1)",
             "Escalar a un analista cuesta una quinta parte que en la línea base.",
             costs={"review_cost": 1.0}),
    Scenario("L05", "limite", "Sin reentrenamiento del modelo",
             "El modelo no se reentrena con los resultados confirmados (la lista negra sí se actualiza).",
             agent={"retrain_every": 10**9}),
    Scenario("L06", "limite", "Confirmaciones muy tardías",
             "El resultado real llega 5.000 transacciones después de la decisión.",
             agent={"feedback_delay": 5000}),
    Scenario("F01", "falla", "Falla de sensores del 30 %",
             "Cada sensor (dispositivo, país, comercio) deja de reportar en el 30 % de las transacciones.",
             env={"p_sensor_dropout": 0.30}),
    Scenario("F02", "falla", "Sensores de contexto caídos",
             "Ningún sensor de dispositivo, país o comercio entrega datos; el agente opera solo con V1 a V28 y el monto.",
             env={"p_sensor_dropout": 1.0}),
    Scenario("F03", "falla", "Analista poco confiable (70 %)",
             "El analista humano acierta solo en el 70 % de las revisiones.",
             env={"analyst_accuracy": 0.7}),
]

_DATA_CACHE: dict = {}


def _get_data(source: dict) -> pd.DataFrame:
    key = json.dumps(source, sort_keys=True)
    if key not in _DATA_CACHE:
        if source["kind"] == "synthetic":
            _DATA_CACHE[key] = make_synthetic_creditcard(source["rows"], seed=0)
        else:
            _DATA_CACHE[key] = load_data(source["path"])
    return _DATA_CACHE[key]


def _run_one(task: dict) -> dict:
    """Ejecuta un escenario con una semilla; nunca lanza excepciones (devuelve el error)."""
    scenario: Scenario = task["scenario"]
    run_dir = Path(task["run_dir"])
    base = {"scenario": scenario.id, "kind": scenario.kind, "seed": task["seed"],
            "evidence": str(run_dir.as_posix())}
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        return {**base, "status": "ok", "resumed": True, **json.loads(metrics_file.read_text(encoding="utf-8"))}
    try:
        df = _get_data(task["source"])
        env_cfg = replace(EnvironmentConfig(seed=task["seed"]), **scenario.env)
        agent_cfg = replace(AgentConfig(), **scenario.agent)
        costs = replace(CostConfig(), **scenario.costs)
        summary = run_experiment(df, env_cfg, agent_cfg, costs, run_dir,
                                 max_test=task["max_test"], log_mode="light", verbose=False)
        return {**base, "status": "ok", "resumed": False, **summary}
    except Exception as exc:  # noqa: BLE001 - se reporta como fallo del caso de prueba
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return {**base, "status": f"error: {type(exc).__name__}: {exc} (detalle en error.txt)",
                "resumed": False}


def _plan_dict(scenarios, reps: int, seed_base: int, source: dict, max_test) -> dict:
    return {
        "measures": [asdict(m) for m in MEASURES],
        "scenarios": [asdict(s) for s in scenarios],
        "repetitions": reps,
        "seeds": [seed_base + i for i in range(reps)],
        "data_source": source,
        "max_test": max_test,
    }


def _aggregate(runs: pd.DataFrame, scenarios) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows, table_rows = [], []
    for sc in scenarios:
        sub = runs[runs["scenario"] == sc.id]
        ok = sub[sub["status"] == "ok"]
        row = {"scenario": sc.id, "kind": sc.kind, "name": sc.name,
               "runs_ok": len(ok), "runs_total": len(sub)}
        for m in MEASURES:
            vals = ok[m.key].to_numpy(dtype=float) if len(ok) else np.array([])
            mean = float(vals.mean()) if len(vals) else float("nan")
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            passed = int(sum(m.passes(v) for v in vals))
            row.update({f"{m.key}_mean": mean, f"{m.key}_std": std,
                        f"{m.key}_min": float(vals.min()) if len(vals) else float("nan"),
                        f"{m.key}_max": float(vals.max()) if len(vals) else float("nan"),
                        f"{m.key}_runs_pass": passed})
            if len(vals) == 0:
                verdict = "Error"
            else:
                verdict = ("Sí" if m.passes(mean) else "No") + f" ({passed} de {len(vals)} repeticiones)"
            n_tx = int(ok["transactions"].iloc[0]) if len(ok) else 0
            table_rows.append({
                "Escenario": f"{sc.id} - {sc.name}",
                "Tipo": sc.kind,
                "Medida de desempeño": m.label,
                "Fórmula": m.formula,
                "Umbral": f"{m.op} {m.threshold:.2f}",
                "Casos ejecutados": f"{len(sub)} repeticiones x {n_tx} transacciones",
                "Resultado obtenido": f"{mean:.4f} +/- {std:.4f}" if len(vals) else "sin resultado",
                "¿Cumple?": verdict,
                "Evidencia": f"scenarios/{sc.id}/",
            })
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(table_rows)


def run_test_plan(source: dict, out_dir, reps: int = 5, seed_base: int = 42, workers: int = 1,
                  max_test: int | None = None, scenario_ids: list[str] | None = None) -> pd.DataFrame:
    scenarios = [s for s in SCENARIOS if not scenario_ids or s.id in scenario_ids]
    if not scenarios:
        raise ValueError("Ningún escenario coincide con los identificadores indicados.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = _plan_dict(scenarios, reps, seed_base, source, max_test)
    plan_file = out_dir / "plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text(encoding="utf-8")) != plan:
        raise SystemExit(f"{plan_file} corresponde a un plan distinto. Usa otra carpeta con --out-dir "
                         "o elimina la existente para empezar de cero.")
    plan_file.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    tasks = [{"scenario": sc, "seed": seed_base + i, "source": source, "max_test": max_test,
              "run_dir": str(out_dir / "scenarios" / sc.id / f"seed_{seed_base + i}")}
             for sc in scenarios for i in range(reps)]
    print(f"{len(scenarios)} escenarios x {reps} repeticiones = {len(tasks)} ejecuciones "
          f"({workers} proceso(s) en paralelo)\n")

    results, started = [], time.perf_counter()

    def report(res: dict, k: int) -> None:
        tag = "retomada" if res.get("resumed") else res["status"]
        detail = (f"recall={res['recall_flagged']:.3f} fpr={res['fpr_flagged']:.4f}"
                  if res["status"] == "ok" else "")
        print(f"[{k}/{len(tasks)}] {res['scenario']} semilla {res['seed']}: {tag} {detail}", flush=True)

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_one, t) for t in tasks]
            for k, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                report(results[-1], k)
    else:
        for k, task in enumerate(tasks, 1):
            results.append(_run_one(task))
            report(results[-1], k)

    runs = pd.DataFrame(results).sort_values(["scenario", "seed"]).reset_index(drop=True)
    if "actions" in runs.columns:
        runs["actions"] = runs["actions"].apply(lambda a: json.dumps(a) if isinstance(a, dict) else a)
    summary, table = _aggregate(runs, scenarios)
    runs.to_csv(out_dir / "runs.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    table.to_csv(out_dir / "tabla_resultados.csv", index=False, encoding="utf-8-sig")

    print(f"\nTerminado en {(time.perf_counter() - started) / 60:.1f} min. Resultados en {out_dir}/")
    print("\nResumen por escenario (media +/- desviación estándar)")
    for _, r in summary.iterrows():
        print(f"  {r['scenario']} {r['name'][:40]:<40} "
              f"recall {r['recall_flagged_mean']:.3f} +/- {r['recall_flagged_std']:.3f}   "
              f"fpr {r['fpr_flagged_mean']:.4f} +/- {r['fpr_flagged_std']:.4f}   "
              f"recall final {r['recall_declined_final_mean']:.3f}   "
              f"ok {r['runs_ok']}/{r['runs_total']}")
    return table
