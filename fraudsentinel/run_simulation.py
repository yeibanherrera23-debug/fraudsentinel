"""Ejecuta una simulación completa: entrenamiento histórico y operación del agente."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .agent import FraudSentinelAgent
from .config import AgentConfig, CostConfig, EnvironmentConfig
from .environment import V_COLUMNS, CardPool, TransactionEnvironment
from .event_log import EventLog
from .risk_model import RiskModel
from .services import Blacklist, CaseManagement, ExternalServices, Notifier, PaymentGateway
from .synthetic import make_synthetic_creditcard
from .utility import APPROVE, BLOCK, ESCALATE, UtilityFunction

REQUIRED_COLUMNS = ["Time", *V_COLUMNS, "Amount", "Class"]
TRAIN_FRACTION = 0.7


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en {path}: {missing}")
    return df.sort_values("Time").reset_index(drop=True)


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


class Tracker:
    """Acumula resultados contra la verdad del entorno (el agente no la ve)."""

    def __init__(self, costs: CostConfig):
        self.costs = costs
        self.rows: list[tuple] = []

    def record(self, amount: float, truth: int, result) -> None:
        cost = 0.0
        if result.first_action == ESCALATE:
            cost += self.costs.review_cost
        if truth == 1 and result.final_status == "APPROVED":
            cost += amount
        if truth == 0 and result.final_status == "DECLINED":
            cost += self.costs.friction_cost
        self.rows.append((truth, result.first_action, result.final_status, amount, cost))

    def summary(self) -> dict:
        truth = np.array([r[0] for r in self.rows])
        first = np.array([r[1] for r in self.rows])
        final = np.array([r[2] for r in self.rows])
        amount = np.array([r[3] for r in self.rows])
        cost = np.array([r[4] for r in self.rows])
        fraud, legit = truth == 1, truth == 0
        flagged, blocked, declined = first != APPROVE, first == BLOCK, final == "DECLINED"
        return {
            "transactions": int(len(truth)),
            "fraud": int(fraud.sum()),
            "legit": int(legit.sum()),
            "actions": dict(Counter(first.tolist())),
            "recall_flagged": safe_rate(int((flagged & fraud).sum()), int(fraud.sum())),
            "recall_declined_final": safe_rate(int((declined & fraud).sum()), int(fraud.sum())),
            "fpr_blocked": safe_rate(int((blocked & legit).sum()), int(legit.sum())),
            "fpr_flagged": safe_rate(int((flagged & legit).sum()), int(legit.sum())),
            "fpr_declined_final": safe_rate(int((declined & legit).sum()), int(legit.sum())),
            "total_cost_agent": float(cost.sum()),
            "total_cost_approve_all": float(amount[fraud].sum()),
        }


def run_experiment(df: pd.DataFrame, env_cfg: EnvironmentConfig, agent_cfg: AgentConfig,
                   costs: CostConfig, run_dir, max_test: int | None = None,
                   log_mode: str = "full", verbose: bool = True) -> dict:
    """Entrena el agente con el historial, lo opera sobre el resto y guarda log y métricas."""
    started = time.perf_counter()
    run_dir = Path(run_dir)
    split = int(len(df) * TRAIN_FRACTION)
    df_train, df_test = df.iloc[:split], df.iloc[split:]
    if max_test:
        df_test = df_test.iloc[:max_test]

    rng = np.random.default_rng(env_cfg.seed)
    pool = CardPool(df["Amount"].to_numpy(dtype=float), env_cfg, rng)
    env_train = TransactionEnvironment(df_train, pool, env_cfg, seed=env_cfg.seed)
    env_test = TransactionEnvironment(df_test, pool, env_cfg, seed=env_cfg.seed + 1, id_offset=len(df_train))
    services = ExternalServices(Blacklist(), PaymentGateway(), CaseManagement(env_test.analyst_review), Notifier())

    with EventLog(run_dir / "events.jsonl", mode=log_mode) as log:
        agent = FraudSentinelAgent(
            RiskModel(agent_cfg.negatives_per_positive, seed=env_cfg.seed),
            UtilityFunction(costs), services, log, agent_cfg)
        if verbose:
            print(f"Entrenando con {len(df_train)} transacciones históricas...")
        agent.warm_up(env_train)

        if verbose:
            print(f"Operando sobre {len(df_test)} transacciones...")
        tracker, due = Tracker(costs), deque()
        for i, raw in enumerate(env_test):
            result = agent.step(raw)
            tracker.record(env_test.true_amount(raw.tx_id), env_test.reveal_outcome(raw.tx_id), result)
            due.append((i + agent_cfg.feedback_delay, raw.tx_id))
            while due and due[0][0] <= i:
                _, tx_id = due.popleft()
                agent.observe_outcome(tx_id, env_test.reveal_outcome(tx_id))

    summary = tracker.summary()
    summary.update(retrains=agent.retrain_count, blacklist_size=len(services.blacklist),
                   cases_opened=services.cases.cases_opened, run_dir=str(run_dir),
                   elapsed_s=round(time.perf_counter() - started, 1))
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simulación del agente FraudSentinel")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="Ruta al archivo creditcard.csv")
    src.add_argument("--synthetic", action="store_true", help="Datos sintéticos (solo verificación técnica)")
    p.add_argument("--rows", type=int, default=20_000, help="Filas sintéticas")
    p.add_argument("--max-test", type=int, default=None, help="Máximo de transacciones de operación")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sensor-dropout", type=float, default=0.0)
    p.add_argument("--analyst-accuracy", type=float, default=1.0)
    p.add_argument("--log-mode", choices=["full", "light"], default="full",
                   help="full: todos los eventos; light: solo transacciones no aprobadas")
    p.add_argument("--log-dir", default="logs")
    return p


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    env_cfg = EnvironmentConfig(seed=args.seed, p_sensor_dropout=args.sensor_dropout,
                                analyst_accuracy=args.analyst_accuracy)
    df = make_synthetic_creditcard(args.rows, seed=args.seed) if args.synthetic else load_data(args.data)
    run_dir = Path(args.log_dir) / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    summary = run_experiment(df, env_cfg, AgentConfig(), CostConfig(), run_dir,
                             max_test=args.max_test, log_mode=args.log_mode)
    print("\nResumen de la ejecución")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    return summary


if __name__ == "__main__":
    main()
