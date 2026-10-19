"""Aplicación web de FraudSentinel (Streamlit): ejecuta el agente y muestra sus decisiones."""
from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

from fraudsentinel.config import AgentConfig, CostConfig, EnvironmentConfig
from fraudsentinel.experiments import MEASURES, SCENARIOS
from fraudsentinel.run_simulation import load_data, run_experiment

DATA_PATH = Path("data/demo_sample.csv.gz")
RESULTS_PATH = Path("results/pruebas/tabla_resultados.csv")
REPO_URL = "https://github.com/yeibanherrera23-debug/fraudsentinel"
SCENARIO_BY_ID = {s.id: s for s in SCENARIOS}

st.set_page_config(page_title="FraudSentinel", page_icon="🛡️", layout="wide")


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_data(str(DATA_PATH))


@st.cache_data(show_spinner=False)
def read_events(path: str) -> pd.DataFrame:
    percepts, decisions, actions, labels = {}, {}, {}, {}
    with open(Path(path) / "events.jsonl", encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            kind = e["event"]
            if kind == "percept":
                percepts[e["tx_id"]] = e
            elif kind == "decision":
                decisions[e["tx_id"]] = e
            elif kind == "action":
                actions[e["tx_id"]] = e
            elif kind == "feedback":
                labels[e["tx_id"]] = e["label"]
    rows = []
    for tx_id, p in percepts.items():
        d, a = decisions.get(tx_id, {}), actions.get(tx_id, {})
        rows.append({
            "tx_id": tx_id, "hora": p["ts"], "tarjeta": p["card_id"], "monto": p["amount"],
            "dispositivo": p["device_id"], "país": p["country"], "categoría": p["merchant_category"],
            "lista_negra": p["blacklist_hit"],
            "p_fraude": d.get("p_fraud"), "decisión": d.get("action"), "motivo": d.get("reason"),
            "estado_final": a.get("final_status"), "fraude_real": labels.get(tx_id),
        })
    return pd.DataFrame(rows)


st.title("🛡️ FraudSentinel")
st.caption("Agente basado en utilidad que decide si aprobar, escalar a revisión o bloquear cada transacción "
           "con tarjeta (ciclo percepción → decisión → acción).")

with st.expander("Instrucciones de uso y casos de prueba sugeridos", expanded=True):
    st.markdown(
        """
**Cómo usarla**
1. En la barra lateral elige un **escenario**, la cantidad de transacciones y la semilla.
2. Pulsa **Ejecutar agente**. El agente se entrena con el historial y luego decide una por una.
3. Revisa las **métricas**, la tabla de **decisiones** y el **registro de eventos** (percepción, decisión y acción con marca de tiempo).
4. En la pestaña **Pruebas** están los resultados completos del plan de pruebas (12 escenarios × 5 repeticiones).

**Casos de prueba sugeridos**
- **N01 – Línea base** (caso típico): debería marcar pocos legítimos y detectar la mayoría de los fraudes.
- **F02 – Sensores de contexto caídos** (falla esperada): dispositivo, país y comercio llegan vacíos y el agente sigue operando solo con V1–V28 y el monto.
- **F03 – Analista poco confiable** (falla esperada): el analista humano acierta solo el 70 % de las revisiones; compara «Falsos positivos finales» con N01.
- **L04 – Costo de revisión bajo (R = 1)** (caso límite): el agente escala muchos más casos.
- **L01 – Defraudador sin señales de contexto** (caso límite): solo V1–V28 distinguen el fraude.

**Aviso.** Esta aplicación usa una muestra liviana del conjunto de datos (todos los fraudes y 40.000 legítimas) porque el
archivo original no cabe en GitHub. Tiene más proporción de fraudes que el conjunto completo, por lo que sus cifras no
reemplazan las de las pruebas oficiales de la pestaña **Pruebas**.
"""
    )

if not DATA_PATH.exists():
    st.error(f"No se encontró {DATA_PATH}. Genérelo con: python tools/make_demo_sample.py data/creditcard.csv")
    st.stop()

with st.sidebar:
    st.header("Configuración")
    scenario_id = st.selectbox("Escenario", list(SCENARIO_BY_ID),
                               format_func=lambda k: f"{k} – {SCENARIO_BY_ID[k].name}")
    scenario = SCENARIO_BY_ID[scenario_id]
    st.caption(scenario.description)
    n_tx = st.slider("Transacciones a operar", 500, 5000, 1500, step=250)
    seed = st.number_input("Semilla", min_value=0, value=42, step=1)
    run = st.button("Ejecutar agente", type="primary")
    st.markdown(f"[Repositorio]({REPO_URL})")

tab_run, tab_tests = st.tabs(["Ejecución", "Pruebas"])

with tab_run:
    if run:
        env_cfg = replace(EnvironmentConfig(seed=int(seed)), **scenario.env)
        agent_cfg = replace(AgentConfig(), **scenario.agent)
        costs = replace(CostConfig(), **scenario.costs)
        run_dir = Path(tempfile.mkdtemp(prefix="fraudsentinel_"))
        with st.spinner("Entrenando con el historial y operando el agente…"):
            summary = run_experiment(get_data(), env_cfg, agent_cfg, costs, run_dir,
                                     max_test=int(n_tx), log_mode="full", verbose=False)
        st.session_state["result"] = {"summary": summary, "run_dir": str(run_dir), "scenario": scenario_id,
                                      "n_tx": int(n_tx), "seed": int(seed)}

    result = st.session_state.get("result")
    if not result:
        st.info("Elige un escenario y pulsa **Ejecutar agente** en la barra lateral.")
    else:
        s = result["summary"]
        st.subheader(f"Resultado: {result['scenario']} – {SCENARIO_BY_ID[result['scenario']].name}")
        c = st.columns(4)
        c[0].metric("Recall (fraudes marcados)", f"{s['recall_flagged']:.2f}" if s["fraud"] else "n/d",
                    help="Fraudes bloqueados o escalados / fraudes totales. Umbral del diseño: ≥ 0,80")
        c[1].metric("Falsos positivos", f"{s['fpr_flagged']:.4f}",
                    help="Legítimas bloqueadas o escaladas / legítimas totales. Umbral del diseño: < 0,05")
        c[2].metric("Costo del agente", f"{s['total_cost_agent']:.2f} €")
        c[3].metric("Costo de aprobar todo", f"{s['total_cost_approve_all']:.2f} €")
        st.write(f"**{s['transactions']}** transacciones, **{s['fraud']}** fraudes, acciones: {s['actions']}, "
                 f"reentrenamientos: {s['retrains']}, lista negra: {s['blacklist_size']} dispositivos, "
                 f"casos abiertos: {s['cases_opened']}, tiempo: {s['elapsed_s']} s.")
        for m in MEASURES:
            if m.official:
                value = s[m.key]
                ok = (value == value) and m.passes(value)
                st.write(f"{'✅' if ok else '❌'} **{m.label}** = {value:.4f} (umbral {m.op} {m.threshold})")

        events = read_events(result["run_dir"])
        st.subheader("Decisiones del agente")
        f1, f2 = st.columns(2)
        only_flagged = f1.checkbox("Solo transacciones no aprobadas", value=True)
        only_fraud = f2.checkbox("Solo fraudes reales")
        view = events
        if only_flagged:
            view = view[view["decisión"] != "APPROVE"]
        if only_fraud:
            view = view[view["fraude_real"] == 1]
        st.dataframe(view, hide_index=True)
        st.caption("«fraude_real» solo se conoce cuando llega la confirmación (con retraso); vacío = aún sin confirmar.")

        with st.expander("Registro de eventos (primeras 40 líneas)"):
            lines = []
            with open(Path(result["run_dir"]) / "events.jsonl", encoding="utf-8") as fh:
                for _, line in zip(range(40), fh):
                    lines.append(line.rstrip())
            st.code("\n".join(lines), language="json")

with tab_tests:
    st.subheader("Resultados del plan de pruebas")
    st.write("12 escenarios (normales, límite y de falla) × 5 repeticiones, 10.000 transacciones cada una, sobre el "
             "conjunto de datos completo. Umbrales fijados antes de ejecutar: recall ≥ 0,80 y falsos positivos < 0,05.")
    if RESULTS_PATH.exists():
        table = pd.read_csv(RESULTS_PATH)
        st.dataframe(table, hide_index=True)
    else:
        st.warning("No se encontró results/pruebas/tabla_resultados.csv en el repositorio.")
    st.markdown(f"Código, registros y documento: [{REPO_URL}]({REPO_URL})")
