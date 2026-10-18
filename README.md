# FraudSentinel

Agente basado en utilidad para la detección de fraude transaccional con tarjeta. Opera sobre un
entorno simulado que emite transacciones una a una; para cada una ejecuta el ciclo
**percepción → decisión → acción** y decide entre aprobar, escalar a revisión humana o bloquear.

## Estructura

```
fraudsentinel/
  config.py          Parámetros del agente, del entorno y de los costos
  environment.py     Simulador: tarjetas, clientes, defraudadores, sensores y analista
  agent.py           Agente: perceive() -> decide() -> act(), memoria y aprendizaje
  utility.py         Función de utilidad (costos esperados por acción)
  risk_model.py      Modelo de riesgo (probabilidad de fraude)
  services.py        Servicios externos simulados (lista negra, pasarela, casos, notificaciones)
  event_log.py       Registro JSON Lines con marca de tiempo
  synthetic.py       Datos sintéticos con el mismo esquema (solo verificación técnica)
  run_simulation.py  Ejecución individual del agente
  experiments.py     Plan de pruebas: medidas, umbrales, escenarios y repeticiones
  run_tests.py       Ejecuta el plan de pruebas
tools/
  analyze_run.py     Desglosa una ejecución (qué acción tomó el agente y qué resultado real tuvo)
```

## Requisitos

Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Probado con Python 3.12, numpy 2.4, pandas 3.0 y scikit-learn 1.8.

## Datos

Se usa el conjunto "Credit Card Fraud Detection" (Worldline / ULB):
<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>

Descargar `creditcard.csv` y ubicarlo en `data/creditcard.csv`. El archivo no se incluye en el
repositorio. Columnas esperadas: `Time`, `V1`…`V28`, `Amount`, `Class`.

## Ejecución

```bash
python -m fraudsentinel.run_simulation --data data/creditcard.csv
```

Opciones:

| Opción | Descripción |
|---|---|
| `--data RUTA` | Ruta del CSV |
| `--synthetic` | Usa datos sintéticos en lugar del CSV (verificación técnica; sus resultados no son evidencia de desempeño) |
| `--rows N` | Filas sintéticas (por defecto 20000) |
| `--max-test N` | Límite de transacciones de operación |
| `--seed N` | Semilla de aleatoriedad (por defecto 42) |
| `--sensor-dropout P` | Probabilidad de que un sensor no entregue dato |
| `--analyst-accuracy P` | Exactitud del analista humano simulado |
| `--log-mode full\|light` | `full` registra todo; `light` solo las transacciones no aprobadas |
| `--log-dir RUTA` | Carpeta de salida (por defecto `logs`) |

Cada ejecución crea `logs/run_<fecha>/` con:

- `events.jsonl`: un evento por línea (`percept`, `decision`, `action`, `feedback`, `retrain`, `warm_up`) con marca de tiempo UTC.
- `metrics.json`: resumen de la ejecución.

## Pruebas contra las medidas de desempeño

El plan de pruebas ejecuta 12 escenarios (3 normales, 6 límite y 3 de falla esperada) con 5
repeticiones cada uno, cada repetición con una semilla distinta:

```bash
python -m fraudsentinel.run_tests --list                        # muestra medidas y escenarios
python -m fraudsentinel.run_tests --data data/creditcard.csv --reps 5 --workers 2
```

| Opción | Descripción |
|---|---|
| `--reps N` | Repeticiones por escenario (por defecto 5) |
| `--workers N` | Ejecuciones en paralelo (cada proceso carga el CSV en memoria y usa un núcleo) |
| `--scenarios N01,L01` | Ejecuta solo algunos escenarios |
| `--max-test N` | Limita las transacciones de operación (solo para verificar) |
| `--out-dir RUTA` | Carpeta de resultados (por defecto `results/pruebas`) |

Las medidas y sus umbrales están definidos en `fraudsentinel/experiments.py`:

| Medida | Fórmula | Umbral |
|---|---|---|
| Recall (medida del diseño) | fraudes marcados (bloqueados o escalados) / fraudes totales | ≥ 0,80 |
| Tasa de falsos positivos (medida del diseño) | legítimas marcadas / legítimas totales | < 0,05 |
| Recall final (complementaria) | fraudes finalmente rechazados / fraudes totales | ≥ 0,80 |
| Falsos positivos finales (complementaria) | legítimas finalmente rechazadas / legítimas totales | < 0,05 |

Los escenarios son: línea base, analista con 90 % de acierto, sensores con 10 % de datos faltantes,
defraudador sin señales de contexto, defraudador con señales débiles, costo de fricción alto,
costo de revisión bajo, sin reentrenamiento, confirmaciones tardías, falla de sensores del 30 %,
sensores de contexto caídos y analista poco confiable (70 %). Cada uno está descrito en
`SCENARIOS` y se lista con `--list`.

La ejecución es reanudable: si se interrumpe, al repetir el mismo comando se retoman las
ejecuciones que ya terminaron. Si se cambia el plan, hay que usar otra carpeta con `--out-dir`.

Resultados en `results/pruebas/`:

- `tabla_resultados.csv`: tabla de resultados (escenario, medida, fórmula, umbral, casos ejecutados, resultado como media ± desviación estándar, cumplimiento y evidencia).
- `summary.csv`: media, desviación estándar, mínimo y máximo por escenario y medida.
- `runs.csv`: una fila por ejecución, con todas las métricas y el estado (`ok` o el error).
- `plan.json`: escenarios, medidas, semillas y fuente de datos usados.
- `scenarios/<id>/seed_<n>/`: `metrics.json` y `events.jsonl` (modo `light`) de cada ejecución.

## Funcionamiento

1. **Entrenamiento histórico (`warm_up`).** El modelo de riesgo se entrena con el primer 70 % de las transacciones (ordenadas por `Time`), con submuestreo de la clase legítima y corrección de prior para recuperar probabilidades calibradas. Se construyen los perfiles iniciales por tarjeta.
2. **Operación.** Con el 30 % restante, el agente procesa cada transacción:
   - `perceive`: arma el vector de características (V1…V28 y variables de contexto calculadas contra el perfil de la tarjeta) y consulta la lista negra.
   - `decide`: si hay coincidencia en lista negra bloquea; en otro caso estima `p` y elige la acción de mayor utilidad esperada.
   - `act`: invoca los servicios externos (pasarela, casos, notificaciones) y actualiza el perfil si la transacción se aprueba.
3. **Aprendizaje.** El resultado real llega con retraso (`feedback_delay`); el agente lo incorpora, alimenta la lista negra y reentrena el modelo cada `retrain_every` confirmaciones.

Función de utilidad (costos en EUR; utilidad = −costo esperado):

| Acción | Costo esperado |
|---|---|
| Aprobar | `p × monto` |
| Escalar | `R` (revisión humana) |
| Bloquear | `(1 − p) × F` (fricción) |

Los valores por defecto son `F = 20` y `R = 5` (`config.py`).

## Componentes de la arquitectura agéntica (marco IBM)

| Componente | Implementación |
|---|---|
| Percepción y manejo de entradas | `RawTransaction`, `FraudSentinelAgent.perceive` |
| Planificación y descomposición de tareas | `FraudSentinelAgent.step` (secuencia percepción → decisión → acción) |
| Memoria | `CardProfile`, `FraudSentinelAgent.profiles`, historial de experiencia |
| Razonamiento y toma de decisiones | `FraudSentinelAgent.decide`, `UtilityFunction`, `RiskModel` |
| Llamada de herramientas y acción | `services.py` (`Blacklist`, `PaymentGateway`, `CaseManagement`), invocados desde `perceive` y `act` |
| Comunicación | `Notifier`, resumen enviado en `CaseManagement.open_case`, `EventLog` |
| Aprendizaje y adaptación | `warm_up`, `observe_outcome`, `_retrain`, `RiskModel.fit` |

## Supuestos del simulador

El conjunto de datos no incluye identificador de tarjeta, comercio, dispositivo ni país. El
simulador los genera con aleatoriedad:

- Cada tarjeta tiene país de origen, 1 a 3 dispositivos, 3 categorías de comercio preferidas y un nivel de gasto habitual.
- Los clientes legítimos usan casi siempre sus dispositivos, país y categorías habituales; los defraudadores usan con alta probabilidad dispositivos y países nuevos (parámetros en `EnvironmentConfig`).
- Las variables V1…V28, el monto y la etiqueta provienen del conjunto de datos; las variables de contexto son sintéticas por construcción.
- Los parámetros `p_fraud_same_tier` y `p_fraud_preferred_category` (0 por defecto) permiten que el defraudador imite el nivel de gasto y las categorías habituales de la tarjeta.
- Pueden simularse fallas de sensores (`--sensor-dropout`) y errores del analista (`--analyst-accuracy`).
