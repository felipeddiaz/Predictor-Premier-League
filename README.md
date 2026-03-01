# Predictor de Partidos — Premier League

> Sistema de predicción de resultados de la Premier League con Machine Learning, ingeniería de features avanzada sobre 10 temporadas (2016–2026), optimización bayesiana y sistema de value betting con Expected Value y Kelly Criterion.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-red)
![Optuna](https://img.shields.io/badge/Optuna-3.x-green)
![Estado](https://img.shields.io/badge/Estado-En%20desarrollo-yellow)

---

## Descripción del Proyecto

Este proyecto predice el resultado de partidos de la Premier League inglesa — **Victoria Local (H), Empate (D) o Victoria Visitante (A)** — usando un pipeline completo de Machine Learning que cubre desde la recolección y limpieza de datos históricos hasta la generación de reportes PDF/Excel por jornada.

Además de predecir resultados, el sistema identifica **value bets**: situaciones donde la probabilidad estimada por el modelo difiere significativamente de las cuotas del mercado, calculando el Expected Value (EV) y el tamaño de apuesta óptimo mediante el criterio de Kelly.

### ¿Qué problema resuelve?

Los modelos de predicción deportiva simples suelen limitarse a clasificar resultados. Este proyecto va más allá:

- Integra señales de mercado (Pinnacle, Bet365, Asian Handicap) junto con estadísticas de rendimiento
- Selecciona automáticamente las features más informativas mediante permutation importance
- Co-optimiza pesos de clase e hiperparámetros simultáneamente con búsqueda bayesiana
- Calibra probabilidades condicionalmente (solo si mejora el F1)
- Aplica tres capas de filtrado de riesgo antes de recomendar una apuesta
- Genera reportes automatizados por jornada

> **Disclaimer:** Este proyecto es estrictamente educativo. No constituye consejo financiero ni garantiza ganancias. Las apuestas deportivas conllevan riesgo de pérdida de capital.

---

## Tecnologías y Librerías

| Categoría | Tecnología |
|---|---|
| Lenguaje | Python 3.9+ |
| Manipulación de datos | pandas, numpy |
| Machine Learning | scikit-learn (RandomForestClassifier, CalibratedClassifierCV, TimeSeriesSplit, permutation_importance) |
| Gradient Boosting | XGBoost |
| Optimización bayesiana | Optuna (TPESampler, 150 trials por modelo) |
| Persistencia de modelos | joblib |
| Visualización | matplotlib, seaborn |
| Generación de reportes PDF | fpdf |
| Exportación Excel | openpyxl |

---

## Fuentes de Datos

### 1. football-data.co.uk — Dataset principal

**10 temporadas** de la Premier League inglesa (2016/17 a 2025/26), descargadas como CSVs individuales y almacenadas en `datos/temporadas/`. Cada CSV cubre una temporada completa e incluye:

| Columna | Descripción |
|---|---|
| `Date`, `HomeTeam`, `AwayTeam` | Identificación del partido |
| `FTHG`, `FTAG`, `FTR` | Goles locales, goles visitantes y resultado final (H/D/A) |
| `HS`, `AS`, `HST`, `AST` | Tiros totales y tiros a puerta por equipo |
| `HF`, `AF` | Faltas cometidas |
| `HC`, `AC` | Corners |
| `HY`, `AY`, `HR`, `AR` | Tarjetas amarillas y rojas |
| `Referee` | Árbitro del partido |
| `B365H`, `B365D`, `B365A` | Cuotas Bet365 apertura (1X2) |
| `B365CH`, `B365CD`, `B365CA` | Cuotas Bet365 cierre (1X2) |
| `PSH`, `PSD`, `PSA` | Cuotas Pinnacle apertura |
| `PSCH`, `PSCD`, `PSCA` | Cuotas Pinnacle cierre |
| `AHh`, `AHCh` | Línea Asian Handicap apertura y cierre (local) |
| `B365AHH`, `B365AHA` | Cuotas Bet365 Asian Handicap apertura |
| `B365CAHH`, `B365CAHA` | Cuotas Bet365 Asian Handicap cierre |

> **Nota sobre cobertura:** Las cuotas Pinnacle y Asian Handicap están disponibles en todas las temporadas (2016–2026). Los datos de Asian Handicap cubren aproximadamente el 61% de los partidos del dataset total.

### 2. FBRef / Understat — Expected Goals (xG)

Datos de Expected Goals por equipo y partido, almacenados en `datos/raw/final_matches_xg.csv`. Se obtienen mediante el script `herramientas/scrape_xg_understat.py` que descarga los datos de Understat.com y los fusiona con el dataset principal por fecha y nombre de equipo.

### 3. Ratings ELO históricos

Ratings ELO por equipo almacenados en `datos/raw/elo/`. Calculados externamente y cargados como CSVs por temporada. Están disponibles en el código (`utils.agregar_features_elo`) pero **excluidos del modelo actual** porque en pruebas redujeron el F1 en -0.014.

### 4. Cuotas actuales por jornada

Las cuotas Bet365 de la jornada a predecir se ingresan manualmente en `jornada/jornada_config.py` antes de cada ejecución.

---

## Estructura del Proyecto

```
mi_predictor_pl/
│
├── config.py                          # Configuración centralizada: rutas, features, params
├── utils.py                           # Librería de feature engineering compartida (1 126 líneas)
├── pyproject.toml                     # Paquete instalable con pip install -e .
├── requirements.txt
├── README.md
│
├── pipeline/                          # Scripts de entrenamiento secuenciales
│   ├── 01_preparar_datos.py           # Limpieza + feature engineering → CSV canónico
│   ├── 02_entrenar_modelo.py          # Entrena RF + XGBoost + calibración → .pkl
│   ├── 03_entrenar_sin_cuotas.py      # Modelo sin cuotas para edge estructural
│   └── 04_analizar_feature_importance.py  # Análisis de importancia (solo lectura)
│
├── core/                              # Módulo de lógica de producción
│   ├── models.py                      # Dataclasses: Partido, Prediccion, ConfigJornada
│   ├── predictor.py                   # Clase Predictor: carga modelo, predice, genera reportes
│   └── sistema_expected_value.py      # EV, Kelly Criterion, simulación ROI
│
├── jornada/                           # Punto de entrada operativo por jornada
│   ├── jornada_config.py              # Define fixtures y cuotas de la jornada actual
│   └── predecir_jornada_completa.py   # Orquestador: Predictor → PDF + Excel
│
├── herramientas/                      # Scripts de experimentación y optimización
│   ├── buscar_pesos_clase.py          # Co-optimización Optuna RF Bal + RF Optuna (150 trials)
│   ├── buscar_pesos_xgb.py            # Co-optimización Optuna XGBoost (150 trials)
│   ├── scrape_xg_understat.py         # Descarga xG de Understat y actualiza el CSV
│   ├── visualizacion.py               # Genera 6 PNGs de portafolio a 300 DPI
│   ├── analisis_rapido.py             # Diagnóstico rápido del CSV procesado
│   ├── verify_setup.py                # Verifica integridad del dataset canónico
│   ├── analizar_features.py           # Análisis de importancia y redundancia de features
│   └── [otros scripts de experimentación histórica]
│
├── datos/
│   ├── temporadas/                    # CSVs crudos de football-data.co.uk (10 temporadas)
│   │   └── 16-17.csv … 25-26.csv
│   ├── procesados/
│   │   └── archive/
│   │       └── premier_league_RESTAURADO.csv   # Dataset canónico con todas las features
│   └── raw/
│       ├── final_matches_xg.csv       # xG de Understat
│       └── elo/                       # Ratings ELO históricos
│
├── modelos/                           # Modelos entrenados y artefactos
│   ├── modelo_final_optimizado.pkl    # Modelo de producción (XGBoost, F1=0.5102)
│   ├── features.pkl                   # Lista exacta de las 55 features del modelo
│   ├── metadata.pkl                   # Métricas y metadatos del entrenamiento
│   ├── modelo_value_betting.pkl       # Modelo alternativo sin cuotas
│   └── optuna_*.json                  # Resultados de búsquedas Optuna por configuración
│
├── reportes/                          # PDFs y Excels generados por jornada
│
└── portafolio_imagenes/               # PNGs 300 DPI para portafolio web
    ├── 01_matriz_confusion.png
    ├── 02_feature_importance.png
    ├── 03_f1_por_clase.png
    ├── 04_evolucion_modelo.png
    ├── 05_roi_simulation.png
    └── 06_caso_estudio.png
```

---

## Pipeline de Trabajo

### Flujo completo

```
datos/temporadas/*.csv  (10 temporadas, football-data.co.uk)
datos/raw/final_matches_xg.csv  (xG, Understat)
         │
         ▼  pipeline/01_preparar_datos.py
datos/procesados/premier_league_RESTAURADO.csv
  (3 691 partidos, 2016-2026, con features calculadas)
         │
         ▼  pipeline/02_entrenar_modelo.py
modelos/modelo_final_optimizado.pkl
  (XGBoost, 55 features, F1-weighted = 0.5102, split 80/20 temporal)
         │
         ▼  jornada/jornada_config.py  →  jornada/predecir_jornada_completa.py
reportes/predicciones_jornada_XX.pdf + .xlsx
  (probabilidades calibradas → EV → Kelly Criterion → apuestas recomendadas)
```

### Paso 1 — Preparar datos (`pipeline/01_preparar_datos.py`)

Carga los 10 CSVs de temporadas, los concatena, limpia fechas y columnas con exceso de NaN, y aplica toda la ingeniería de features descrita en la siguiente sección. El resultado es el dataset canónico con 3 691 partidos y más de 80 features disponibles.

### Paso 2 — Entrenar modelos (`pipeline/02_entrenar_modelo.py`)

Entrena y compara 3 modelos sobre un split **80/20 temporal sin shuffle** (los últimos 739 partidos como test):

1. **XGBoost** — se entrena primero para evitar contaminación de threads con scikit-learn
2. **Random Forest Balanceado** — pesos de clase co-optimizados con Optuna
3. **Random Forest Optuna** — hiperparámetros y pesos co-optimizados simultáneamente

El ganador por F1-weighted en test se calibra condicionalmente con Platt Scaling (solo si la calibración mejora el F1) y se guarda como `modelos/modelo_final_optimizado.pkl` junto con la lista de features y metadatos.

### Paso 3 — Predecir jornada (`jornada/predecir_jornada_completa.py`)

Se edita `jornada/jornada_config.py` con los fixtures y cuotas actuales de Bet365. El orquestador carga el modelo, construye las features en memoria para cada partido, genera probabilidades, aplica el sistema de value betting de 3 capas y produce un PDF y un Excel en `reportes/`.

---

## Ingeniería de Features

El universo completo de features disponibles es de **84 variables** agrupadas en 9 categorías. Durante el entrenamiento se aplica un proceso de **selección por permutation importance** (RF balanced, n_repeats=5, evaluado en el último fold del TimeSeriesSplit(n_splits=5) sobre el 80% de train) para seleccionar las **55 features más informativas** que usa el modelo de producción.

Todas las features temporales utilizan `shift(1)` explícito para evitar data leakage — el partido actual nunca se incluye en su propia ventana de cálculo.

---

### Categoría 1 — Estadísticas Base Rolling (10 features)

**Fuente:** CSVs de temporadas (football-data.co.uk).  
**Calculadas en:** `utils.agregar_xg_rolling()` y `pipeline/01_preparar_datos.py`.  
**Ventana:** últimos 5 partidos, agrupada por equipo.

| Feature | Descripción |
|---|---|
| `HT_AvgGoals` | Promedio de goles marcados por el local en sus últimos 5 partidos |
| `AT_AvgGoals` | Promedio de goles marcados por el visitante en sus últimos 5 partidos |
| `HT_AvgShotsTarget` | Promedio de tiros a puerta del local en sus últimos 5 partidos |
| `AT_AvgShotsTarget` | Promedio de tiros a puerta del visitante en sus últimos 5 partidos |
| `HT_Form_W/D/L` | Victorias / empates / derrotas del local en los últimos 5 partidos |
| `AT_Form_W/D/L` | Victorias / empates / derrotas del visitante en los últimos 5 partidos |

**Todas incluidas en FEATURES_SELECCIONADAS:** `HT_AvgGoals`, `AT_AvgGoals`, `HT_AvgShotsTarget`, `AT_AvgShotsTarget`, `HT_Form_W`, `AT_Form_W`. Las features `HT_Form_D`, `HT_Form_L`, `AT_Form_D`, `AT_Form_L` no fueron seleccionadas por permutation importance.

---

### Categoría 2 — Cuotas de Mercado Bet365 (6 features)

**Fuente:** CSVs de temporadas (football-data.co.uk), columnas `B365*`.  
**Disponibilidad:** todas las temporadas (2016–2026).

| Feature | Descripción |
|---|---|
| `B365H`, `B365D`, `B365A` | Cuotas Bet365 apertura: victoria local, empate, victoria visitante |
| `B365CH`, `B365CD`, `B365CA` | Cuotas Bet365 cierre: mismas tres opciones al cierre del mercado |

**Incluidas en FEATURES_SELECCIONADAS:** ninguna directamente. Su información se consume a través de las features derivadas (Categoría 3).

---

### Categoría 3 — Señales Derivadas de Cuotas (10 features)

**Fuente:** calculadas en memoria a partir de las cuotas Bet365 apertura/cierre.  
**Calculadas en:** `utils.agregar_features_cuotas_derivadas()`.

| Feature | Descripción |
|---|---|
| `Prob_H`, `Prob_D`, `Prob_A` | Probabilidades implícitas normalizadas (eliminado el margen de la casa) |
| `Prob_Move_H`, `Prob_Move_D`, `Prob_Move_A` | Movimiento de probabilidad implícita entre apertura y cierre |
| `Market_Move_Strength` | Intensidad total del movimiento de mercado |
| `Prob_Spread` | Diferencia entre la probabilidad más alta y más baja (proxy de certeza del mercado) |
| `Market_Confidence` | Qué tan concentrada está la probabilidad en un resultado |
| `Home_Advantage_Prob` | Ventaja de probabilidad del local sobre el visitante |

**Incluidas en FEATURES_SELECCIONADAS:** `Prob_A`, `Prob_Move_H`, `Prob_Move_D`, `Prob_Move_A`, `Prob_Spread`. Excluidas: `Prob_H`, `Prob_D`, `Market_Move_Strength`, `Market_Confidence`, `Home_Advantage_Prob`.

---

### Categoría 4 — Expected Goals (xG) (6 features)

**Fuente:** `datos/raw/final_matches_xg.csv`, obtenido de Understat.com vía `herramientas/scrape_xg_understat.py`.  
**Calculadas en:** `utils.agregar_xg_rolling()`.  
**Ventana:** últimos 5 partidos.

| Feature | Descripción |
|---|---|
| `HT_xG_Avg` | xG promedio creado por el local en los últimos 5 partidos |
| `AT_xG_Avg` | xG promedio creado por el visitante en los últimos 5 partidos |
| `HT_xGA_Avg` | xG concedido promedio por el local en los últimos 5 partidos |
| `AT_xGA_Avg` | xG concedido promedio por el visitante en los últimos 5 partidos |
| `xG_Diff` | Diferencia de xG creado entre local y visitante |
| `xG_Total` | Suma del xG de ambos equipos |

**Todas incluidas en FEATURES_SELECCIONADAS:** `xG_Total`, `AT_xGA_Avg`, `HT_xG_Avg`, `xG_Diff`, `AT_xG_Avg`.  
`HT_xGA_Avg` no fue seleccionada por permutation importance.

---

### Categoría 5 — Head-to-Head (9 features)

**Fuente:** historial calculado sobre el propio dataset (partidos anteriores entre los mismos equipos).  
**Calculadas en:** `utils.calcular_h2h_features()`.  
**Ventana:** últimos 5 enfrentamientos directos entre el par de equipos.

| Feature | Descripción |
|---|---|
| `H2H_Matches` | Número de enfrentamientos H2H disponibles |
| `H2H_Home_Goals_Avg` | Media de goles del local en los enfrentamientos H2H |
| `H2H_Away_Goals_Avg` | Media de goles del visitante en los enfrentamientos H2H |
| `H2H_Home_Win_Rate` | Tasa de victoria del local en los enfrentamientos H2H |
| `H2H_BTTS_Rate` | Tasa de partidos en que ambos equipos marcaron (BTTS) en H2H |
| `H2H_Goal_Diff` | Diferencia media de goles en los enfrentamientos H2H |
| `H2H_Win_Advantage` | Diferencia de tasas de victoria entre local y visitante en H2H |
| `H2H_Total_Goals_Avg` | Media de goles totales por partido en H2H |
| `H2H_Home_Consistent` | Flag de consistencia histórica del local en H2H |

**Incluidas en FEATURES_SELECCIONADAS:** `H2H_Away_Goals_Avg`, `H2H_Matches`, `H2H_Total_Goals_Avg`, `H2H_Home_Win_Rate`, `H2H_BTTS_Rate`. Excluidas: `H2H_Home_Goals_Avg`, `H2H_Goal_Diff`, `H2H_Win_Advantage`, `H2H_Home_Consistent`.

---

### Categoría 6 — Posición en Tabla y Contexto de Partido (11 features)

**Fuente:** calculadas a partir de los resultados acumulados del propio dataset, temporada a temporada.  
**Calculadas en:** `utils.agregar_features_tabla()`.  
La posición en tabla de cada equipo se recalcula *antes* de cada partido para evitar leakage, usando el criterio oficial de la Premier League (puntos → diferencia de goles → goles a favor → goles en contra como desempate).

| Feature | Descripción |
|---|---|
| `HT_Position` | Posición en tabla del local antes del partido |
| `AT_Position` | Posición en tabla del visitante antes del partido |
| `Position_Diff` | Diferencia de posición (local − visitante) |
| `Position_Diff_Weighted` | Diferencia ponderada por el progreso de la temporada |
| `HT_Points` | Puntos del local acumulados hasta ese partido |
| `AT_Points` | Puntos del visitante acumulados hasta ese partido |
| `Season_Progress` | Fracción de la temporada completada (0–1) |
| `Position_Reliability` | Fiabilidad de la posición (baja al inicio de la temporada) |
| `Match_Type` | Tipo de partido: duelo de grandes, lucha por descenso, entre zonas, etc. |
| `HT_Pressure` | Presión del local (importancia relativa del resultado para sus objetivos) |
| `AT_Pressure` | Presión del visitante |

**Incluidas en FEATURES_SELECCIONADAS:** `Position_Diff`, `Position_Diff_Weighted`, `HT_Points`, `AT_Position`, `AT_Points`, `Season_Progress`, `Position_Reliability`, `Match_Type`, `HT_Pressure`. Excluidas: `HT_Position`, `AT_Pressure`.

---

### Categoría 7 — Forma y Momentum (15 features)

**Fuente:** calculadas a partir de los resultados del propio dataset.  
**Calculadas en:** `utils.agregar_features_forma_momentum()`.  
**Ventana:** últimos 5 partidos generales + últimos 5 partidos por venue (en casa / fuera).

| Feature | Descripción |
|---|---|
| `HT_WinRate5` | % victorias del local en sus últimos 5 partidos (cualquier sede) |
| `AT_WinRate5` | % victorias del visitante en sus últimos 5 partidos (cualquier sede) |
| `HT_Streak` | Racha actual del local (+N victorias consecutivas / −N derrotas) |
| `AT_Streak` | Racha actual del visitante |
| `HT_Pts5` | Puntos del local en sus últimos 5 partidos |
| `AT_Pts5` | Puntos del visitante en sus últimos 5 partidos |
| `HT_GoalsFor5` | Goles marcados por el local en sus últimos 5 partidos |
| `AT_GoalsFor5` | Goles marcados por el visitante en sus últimos 5 partidos |
| `HT_GoalsAgainst5` | Goles encajados por el local en sus últimos 5 partidos |
| `AT_GoalsAgainst5` | Goles encajados por el visitante en sus últimos 5 partidos |
| `Momentum_Diff` | Diferencia de puntos recientes: HT_Pts5 − AT_Pts5 |
| `HT_HomeWinRate5` | % victorias del local jugando en casa (últimos 5 en casa) |
| `AT_AwayWinRate5` | % victorias del visitante jugando fuera (últimos 5 fuera) |
| `HT_HomeGoals5` | Goles marcados por el local en sus últimos 5 partidos en casa |
| `AT_AwayGoals5` | Goles marcados por el visitante en sus últimos 5 partidos fuera |

**Incluidas en FEATURES_SELECCIONADAS:** `HT_HomeGoals5`, `AT_GoalsFor5`, `HT_HomeWinRate5`, `HT_GoalsFor5`, `Momentum_Diff`, `HT_Streak`, `HT_Form_W`, `AT_Form_W`. Excluidas: `HT_WinRate5`, `AT_WinRate5`, `AT_Streak`, `HT_Pts5`, `AT_Pts5`, `HT_GoalsAgainst5`, `AT_GoalsAgainst5`, `AT_AwayWinRate5`, `AT_AwayGoals5`.

---

### Categoría 8 — Señales Pinnacle / Sharp Money (6 features)

**Fuente:** CSVs de temporadas (football-data.co.uk), columnas `PSH`, `PSD`, `PSA`, `PSCH`, `PSCD`, `PSCA`.  
**Calculadas en:** `utils.agregar_features_pinnacle_move()`.  
**Disponibilidad:** todas las temporadas (2016–2026).

Pinnacle es considerada la casa de apuestas con los mercados más eficientes del mundo. El movimiento de sus líneas entre apertura y cierre es una señal de "sharp money" (dinero inteligente) muy valorada.

| Feature | Descripción |
|---|---|
| `Pinnacle_Move_H` | Movimiento de cuota Pinnacle local: cierre − apertura (positivo = cuota subió, local menos favorito) |
| `Pinnacle_Move_A` | Movimiento de cuota Pinnacle visitante |
| `Pinnacle_Move_D` | Movimiento de cuota Pinnacle empate |
| `Pinnacle_Sharp_H` | Probabilidad implícita cierre Pinnacle local (sin vig, normalizada) |
| `Pinnacle_Sharp_A` | Probabilidad implícita cierre Pinnacle visitante (sin vig, normalizada) |
| `Pinnacle_Conf` | Confianza del mercado sharp: max(Sharp_H, Sharp_A) − 1/3 |

**Todas incluidas en FEATURES_SELECCIONADAS.** `Pinnacle_Sharp_H` y `Pinnacle_Sharp_A` son las **dos features más importantes** del modelo actual (importancias XGBoost de 0.0783 y 0.0680 respectivamente).

---

### Categoría 9 — Asian Handicap (7 features)

**Fuente:** CSVs de temporadas (football-data.co.uk), columnas `AHh`, `AHCh`, `B365AHH`, `B365AHA`, `B365CAHH`, `B365CAHA`.  
**Calculadas en:** `utils.agregar_features_asian_handicap()`.  
**Disponibilidad:** aproximadamente el 61% de los partidos del dataset (no disponible en partidos más antiguos o sin cobertura de la fuente).

El Asian Handicap elimina el empate del mercado, obligando al apostador a elegir entre dos lados. La línea de hándicap refleja la fuerza relativa percibida entre los equipos con gran precisión.

| Feature | Descripción |
|---|---|
| `AHh` | Línea de hándicap apertura del equipo local (ej: −1.5 = local favorito por 1.5 goles) |
| `AHCh` | Línea de hándicap cierre del equipo local |
| `AH_Line_Move` | Movimiento de línea AH: AHCh − AHh |
| `AH_Implied_Home` | Probabilidad implícita del local a partir de cuotas AH apertura (sin vig) |
| `AH_Edge_Home` | Diferencia entre probabilidad AH y probabilidad 1X2 del mercado para el local |
| `AH_Market_Conf` | Confianza del mercado AH: distancia de la cuota respecto a la línea justa (1.909) |
| `AH_Close_Move_H` | Movimiento de cuota AH local de apertura a cierre |

**Todas incluidas en FEATURES_SELECCIONADAS.** Los NaN (partidos sin datos AH) se imputan con la mediana del resto de partidos — **no con 0**, ya que `AHh=0` significa "partido parejo" y no "dato ausente". `AH_Edge_Home` es la **cuarta feature más importante** del modelo (importancia XGBoost = 0.0485).

---

### Categoría 10 — Features Rolling Adicionales (4 features)

**Fuente:** calculadas a partir del propio dataset y de cuotas Pinnacle.  
**Calculadas en:** `utils.agregar_features_rolling_extra()`.

| Feature | Descripción |
|---|---|
| `HT_Goals_Diff` | Diferencia de goles rolling del local jugando en casa (últimos 5 como local) |
| `AT_Goals_Diff` | Diferencia de goles rolling del visitante jugando fuera (últimos 5 como visitante) |
| `AT_HTR_Rate` | % partidos en que el visitante ganaba al descanso (últimos 5 partidos) |
| `PS_vs_Avg_H` | Señal sharp: diferencia entre prob. implícita Pinnacle local y promedio de mercado |

**Incluidas en FEATURES_SELECCIONADAS:** `HT_Goals_Diff`, `PS_vs_Avg_H`. Excluidas: `AT_Goals_Diff`, `AT_HTR_Rate`.

---

### Categoría 11 — Árbitro (5 features)

**Fuente:** CSVs de temporadas (football-data.co.uk), columna `Referee`.  
**Calculadas en:** `utils.agregar_features_arbitro()`.  
**Ventana:** últimos 20 partidos arbitrados por cada árbitro.

| Feature | Descripción |
|---|---|
| `Ref_Home_WinRate` | % victorias del equipo local en partidos arbitrados por este árbitro |
| `Ref_Goals_Avg` | Promedio de goles totales por partido con este árbitro |
| `Ref_Yellow_Avg` | Promedio de tarjetas amarillas totales con este árbitro |
| `Ref_Home_Yellow` | Promedio de amarillas al equipo local con este árbitro |
| `Ref_Away_Yellow` | Promedio de amarillas al equipo visitante con este árbitro |

**Todas incluidas en FEATURES_SELECCIONADAS:** `Ref_Away_Yellow`, `Ref_Yellow_Avg`, `Ref_Home_WinRate`, `Ref_Goals_Avg`.  
`Ref_Home_Yellow` no fue seleccionada por permutation importance.

---

### Categoría 12 — Ratings ELO (3 features) — Disponibles, no incluidas

**Fuente:** `datos/raw/elo/`, calculados externamente.  
**Calculadas en:** `utils.agregar_features_elo()`.

| Feature | Descripción |
|---|---|
| `HT_ELO` | Rating ELO del local en la fecha del partido |
| `AT_ELO` | Rating ELO del visitante en la fecha del partido |
| `ELO_Diff` | Diferencia de ratings: HT_ELO − AT_ELO |

**No incluidas en `ALL_FEATURES` ni en `FEATURES_SELECCIONADAS`:** en pruebas controladas con XGBoost, estas features **redujeron el F1 en −0.014** respecto a no incluirlas. Se mantienen disponibles en el código para futura experimentación.

---

### Resumen de selección de features

| Categoría | Disponibles | En modelo |
|---|---|---|
| Estadísticas Base Rolling | 10 | 6 |
| Cuotas Bet365 | 6 | 0 (via derivadas) |
| Señales Derivadas de Cuotas | 10 | 5 |
| Expected Goals (xG) | 6 | 5 |
| Head-to-Head | 9 | 5 |
| Posición en Tabla | 11 | 9 |
| Forma y Momentum | 15 | 8 |
| Pinnacle / Sharp Money | 6 | 6 |
| Asian Handicap | 7 | 7 |
| Rolling Adicionales | 4 | 2 |
| Árbitro | 5 | 4 |
| ELO | 3 | 0 |
| **Total** | **84** | **55** |

Las 55 features finales se seleccionaron en dos pasos:
1. **Top 45** por permutation importance (RF balanced, n_repeats=5, evaluado en el último fold de TimeSeriesSplit(n_splits=5) sobre el 80% de entrenamiento)
2. **+ 10 candidatas** de Pinnacle Sharp y Asian Handicap, añadidas manualmente por su valor informativo conocido y validadas empíricamente (mejoran el F1)

---

## Modelos y Entrenamiento

### Los tres modelos competidores

En cada ejecución de `pipeline/02_entrenar_modelo.py` se entrenan y comparan tres modelos sobre el mismo split temporal 80/20:

| Modelo | Descripción |
|---|---|
| **XGBoost** | Gradient boosting con pesos de muestra. Se entrena primero para evitar no-determinismo causado por contaminación del pool de threads OpenMP de scikit-learn |
| **RF Balanceado** | Random Forest con pesos de clase co-optimizados. Arquitectura fija (300 estimadores, max_depth=10) con pesos ajustados por Optuna |
| **RF Optuna** | Random Forest con arquitectura y pesos de clase co-optimizados simultáneamente con Optuna |

### Co-optimización con Optuna

Los pesos de clase y los hiperparámetros de cada modelo se optimizan **conjuntamente** en un único estudio Optuna de **150 trials** con TPESampler (búsqueda bayesiana). El objetivo es el F1-weighted en validación cruzada temporal (TimeSeriesSplit con 5 folds), penalizado si el último fold tiene un F1 mucho menor que la media (para controlar el sobreajuste en datos recientes).

Los estudios de optimización se ejecutan con los scripts de `herramientas/`:
- `herramientas/buscar_pesos_clase.py --solo-rf` → optimiza RF Balanceado y RF Optuna
- `herramientas/buscar_pesos_xgb.py` → optimiza XGBoost (RAM-friendly: n_jobs=1 en trials)

Los mejores parámetros se escriben automáticamente en `config.py` y se guardan como JSON en `modelos/`.

### Calibración condicional

Tras seleccionar el modelo ganador, se prueba Platt Scaling (`CalibratedClassifierCV`, method='sigmoid', cv=3). La calibración solo se adopta si mejora el F1-weighted en test. En la versión actual, la calibración no mejora el F1 y se descarta — el modelo de producción es el XGBoost sin calibrar.

### Resultados actuales (versión 2026-03)

| Modelo | F1-weighted (test 20%) |
|---|---|
| **XGBoost** (ganador) | **0.5102** |
| RF Optuna | 0.5057 |
| RF Balanceado | 0.4907 |
| Baseline original (Ensemble RF40%+XGB60%, 25 feats) | 0.5084 |

**Configuración del modelo ganador (XGBoost):**

```python
PESOS_XGB  = {0: 0.9469, 1: 1.5348, 2: 1.1303}   # Local / Empate / Visitante

PARAMS_XGB = {
    'n_estimators':      350,
    'max_depth':         6,
    'learning_rate':     0.00566,
    'subsample':         0.5697,
    'colsample_bytree':  0.8134,
    'colsample_bylevel': 0.6459,
    'reg_alpha':         0.7067,
    'reg_lambda':        2.0347,
    'min_child_weight':  20,
    'gamma':             0.9796,
}
```

### Top 10 features más importantes (XGBoost, importancia por ganancia)

| # | Feature | Importancia | Categoría |
|---|---|---|---|
| 1 | `Pinnacle_Sharp_H` | 0.0783 | Pinnacle Sharp |
| 2 | `Pinnacle_Sharp_A` | 0.0680 | Pinnacle Sharp |
| 3 | `Prob_A` | 0.0626 | Cuotas Derivadas |
| 4 | `AH_Edge_Home` | 0.0485 | Asian Handicap |
| 5 | `Pinnacle_Conf` | 0.0269 | Pinnacle Sharp |
| 6 | `Prob_Spread` | 0.0231 | Cuotas Derivadas |
| 7 | `Match_Type` | 0.0183 | Posición en Tabla |
| 8 | `HT_HomeWinRate5` | 0.0176 | Forma y Momentum |
| 9 | `Position_Reliability` | 0.0163 | Posición en Tabla |
| 10 | `Position_Diff_Weighted` | 0.0156 | Posición en Tabla |

Las tres primeras posiciones están dominadas por señales de mercado Pinnacle, lo que confirma la hipótesis de que los mercados sharp contienen información predictiva que no está plenamente capturada por las estadísticas de rendimiento.

### Dataset de entrenamiento

| Métrica | Valor |
|---|---|
| Partidos totales | 3 691 |
| Rango temporal | 2016-08-13 a 2026-02-23 |
| Partidos de train (80%) | 2 952 |
| Partidos de test (20%) | 739 |
| Split | Temporal sin shuffle (los 739 más recientes = test) |
| Distribución test | Local 41.8% / Empate 25.0% / Visitante 33.2% |

---

## Rendimiento del Modelo

| Métrica | Valor |
|---|---|
| **F1-weighted (test)** | **0.5102** |
| Accuracy (test) | 52.91% |
| F1 Victoria local | ~0.64 |
| F1 Empate | ~0.22 (clase más difícil) |
| F1 Victoria visitante | ~0.57 |
| Recall Victoria local | 70.55% |
| Recall Empate | 17.84% |
| Recall Victoria visitante | 57.14% |

> El baseline aleatorio para un problema de 3 clases es 33.3%. Un accuracy del 52.91% y un F1-weighted de 0.5102 representan una mejora significativa sobre el azar y son consistentes con la literatura académica en predicción deportiva con datos de mercado.

El empate es estructuralmente la clase más difícil de predecir: históricamente ocurre el 23-25% de las veces pero es la más impredecible incluso para los mercados de apuestas.

---

## Sistema de Value Betting

El módulo `core/sistema_expected_value.py` y el orquestador `jornada/predecir_jornada_completa.py` implementan un sistema de 3 capas para identificar y dimensionar apuestas de valor.

### Capa 1 — Ajuste conservador de probabilidades

Las probabilidades del modelo se ajustan regresando un **40% hacia la distribución uniforme** (1/3 por clase). Esto reduce el overconfidence y penaliza predicciones extremas:

```
prob_ajustada = 0.60 × prob_modelo + 0.40 × (1/3)
```

### Capa 2 — Filtros de calidad de apuesta

Una apuesta solo pasa si cumple **todos** los criterios:

- Edge mínimo del **10%** sobre la probabilidad implícita de la cuota
- Cuota máxima de **5.0** (evitar underdogs extremos)
- Probabilidad mínima del modelo: **35%**

### Capa 3 — Dimensionado con Kelly Criterion

```
f* = (p × b − q) / b

donde:
  p = probabilidad estimada por el modelo (ajustada)
  b = beneficio neto por unidad apostada (cuota − 1)
  q = 1 − p
```

Se aplica **Kelly fraccionario al 25%** del Kelly completo, con un máximo de **2.5% del bankroll** por apuesta.

### Funciones principales

| Función | Descripción |
|---|---|
| `calcular_ev(prob, odds, stake)` | Calcula el Expected Value de una apuesta |
| `kelly_criterion(prob, odds, fraction)` | Calcula el tamaño óptimo de apuesta |
| `analizar_apuesta(prediccion, odds, stake, bankroll)` | Análisis completo con clasificación de la apuesta |
| `simular_roi_historico(...)` | Simulación de ROI histórico con 3 estrategias comparadas |

---

## Generación de Reportes

El script `jornada/predecir_jornada_completa.py` genera reportes automáticos cada jornada:

### Flujo operativo

1. Editar `jornada/jornada_config.py` con los fixtures y cuotas Bet365 actuales
2. Ejecutar `python jornada/predecir_jornada_completa.py`
3. El reporte aparece en `reportes/predicciones_jornada_XX_YYYYMMDD_HHMMSS.pdf`

### Reporte PDF

- Resumen ejecutivo de la jornada
- Análisis individual por partido: probabilidades por resultado, confianza, comparativa contra el mercado implícito
- Tabla de value bets detectadas con EV y stake Kelly recomendado

### Reporte Excel

- Tabla resumen de todos los partidos de la jornada
- Probabilidades por resultado, predicción, edge vs. mercado y cuotas de referencia

---

## Herramientas de Optimización y Experimentación

Los scripts en `herramientas/` cubren el ciclo de vida de investigación del modelo. Los más relevantes actualmente:

### Optimización principal

| Script | Descripción |
|---|---|
| `buscar_pesos_clase.py` | Maestro de optimización para los dos RF. Carga las 84 features disponibles (ALL_FEATURES), las filtra según FEATURES_SELECCIONADAS de config.py, y lanza 150 trials Optuna co-optimizando pesos de clase + hiperparámetros para RF Balanceado y RF Optuna simultáneamente. Soporta `--solo-rf` para omitir la selección de features por permutation importance. Actualiza `config.py` automáticamente. |
| `buscar_pesos_xgb.py` | Co-optimización Optuna para XGBoost: 150 trials, warm start con params actuales, n_jobs=1 en trials (RAM-friendly) y n_jobs=-1 solo en la evaluación final. Actualiza `PESOS_XGB` y `PARAMS_XGB` en `config.py` y guarda `modelos/optuna_xgb_nuevas_feats.json`. |

### Mantenimiento de datos

| Script | Descripción |
|---|---|
| `scrape_xg_understat.py` | Descarga datos de Expected Goals de Understat.com para la temporada en curso y actualiza `datos/raw/final_matches_xg.csv`. Soporta flags `--season`, `--dry-run` y `--csv`. |
| `analisis_rapido.py` | Diagnóstico rápido del dataset canónico: número de partidos, distribución de resultados, features disponibles y NaN por columna. |
| `verify_setup.py` | Verifica que el dataset canónico existe, tiene las columnas requeridas y está en buen estado. Herramienta de sanidad del entorno antes de entrenar. |

### Análisis

| Script | Descripción |
|---|---|
| `analizar_features.py` | Análisis completo de importancia y redundancia de features usando RF. Imprime ranking, identifica candidatas a eliminar y guarda gráficos en `modelos/`. |
| `visualizacion.py` | Genera los 6 gráficos PNG a 300 DPI para portafolio web (matriz de confusión, feature importance, F1 por clase, evolución de versiones, simulación ROI, caso de estudio). |

### Scripts de experimentación histórica

Existen numerosos scripts `optimizar_*.py`, `experimento_*.py` y `fast_search*.py` en `herramientas/` que documentan el historial de experimentos: pruebas con distintas cantidades de features (25, 27, 45, 84), distintos modelos (LightGBM, ExtraTrees — ya eliminados del pipeline principal), distintos horizontes temporales y comparativas de arquitecturas. Se conservan como registro de la evolución del proyecto.

---

## Instalación y Ejecución

### 1. Clonar el repositorio y crear entorno virtual

```bash
git clone <url-del-repo>
cd mi_predictor_pl

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
# o como paquete editable:
pip install -e .
```

### 3. Ejecutar el pipeline completo

```bash
# Paso 1: Preparar datos y construir features
python pipeline/01_preparar_datos.py

# Paso 2: Entrenar modelos y seleccionar el ganador
python pipeline/02_entrenar_modelo.py

# Paso 3: Predecir la jornada actual
#   (editar primero jornada/jornada_config.py con los fixtures)
python jornada/predecir_jornada_completa.py
```

### 4. Re-optimizar hiperparámetros (opcional)

```bash
# Re-optimizar los dos RF con las features actuales de config.py (150 trials cada uno)
python herramientas/buscar_pesos_clase.py --solo-rf

# Re-optimizar XGBoost (150 trials, RAM-friendly)
python herramientas/buscar_pesos_xgb.py

# Después de cualquier re-optimización, volver a ejecutar el pipeline:
python pipeline/02_entrenar_modelo.py
```

### 5. Actualizar datos de xG

```bash
# Descargar xG de la temporada en curso de Understat
python herramientas/scrape_xg_understat.py --season 2025-26

# Luego volver a ejecutar el pipeline completo
python pipeline/01_preparar_datos.py
python pipeline/02_entrenar_modelo.py
```

---

## Notas de Desarrollo

- El split de entrenamiento/validación es siempre **temporal sin shuffle** (`train_test_split(..., shuffle=False)`). Los últimos 739 partidos cronológicamente son siempre el conjunto de test. Esto evita data leakage entre partidos de distintas fechas.
- Las features rolling usan `shift(1)` explícito: el partido actual nunca se incluye en su propia ventana de cálculo.
- **No-determinismo de XGBoost con n_jobs=-1:** XGBoost puede dar F1 ligeramente distintos entre ejecuciones si el pool de threads OpenMP ya fue inicializado por scikit-learn. Por eso XGBoost se entrena **primero** en el pipeline (antes que los RF). El F1 de referencia es 0.5102 en proceso limpio con las 55 features actuales.
- Los modelos se guardan en `modelos/` como archivos `.pkl` junto con la lista exacta de features usadas y metadatos del entrenamiento, para garantizar reproducibilidad en predicción.
- `config.py` es la única fuente de verdad para todos los hiperparámetros. Los scripts de optimización actualizan `config.py` automáticamente al terminar.
- La imputación de Asian Handicap usa la **mediana** de los partidos con datos disponibles — nunca 0, porque `AHh=0` significa "partido parejo" y no "dato ausente".

---

## Disclaimer

Este proyecto fue desarrollado con fines **estrictamente educativos**. No constituye consejo financiero, de inversión ni de apuestas. Los resultados históricos simulados no garantizan rendimientos futuros. Las apuestas deportivas implican riesgo real de pérdida de capital.
