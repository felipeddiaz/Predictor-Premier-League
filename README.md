# Predictor de Partidos — Premier League

> Sistema de predicción de resultados de la Premier League con Machine Learning, ingeniería de features avanzada, optimización bayesiana y sistema de value betting con Expected Value y Kelly Criterion.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![Optuna](https://img.shields.io/badge/Optuna-3.x-green)
![Estado](https://img.shields.io/badge/Estado-En%20desarrollo-yellow)

---

## Descripcion del Proyecto

Este proyecto predice el resultado de partidos de la Premier League inglesa (victoria local, empate o victoria visitante) usando un pipeline completo de Machine Learning que cubre desde la recoleccion y limpieza de datos historicos hasta la generacion de reportes PDF/Excel por jornada.

Ademas de predecir resultados, el sistema identifica **value bets**: situaciones donde la probabilidad estimada por el modelo difiere significativamente de las cuotas del mercado, calculando el Expected Value (EV) y el tamano de apuesta optimo mediante el criterio de Kelly.

### Que problema resuelve?

Los modelos de prediccion deportiva simples suelen limitarse a clasificar resultados. Este proyecto va mas alla:

- Calibra las probabilidades del modelo para que sean realistas (no solo rankings)
- Compara esas probabilidades contra las cuotas implicitas del mercado
- Aplica tres capas de filtrado de riesgo antes de recomendar una apuesta
- Genera reportes automatizados por jornada con analisis completo

> **Disclaimer:** Este proyecto es estrictamente educativo. No constituye consejo financiero ni garantiza ganancias. Las apuestas deportivas conllevan riesgo de perdida de capital.

---

## Tecnologias y Librerias

| Categoria | Tecnologia |
|---|---|
| Lenguaje | Python 3.9+ |
| Manipulacion de datos | pandas, numpy |
| Machine Learning | scikit-learn (RandomForest, CalibratedClassifierCV, TimeSeriesSplit) |
| Optimizacion bayesiana | Optuna (TPESampler, 100 trials) |
| Gradient Boosting (opcional) | XGBoost |
| Balance de clases (opcional) | imbalanced-learn / SMOTE |
| Persistencia de modelos | joblib |
| Visualizacion | matplotlib, seaborn |
| Generacion de reportes PDF | fpdf |
| Exportacion Excel | openpyxl |

---

## Estructura del Proyecto

```
mi_predictor_pl/
│
├── 01_preparar_datos.py                  # Pipeline de limpieza y feature engineering
├── 02_entrenar_modelo.py                 # Entrenamiento RF + Optuna + calibracion Platt
├── 03_entrenar_con_cuotas.py             # Modelo alternativo orientado a value betting
├── 04_analizar_feature_importance.py     # Analisis de importancia y redundancia de features
│
├── predecir_jornada_completa.py          # Prediccion de jornada completa → PDF + Excel
├── sistema_expected_value.py             # Libreria de EV, Kelly Criterion y simulacion ROI
├── visualizacion..py                     # Genera 6 graficos de portafolio (300 DPI)
├── agregar_features_derivadas.py         # Script puntual: añade Goal_Diff/Form_Diff/Shots_Diff
├── analisis_rapido.py                    # Diagnostico rapido del CSV procesado
├── verify_setup.py                       # Verifica integridad del dataset canonico
│
├── requirements.txt
├── README.md
│
├── datos/
│   ├── temporadas/                       # CSVs originales por temporada (20-21 a 25-26)
│   ├── procesados/                       # Datasets intermedios y dataset canonico con features
│   └── raw/
│       └── final_matches_xg.csv          # Datos externos de xG (Expected Goals)
│
├── modelos/                              # Modelos .pkl entrenados, feature lists y metricas
│   ├── modelo_final_optimizado.pkl
│   ├── features.pkl
│   ├── metadata.pkl
│   └── *.png                            # Matriz de confusion, feature importance, calibracion
│
└── portafolio_imagenes/                  # Graficos de alta calidad para portafolio web
    ├── 01_matriz_confusion.png
    ├── 02_feature_importance.png
    ├── 03_f1_por_clase.png
    ├── 04_evolucion_modelo.png
    ├── 05_roi_simulation.png
    └── 06_caso_estudio.png
```

---

## Pipeline de Trabajo

El proyecto sigue un flujo de 4 pasos secuenciales:

```
datos/temporadas/*.csv
        │
        ▼
01_preparar_datos.py          →  datos/procesados/premier_league_con_features.csv
        │
        ▼
02_entrenar_modelo.py         →  modelos/modelo_final_optimizado.pkl
        │
        ▼
predecir_jornada_completa.py  →  predicciones_jornada_XX.pdf / .xlsx
        │
        ▼
sistema_expected_value.py     →  EV + Kelly Criterion por apuesta recomendada
```

---

## Ingenieria de Features

El script `01_preparar_datos.py` construye las variables predictoras en 4 fases:

### Fase 1 — Estadisticas Rolling (ventana de 5 partidos)
- Promedio de goles anotados y encajados por equipo (perspectivas local y visitante)
- Promedio de tiros a puerta realizados y recibidos
- Se usa `shift(1)` para evitar data leakage (el partido actual no se incluye en su propia media)

### Fase 2 — Features de Forma
- Conteo de victorias, empates y derrotas en los ultimos 5 partidos por equipo
- Calculadas por separado para rol local y visitante

### Fase 3 — Head-to-Head (H2H)
- Historial de los ultimos 5 enfrentamientos directos entre cada par de equipos
- Variables: tasa de victorias local/visitante, media de goles, tasa de ambos marcan (BTTS)
- Flag de disponibilidad (`H2H_Available`) cuando hay enfrentamientos previos suficientes
- Features derivadas de diferencia: `H2H_Win_Diff`, `H2H_Goals_Diff`

### Fase 4 — Expected Goals (xG) Rolling
- Promedio de xG creado y concedido por equipo en los ultimos 5 partidos
- Solo cuando existe el dataset externo `final_matches_xg.csv`
- Features derivadas: `xG_Diff`, `xG_Total`

### Features adicionales (calculadas en memoria durante el entrenamiento)
- **Cuotas de mercado**: Bet365 apertura y cierre, probabilidades implicitas, movimiento de mercado
- **Posicion en tabla**: posicion, puntos, progreso de temporada, tipo de partido (duelo de grandes, zona baja, etc.)
- **Features de presion**: diferencia de posicion, importancia relativa del partido

---

## Modelos y Entrenamiento

### Comparativa de modelos entrenados (`02_entrenar_modelo.py`)

| Modelo | Descripcion |
|---|---|
| RF Basico | Random Forest sin ajuste de pesos de clase |
| RF Balanced | Random Forest con `class_weight='balanced'` |
| **RF Optuna** | Random Forest con pesos de clase optimizados por Optuna (ganador) |

### Configuracion del modelo ganador

- **Algoritmo**: Random Forest (`n_estimators=229`, `max_depth=8`, `min_samples_leaf=3`)
- **Pesos de clase**: Local: 1.25, Empate: 3.32, Visitante: 1.95 (optimizados con 100 trials Optuna)
- **Calibracion**: Platt Scaling (`CalibratedClassifierCV` con `method='sigmoid'`)
- **Split temporal**: 80% entrenamiento / 20% validacion sin mezcla aleatoria (respeta el orden cronologico)

### Por que calibracion de probabilidades?

Un Random Forest sin calibrar tiende a producir probabilidades sesgadas hacia 50%. La calibracion Platt Scaling ajusta la salida del modelo para que, si predice 65%, eso se corresponda realmente con un 65% de frecuencia observada historicamente.

---

## Sistema de Value Betting

El modulo `sistema_expected_value.py` y el script `predecir_jornada_completa.py` implementan un sistema de 3 capas para identificar y dimensionar apuestas de valor:

### Capa 1 — Ajuste conservador de probabilidades
Las probabilidades del modelo se ajustan regresando un 40% hacia la distribucion uniforme (1/3 por clase). Esto reduce el overconfidence y penaliza predicciones extremas.

### Capa 2 — Filtros de calidad de apuesta
Una apuesta solo pasa si cumple **todos** los criterios:
- Edge minimo del 10% sobre la probabilidad implicita de la cuota
- Cuota maxima de 5.0 (evitar eventos de muy baja probabilidad)
- Probabilidad minima del modelo: 35%

### Capa 3 — Dimensionado con Kelly Criterion

```
f* = (p * b - q) / b

donde:
  p = probabilidad estimada por el modelo
  b = beneficio neto por unidad apostada (cuota - 1)
  q = 1 - p
```

Se aplica **Kelly fraccionario al 25%** del Kelly completo, con un maximo de 2.5% del bankroll por apuesta.

### Funciones principales de `sistema_expected_value.py`

| Funcion | Descripcion |
|---|---|
| `calcular_ev(prob, odds, stake)` | Calcula el Expected Value de una apuesta |
| `kelly_criterion(prob, odds, fraction)` | Calcula el tamano optimo de apuesta |
| `analizar_apuesta(prediccion, odds, stake, bankroll)` | Analisis completo con clasificacion de la apuesta |
| `simular_roi_historico(...)` | Simulacion de ROI historico con 3 estrategias |

---

## Generacion de Reportes

El script `predecir_jornada_completa.py` genera reportes automaticos por jornada:

### Reporte PDF
- Resumen ejecutivo de la jornada
- Analisis individual por partido: probabilidades, confianza, comparativa contra mercado
- Tabla de value bets con EV y stake Kelly recomendado

### Reporte Excel
- Tabla resumen de todos los partidos de la jornada
- Probabilidades por resultado, prediccion, edge y cuotas de referencia

---

## Visualizaciones de Portafolio

El script `visualizacion..py` genera 6 graficos PNG a 300 DPI listos para portafolio web:

| Archivo | Contenido |
|---|---|
| `01_matriz_confusion.png` | Matriz de confusion del modelo final |
| `02_feature_importance.png` | Top 15 features mas importantes |
| `03_f1_por_clase.png` | Precision, Recall y F1 por clase (Local/Empate/Visitante) |
| `04_evolucion_modelo.png` | Evolucion del accuracy por version del modelo (v1–v5) |
| `05_roi_simulation.png` | Simulacion de ROI a lo largo de una temporada |
| `06_caso_estudio.png` | Caso de estudio: partido sin value bet detectado |

---

## Instalacion y Ejecucion

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

# Dependencias adicionales
pip install fpdf openpyxl optuna
```

### 3. Ejecutar el pipeline completo

```bash
# Paso 1: Limpiar datos y construir features
python 01_preparar_datos.py

# Paso 2: Entrenar modelos (RF basico + balanced + Optuna + calibracion)
python 02_entrenar_modelo.py

# Paso 3 (alternativo): Modelo orientado a value betting sin cuotas como features
python 03_entrenar_con_cuotas.py

# Paso 4: Analizar que features aportan mas al modelo
python 04_analizar_feature_importance.py

# Predecir una jornada completa y generar PDF + Excel
python predecir_jornada_completa.py
```

### 4. Scripts de utilidad

```bash
python analisis_rapido.py         # Diagnostico rapido del CSV procesado
python verify_setup.py            # Verifica integridad del dataset canonico
python visualizacion..py          # Genera los 6 graficos de portafolio
python sistema_expected_value.py  # Demo de calculos EV y Kelly
```

---

## Rendimiento del Modelo

| Metrica | Valor tipico |
|---|---|
| Accuracy | 51–54% |
| F1 ponderado | 0.40–0.50 |
| F1 — Victoria local | ~0.60 |
| F1 — Empate | ~0.31 (clase mas dificil) |
| F1 — Victoria visitante | ~0.50 |
| ROI simulado (value betting) | +10–15% sobre datos historicos |

> El baseline aleatorio para un problema de 3 clases es 33.3%. Un accuracy del 51–54% representa una mejora significativa y consistente con la literatura academica en prediccion deportiva.

---

## Fuentes de Datos

| Fuente | Ubicacion | Descripcion |
|---|---|---|
| Football-data.co.uk | `datos/temporadas/*.csv` | Datos de 6 temporadas EPL (2020/21–2025/26): goles, tiros, corners, faltas, tarjetas y cuotas de multiples casas |
| FBRef / fuente externa | `datos/raw/final_matches_xg.csv` | Expected Goals (xG) por equipo y partido |
| Cuotas de mercado | `predecir_jornada_completa.py` | Cuotas Bet365 apertura/cierre, actualizadas manualmente cada jornada |

### Columnas principales del dataset crudo

| Columna | Descripcion |
|---|---|
| `Date`, `HomeTeam`, `AwayTeam` | Identificacion del partido |
| `FTHG`, `FTAG`, `FTR` | Goles y resultado final (H/D/A) |
| `HS`, `AS`, `HST`, `AST` | Tiros totales y a puerta |
| `HF`, `AF`, `HC`, `AC` | Faltas y corners |
| `HY`, `AY`, `HR`, `AR` | Tarjetas amarillas y rojas |
| `B365H`, `B365D`, `B365A` | Cuotas Bet365 (apertura) |

---

## Notas de Desarrollo

- El split de entrenamiento/validacion es **siempre temporal** (sin shuffle) para evitar data leakage entre partidos de distintas fechas
- Las features rolling usan `shift(1)` explícito: el partido actual nunca se incluye en su propia ventana de calculo
- Los modelos se guardan en `modelos/` como archivos `.pkl` junto con la lista exacta de features usadas, para garantizar reproducibilidad en prediccion
- El script `predecir_jornada_completa.py` requiere actualizacion manual de fixtures y cuotas cada jornada

---

## Disclaimer

Este proyecto fue desarrollado con fines **estrictamente educativos**. No constituye consejo financiero, de inversion ni de apuestas. Los resultados historicos simulados no garantizan rendimientos futuros. Las apuestas deportivas implican riesgo real de perdida de capital.
