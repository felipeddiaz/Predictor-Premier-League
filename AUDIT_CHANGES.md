# Audit ML - Premier League Predictor

## Resumen Ejecutivo

Se implementaron 2 mejoras principales al sistema de prediccion, optimizadas para **ROI en value betting**:

1. **Seleccion de Features**: 98 → 35 features (reduccion de overfitting)
2. **Elo Ratings**: Feature compacta de fuerza relativa de equipos

**Criterio de seleccion de modelo: ROI** (no Log Loss). El objetivo es encontrar valor en apuestas.

---

## 1. Seleccion de Features (98 → 35)

### Problema
El modelo usaba 98 features, muchas redundantes o de bajo valor predictivo. Esto causaba overfitting y pobre generalizacion.

### Solucion
Se entrena un XGBoost rapido (200 arboles, max_depth=5) para rankear features por importancia, y se seleccionan las top N (configurable, default=35).

### Implementacion

**`config.py`**:
```python
N_FEATURES_SELECCION = 35  # nuevo parametro
```

**`pipeline/02_entrenar_modelo.py`** — nueva funcion `seleccionar_features()`:
```python
def seleccionar_features(X_train, y_train, features, n_top=N_FEATURES_SELECCION):
    selector = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, ...)
    selector.fit(X_train, y_train, sample_weight=sample_weights)
    importances = selector.feature_importances_
    # Retorna top N features ordenadas por importancia
```

### Top 10 Features (por importancia en modelo final)

| # | Feature | Importancia |
|---|---------|-------------|
| 1 | Home_Advantage_Prob | 0.0827 |
| 2 | Pinnacle_Open_H | 0.0786 |
| 3 | B365A | 0.0645 |
| 4 | Pinnacle_Open_A | 0.0639 |
| 5 | B365H | 0.0636 |
| 6 | Prob_A | 0.0565 |
| 7 | AH_Edge_Home | 0.0515 |
| 8 | Pinnacle_Conf | 0.0297 |
| 9 | Market_Confidence | 0.0292 |
| 10 | Prob_D | 0.0262 |

Features Elo en el top 35: HT_Elo (#22), AT_Elo (#18), Elo_Diff (#20), Elo_WinProb_H (#12).

### Resultado
Menos features = menos ruido, mejor generalizacion, entrenamiento mas rapido.

---

## 2. Elo Ratings

### Problema
No habia una feature que capturara la fuerza historica acumulada de cada equipo de forma compacta.

### Solucion
Sistema Elo incremental inspirado en el ajedrez, adaptado al futbol:

| Parametro | Valor | Descripcion |
|-----------|-------|-------------|
| K-factor | 20 | Velocidad de ajuste por partido |
| Home advantage | 50 | Bonus Elo para el local |
| Elo inicial | 1500 | Rating de arranque |
| Regresion temporada | 0.33 | Al inicio de cada temporada, 33% regresion a la media |

### Features Generadas

| Feature | Descripcion |
|---------|-------------|
| `HT_Elo` | Rating Elo del local antes del partido |
| `AT_Elo` | Rating Elo del visitante antes del partido |
| `Elo_Diff` | HT_Elo - AT_Elo |
| `Elo_WinProb_H` | Probabilidad implicita Elo de victoria local: `1 / (1 + 10^(-Elo_Diff/400))` |

### Implementacion

**`utils.py`** — nueva funcion `agregar_features_elo()`:
- Recorre partidos cronologicamente
- Usa `shift(1)` implicito: el Elo asignado a un partido es el previo al partido
- Al inicio de cada temporada nueva, regresiona todos los ratings 33% hacia 1500
- Los equipos nuevos (ascendidos) arrancan con Elo=1500

**`config.py`**:
```python
FEATURES_ELO = ['HT_Elo', 'AT_Elo', 'Elo_Diff', 'Elo_WinProb_H']
```
Agregadas a `FEATURES_CON_CUOTAS_APERTURA` y `FEATURES_ESTRUCTURALES`.

### Impacto
4 de 4 features Elo fueron seleccionadas en el top 35, confirmando su valor predictivo.

---

## Archivos Modificados

| Archivo | Cambios |
|---------|---------|
| `config.py` | +`N_FEATURES_SELECCION=35`, +`FEATURES_ELO` (4 features), actualizado `FEATURES_CON_CUOTAS_APERTURA` y `FEATURES_ESTRUCTURALES` |
| `utils.py` | +funcion `agregar_features_elo()` |
| `pipeline/02_entrenar_modelo.py` | +`seleccionar_features()`, reescrito `main()`, seleccion por ROI, +import `agregar_features_elo` |
| `pipeline/03_entrenar_sin_cuotas.py` | +import y call `agregar_features_elo` en carga de datos |

---

## Resultados: Seleccion por ROI

### Modelo Principal: XGBoost (35 features seleccionadas)

| Modelo | ROI | Log Loss | Brier Score | Accuracy |
|--------|-----|----------|-------------|----------|
| **XGBoost** | **+24.22%** | 0.9829 | 0.1952 | 52.37% |
| RF Balanceado | +17.70% | 0.9851 | 0.1955 | 51.83% |
| RF Optuna | +10.90% | 0.9937 | 0.1984 | 48.31% |
| RF Basico | -3.83% | 0.9900 | 0.1963 | 52.23% |

**Ganador: XGBoost** — mejor ROI (+24.22%) Y mejor Log Loss (0.9829).

### Modelo Estructural (sin cuotas, 78 features)

| Metrica | Valor |
|---------|-------|
| Log Loss | 0.9921 |
| Brier Score | 0.1975 |
| Accuracy | 53.45% |

### Walk-Forward Validation (temporada por temporada)

| Temporada | Train | Test | Log Loss | ROI |
|-----------|-------|------|----------|-----|
| 2020-21 | 1520 | 380 | 1.0174 | +16.62% |
| 2021-22 | 1900 | 380 | 0.9665 | +37.00% |
| 2022-23 | 2280 | 380 | 0.9969 | -16.30% |
| 2023-24 | 2660 | 380 | 0.9318 | +0.17% |
| 2024-25 | 3040 | 380 | 0.9747 | +40.60% |
| 2025-26 | 3420 | 271 | 1.0183 | +27.15% |
| **PROMEDIO** | | | **0.9843** | **+17.54%** |

ROI positivo en 5 de 6 temporadas. STD Log Loss: 0.0305 (consistente).

---

## Artefactos Generados

| Archivo | Contenido |
|---------|-----------|
| `modelos/modelo_final_optimizado.pkl` | XGBoost (35 features, seleccionado por ROI) |
| `modelos/modelo_value_betting.pkl` | XGBoost estructural (78 features, sin cuotas) |
| `modelos/features.pkl` | Lista de 35 features seleccionadas |
| `modelos/features_value_betting.pkl` | Lista de 78 features estructurales |
| `modelos/metadata.pkl` | Metadata del entrenamiento |
| `modelos/confusion_matrix_final.png` | Matriz de confusion del modelo final |
| `modelos/feature_importance_final.png` | Importancia de features |

---

## Branch

- **Branch**: `claude/football-ml-audit-EKYmn`
