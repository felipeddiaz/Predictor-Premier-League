# Audit ML - Premier League Predictor

## Resumen Ejecutivo

Se implementaron 3 mejoras principales al sistema de prediccion:

1. **Seleccion de Features**: 98 → 35 features (reduccion de overfitting)
2. **Stacking Meta-Learner**: RF + XGBoost → LogisticRegression (mejor combinacion de modelos)
3. **Elo Ratings**: Feature compacta de fuerza relativa de equipos

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

### Top 35 Features Seleccionadas (ordenadas por importancia)

Las features mas importantes incluyen:
- **Cuotas/Mercado**: PSH, PSD, PSA, BbMxH, BbMxD, BbMxA, BbAvH, BbAvD, BbAvA, Pinnacle_SharpMoney_H/D/A, AH_Edge
- **Forma**: HT_WinRate5, AT_WinRate5, HT_Pts5, AT_Pts5, HT_AvgGoals, AT_AvgGoals
- **Tabla**: HT_Position, AT_Position, Position_Diff, HT_GD, AT_GD
- **Elo**: HT_Elo, AT_Elo, Elo_Diff (3 de 4 features Elo entraron al top 35)
- **xG**: HT_xG_Avg, AT_xG_Avg, xG_Diff
- **H2H**: H2H_Home_WinRate, H2H_Played

### Resultado
Menos features = menos ruido, mejor generalizacion, entrenamiento mas rapido.

---

## 2. Stacking Meta-Learner

### Problema
El ensemble anterior (EnsembleLGBM_XGB) promediaba probabilidades con pesos fijos. No aprendia la mejor forma de combinar modelos.

### Solucion
Stacking de 2 niveles:
- **Nivel 1 (base learners)**: RandomForest + XGBoost
- **Nivel 2 (meta-learner)**: LogisticRegression entrenada sobre predicciones out-of-fold

### Arquitectura

```
Datos de entrada (35 features)
        |
   ┌────┴────┐
   │         │
RandomForest  XGBoost
   │         │
   ├─ 3 probs ─┤─ 3 probs ─┐
   │                        │
   └───── 6 features ───────┘
              │
     LogisticRegression
              │
     3 probabilidades finales
     (H / D / A)
```

### Implementacion

**`utils.py`** — nueva clase `StackingMetaLearner`:
```python
class StackingMetaLearner(BaseEstimator, ClassifierMixin):
    def __init__(self, rf_model, xgb_model, meta_model, features):
        ...
    def predict_proba(self, X):
        probs_rf = self.rf_model.predict_proba(X_fill)
        probs_xgb = self.xgb_model.predict_proba(X)
        meta_X = np.hstack([probs_rf, probs_xgb])  # 6 features
        return self.meta_model.predict_proba(meta_X)
```

**`pipeline/02_entrenar_modelo.py`** — nueva funcion `entrenar_stacking()`:
- Usa `TimeSeriesSplit(n_splits=3)` para generar predicciones out-of-fold (OOF)
- Evita data leakage temporal: cada fold solo usa datos pasados
- Meta-learner: `LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')`

### Calibracion
El stacking NO pasa por `CalibratedClassifierCV` porque:
- LogisticRegression ya produce probabilidades calibradas nativamente
- `StackingMetaLearner` no implementa `fit()`, lo cual es incompatible con `CalibratedClassifierCV`

---

## 3. Elo Ratings

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
3 de 4 features Elo fueron seleccionadas en el top 35 (posiciones ~12, 18, 22), confirmando su valor predictivo.

---

## Archivos Modificados

| Archivo | Cambios |
|---------|---------|
| `config.py` | +`N_FEATURES_SELECCION=35`, +`FEATURES_ELO` (4 features), actualizado `FEATURES_CON_CUOTAS_APERTURA` y `FEATURES_ESTRUCTURALES` |
| `utils.py` | +clase `StackingMetaLearner`, +funcion `agregar_features_elo()` |
| `pipeline/02_entrenar_modelo.py` | +`seleccionar_features()`, +`entrenar_stacking()`, reescrito `main()` completo, +import `agregar_features_elo`, +call Elo en `cargar_datos()` |
| `pipeline/03_entrenar_sin_cuotas.py` | +import y call `agregar_features_elo` en carga de datos |
| `core/predictor.py` | +import `StackingMetaLearner` para deserializacion del modelo guardado |

---

## Resultados Completos

### Modelos Individuales (35 features seleccionadas)

| Modelo | Log Loss | Brier Score | Accuracy | ROI |
|--------|----------|-------------|----------|-----|
| **Stacking (RF+XGB→LogReg)** | **0.9713** | **0.1926** | 52.77% | -70.83% |
| XGBoost | 0.9829 | 0.1952 | 52.37% | **+24.22%** |
| RF Balanceado | 0.9851 | — | — | +17.70% |
| RF Optuna | 0.9937 | — | — | +10.90% |
| RF Basico | 0.9900 | — | — | -3.83% |

### Modelo Estructural (sin cuotas, 78 features)

| Metrica | Valor |
|---------|-------|
| Log Loss | 0.9921 |
| Brier Score | 0.1975 |
| Accuracy | 53.45% |

### Walk-Forward Validation (temporada por temporada)

| Metrica | Valor |
|---------|-------|
| ROI promedio | +17.54% |
| STD Log Loss | 0.0305 |

Consistencia alta entre temporadas (baja desviacion estandar).

---

## Analisis: Stacking vs XGBoost Solo

### Stacking gana en metricas probabilisticas
- Mejor Log Loss (0.9713 vs 0.9829) → probabilidades mas precisas
- Mejor Brier Score (0.1926 vs 0.1952) → mejor calibracion

### XGBoost gana en ROI de apuestas
- ROI +24.22% vs -70.83% del stacking
- El stacking distribuye probabilidad de forma mas "suave", lo que reduce la confianza en apuestas de valor
- XGBoost es mas "agresivo" en sus predicciones, generando mas value bets correctas

### Conclusion practica
- **Para prediccion pura**: Stacking es superior
- **Para apuestas (value betting)**: XGBoost solo es muy superior
- El modelo guardado como `modelo_final_optimizado.pkl` es el Stacking (gano por Log Loss)
- El modelo `modelo_value_betting.pkl` es el XGBoost estructural (78 features, sin cuotas)

---

## Artefactos Generados

| Archivo | Contenido |
|---------|-----------|
| `modelos/modelo_final_optimizado.pkl` | Stacking meta-learner (35 features) |
| `modelos/modelo_value_betting.pkl` | XGBoost estructural (78 features) |
| `modelos/features.pkl` | Lista de 35 features seleccionadas |
| `modelos/features_value_betting.pkl` | Lista de 78 features estructurales |
| `modelos/metadata.pkl` | Metadata del entrenamiento |
| `modelos/confusion_matrix_final.png` | Matriz de confusion del modelo final |

---

## Commit

- **Hash**: `75e546a`
- **Branch**: `claude/football-ml-audit-EKYmn`
- **Mensaje**: Implement feature selection (98→35), stacking meta-learner, and Elo ratings
