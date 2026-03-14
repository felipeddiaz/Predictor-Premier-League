# Plan de Mejora - Auditoría del Sistema de Predicción Premier League

**Fecha de auditoría:** 2026-03-14
**Auditor:** Football ML Value Betting Engineer
**Estado del sistema:** No listo para producción

---

## Resumen Ejecutivo

El sistema presenta una arquitectura sólida con un pipeline reproducible, pero tiene **debilidades estructurales críticas** que comprometen la fiabilidad del edge reportado (+24.22% ROI). El modelo depende excesivamente de señales de mercado (~52% de importancia), tiene un bug de calibración sin resolver, y carece de validación CLV (Closing Line Value) que confirme que el edge es real y no un artefacto.

---

## 1 - Auditoría del Modelo

### Diagnóstico

El modelo XGBoost seleccionado reporta ROI +24.22% en test y +17.54% promedio en walk-forward (5/6 temporadas positivas). Sin embargo, la estructura de importancia de features revela que **el modelo está replicando la opinión del mercado**, no descubriendo información independiente.

### Problemas detectados

| # | Problema | Severidad | Evidencia |
|---|----------|-----------|-----------|
| 1 | **Dependencia extrema de cuotas** | CRITICA | Las 6 features más importantes son señales de mercado (Home_Advantage_Prob=0.0827, Pinnacle_Open_H=0.0786, B365A=0.0645, etc.) |
| 2 | **Bug de calibración** | CRITICA | `02_entrenar_modelo.py:594` guarda `modelo_final` (sin calibrar) en vez de `modelo_calibrado` |
| 3 | **Train/serve mismatch parcial** | ALTA | Prob_Move_H/D/A se setean a 0.0 en producción (`predictor.py:474`) pero el modelo fue entrenado con valores reales |
| 4 | **Sobreajuste temporal** | MEDIA | ROI -16.30% en temporada 2022-23 indica fragilidad ante cambios de mercado |
| 5 | **Modelo estructural sin ROI** | MEDIA | El modelo sin cuotas (78 features) no puede calcular ROI, impidiendo comparar edge independiente |

### Mejoras recomendadas

1. **Entrenar modelo dual**: un modelo SOLO con features independientes (xG, Elo, forma, posición) y comparar su calibración contra el modelo completo
2. **Corregir bug de calibración**: guardar `modelo_calibrado` cuando mejora Brier Score
3. **Eliminar features con mismatch train/serve**: si un feature no está disponible en producción con el mismo valor que en entrenamiento, eliminarlo

### Experimentos sugeridos

- **Exp-1**: Entrenar XGBoost solo con features independientes → medir Log Loss, Brier Score, accuracy
- **Exp-2**: Entrenar XGBoost con cuotas de apertura únicamente (sin derivadas) → comparar con modelo actual
- **Exp-3**: Medir SHAP values por temporada para detectar concept drift en la importancia de features

### Prioridad: **ALTA**

---

## 2 - Evaluación Cuantitativa

### Diagnóstico

El sistema reporta métricas razonables (Log Loss=0.9829, Brier=0.1952, Accuracy=52.37%) pero **carece de las métricas fundamentales para validar un sistema de value betting**.

### Problemas detectados

| # | Problema | Severidad | Evidencia |
|---|----------|-----------|-----------|
| 1 | **Sin CLV tracking** | CRITICA | No se valida si las predicciones superan la línea de cierre |
| 2 | **Sin simulación Monte Carlo** | ALTA | No hay análisis de drawdown, riesgo de ruina, o distribución de retornos |
| 3 | **ROI potencialmente inflado** | ALTA | Edge calculado sin eliminar vig (~5%), inflando el edge aparente |
| 4 | **Sin test prospectivo real** | ALTA | 2025-26 está en progreso pero no hay tracking live documentado |
| 5 | **Varianza de ROI no reportada** | MEDIA | Walk-forward muestra rango [-16.3%, +40.6%] pero no se calcula intervalo de confianza |

### Mejoras recomendadas

1. **Implementar CLV tracker completo** en `core/clv_tracker.py`:
   - Registrar cuota de apertura al momento de la predicción
   - Registrar cuota de cierre post-partido
   - Calcular: `CLV = (1/cuota_apertura - 1/cuota_cierre) / (1/cuota_cierre)`
   - Si CLV > 0 consistentemente → edge real confirmado

2. **Simulación Monte Carlo de bankroll**:
   - 10,000 simulaciones con los resultados históricos
   - Calcular: prob_ruina, max_drawdown, Sharpe ratio del bankroll
   - Definir bankroll mínimo viable

3. **Eliminar vig antes de calcular edge**:
   ```python
   # Actual (incorrecto):
   prob_mercado = 1 / cuota  # incluye ~5% vig

   # Correcto:
   total_implied = 1/cuota_H + 1/cuota_D + 1/cuota_A
   prob_fair = (1/cuota) / total_implied  # sin vig
   ```

4. **Calcular intervalo de confianza del ROI** via bootstrap (1000 resamples)

### Experimentos sugeridos

- **Exp-4**: Bootstrap del ROI en walk-forward → IC 95% por temporada
- **Exp-5**: Monte Carlo con 10K simulaciones → probabilidad de ruina con bankroll de 2000EUR
- **Exp-6**: Comparar ROI con y sin eliminación de vig → cuantificar inflación

### Prioridad: **ALTA**

---

## 3 - Sistema de Apuestas

### Diagnóstico

El sistema de 3 capas (shrinkage conservador → filtros de calidad → Kelly fraccional) es conceptualmente sólido, pero los parámetros son arbitrarios y hay doble penalización de probabilidades.

### Problemas detectados

| # | Problema | Severidad | Evidencia |
|---|----------|-----------|-----------|
| 1 | **Doble shrinkage** | ALTA | Capa 1 reduce prob 40% hacia 1/3, luego Kelly reduce stake 75% → penalización excesiva |
| 2 | **Factor conservador arbitrario (0.60)** | ALTA | No optimizado empíricamente; debería calibrarse via Brier Score |
| 3 | **Sin eliminación de vig en edge** | ALTA | `prob_mercado = 1/cuota` incluye margen de la casa |
| 4 | **Kelly sin ajuste simultáneo sistemático** | MEDIA | Implementado en código pero no aplicado consistentemente |
| 5 | **Stake máximo 2.5% puede ser excesivo** | MEDIA | Sin análisis de riesgo de ruina para justificar el límite |
| 6 | **Umbral edge=5% no validado** | MEDIA | Reducido de 10% post-auditoría pero sin análisis de sensibilidad |

### Mejoras recomendadas

1. **Optimizar factor conservador empíricamente**:
   - Grid search de FACTOR_CONSERVADOR en [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
   - Métrica: Brier Score en walk-forward
   - Seleccionar el factor que minimice Brier Score

2. **Implementar eliminación de vig** en `core/sistema_expected_value.py`:
   ```python
   def cuota_fair(cuota_H, cuota_D, cuota_A):
       total = 1/cuota_H + 1/cuota_D + 1/cuota_A
       return {
           'H': total / (1/cuota_H),  # cuota_fair = cuota * (1/overround)
           'D': total / (1/cuota_D),
           'A': total / (1/cuota_A),
       }
   ```

3. **Análisis de sensibilidad del umbral de edge**:
   - Simular ROI con edge_min en [0.01, 0.03, 0.05, 0.07, 0.10, 0.15]
   - Reportar: ROI, #apuestas, drawdown, Sharpe por cada umbral
   - Seleccionar umbral que maximice Sharpe (no solo ROI)

4. **Reducir doble shrinkage**:
   - Opción A: Eliminar capa 1 y confiar solo en Kelly fraccional
   - Opción B: Calibrar modelo con Platt/Temperature y eliminar shrinkage manual
   - Opción C: Reducir Kelly fraction a 0.15 pero usar prob_modelo directa

5. **Implementar Kelly simultáneo consistentemente**:
   ```python
   f_simultaneous = f_kelly / sqrt(N_apuestas_jornada)
   ```

### Experimentos sugeridos

- **Exp-7**: Grid search de FACTOR_CONSERVADOR con Brier Score en walk-forward
- **Exp-8**: Análisis de sensibilidad edge_min → ROI, Sharpe, #bets
- **Exp-9**: Comparar: (a) shrinkage+Kelly25% vs (b) calibrado+Kelly15% vs (c) raw+Kelly10%
- **Exp-10**: Simulación Monte Carlo de bankroll con Kelly simultáneo vs independiente

### Prioridad: **ALTA**

---

## 4 - Ingeniería de Features

### Diagnóstico

El sistema genera 98 features y selecciona 35 automáticamente. La selección está dominada por features de mercado. Las features independientes tienen potencial pero están infrautilizadas.

### Problemas detectados

| # | Problema | Severidad | Evidencia |
|---|----------|-----------|-----------|
| 1 | **Features de mercado dominan selección** | CRITICA | 52% de importancia total en señales de mercado |
| 2 | **Ventana rolling fija (5 partidos)** | MEDIA | No captura tendencias a mediano plazo |
| 3 | **Sin decay exponencial** | MEDIA | Partidos recientes pesan igual que los de hace 5 jornadas |
| 4 | **Asian Handicap con 61% cobertura** | MEDIA | Imputación por mediana puede distorsionar señal |
| 5 | **Sin features de congestión de calendario** | BAJA | games_in_15d existe pero no se usa diferencia entre equipos |
| 6 | **Sin diferencias xG entre equipos** | BAJA | xG individual existe pero no xG_diff del matchup |

### Mejoras recomendadas

1. **Implementar rolling con decay exponencial**:
   ```python
   def ewm_rolling(series, span=5):
       return series.shift(1).ewm(span=span, adjust=False).mean()
   ```
   - Aplicar a: goles, shots, xG, puntos
   - Ventaja: partidos recientes pesan más sin perder contexto

2. **Agregar features multi-escala**:
   - Window=3 (forma inmediata)
   - Window=5 (forma reciente - actual)
   - Window=10 (tendencia mediana)
   - Calcular ratio: forma_corta / forma_larga (momentum)

3. **Nuevas features independientes propuestas**:

   | Feature | Fórmula | Rationale |
   |---------|---------|-----------|
   | `xG_Diff` | xG_home_rolling - xG_away_rolling | Diferencia de calidad ofensiva esperada |
   | `Elo_Diff` | Elo_home - Elo_away | Ya existe, verificar que se usa |
   | `Form_Momentum` | forma_3 / forma_10 | Detecta equipos en racha ascendente/descendente |
   | `Rest_Diff` | rest_home - rest_away | Ventaja de descanso relativa |
   | `Calendar_Congestion_Diff` | games_15d_home - games_15d_away | Fatiga relativa |
   | `Defensive_Rating` | xGA_rolling / league_avg_xGA | Rating defensivo normalizado |
   | `Offensive_Rating` | xGF_rolling / league_avg_xGF | Rating ofensivo normalizado |
   | `SoR` (Strength of Recent Schedule) | avg(Elo_oponentes_últimos_5) | Fuerza de rivales recientes |

4. **Separar features en grupos para análisis**:
   - Grupo A: Solo mercado (cuotas)
   - Grupo B: Solo independientes (xG, Elo, forma)
   - Grupo C: Combinado (actual)
   - Comparar rendimiento de cada grupo

### Experimentos sugeridos

- **Exp-11**: Comparar rolling fijo vs EWM (span=5) en Brier Score
- **Exp-12**: Entrenar modelo solo con Grupo B (independientes) → baseline de edge real
- **Exp-13**: Agregar SoR + Form_Momentum + xG_Diff → medir mejora incremental
- **Exp-14**: Análisis de correlación entre features de mercado e independientes

### Prioridad: **MEDIA**

---

## 5 - Validación del Modelo

### Diagnóstico

El walk-forward por temporada está implementado (6 folds, 5/6 positivos), pero falta profundidad en la validación. No hay detección de drift ni tests de significancia estadística.

### Problemas detectados

| # | Problema | Severidad | Evidencia |
|---|----------|-----------|-----------|
| 1 | **Sin test de significancia estadística** | ALTA | ROI +17.54% podría ser azar (no hay p-value) |
| 2 | **Sin detección de concept drift** | ALTA | 2022-23 tiene ROI -16.30% sin explicación |
| 3 | **Walk-forward solo por temporada** | MEDIA | Granularidad gruesa; no detecta drift intra-season |
| 4 | **Sin validación cruzada temporal con gap** | MEDIA | Posible leakage entre train/test por proximidad temporal |

### Mejoras recomendadas

1. **Test de permutación para ROI**:
   - Permutar labels 10,000 veces
   - Calcular ROI en cada permutación
   - p-value = proporción de ROI_permutado >= ROI_real
   - Si p < 0.05 → edge estadísticamente significativo

2. **Detección de concept drift**:
   ```python
   # PSI (Population Stability Index) entre temporadas
   def psi(expected, actual, bins=10):
       e_pct = np.histogram(expected, bins=bins)[0] / len(expected)
       a_pct = np.histogram(actual, bins=bins)[0] / len(actual)
       return np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
   # PSI > 0.25 → drift significativo
   ```

3. **Walk-forward rolling con gap**:
   - Train: partidos hasta jornada N-2
   - Gap: jornada N-1 (evita leakage por proximidad)
   - Test: jornada N
   - Reportar métricas acumuladas

4. **SHAP stability analysis**:
   - Calcular SHAP values por temporada
   - Correlacionar rankings de importancia entre temporadas
   - Si correlación < 0.7 → modelo inestable

### Experimentos sugeridos

- **Exp-15**: Test de permutación (10K) → p-value del ROI
- **Exp-16**: PSI entre temporadas para features top-10
- **Exp-17**: SHAP por temporada → correlación de rankings
- **Exp-18**: Walk-forward mensual (no por temporada) con gap de 1 jornada

### Prioridad: **ALTA**

---

## 6 - Optuna (Optimización de Hiperparámetros)

### Diagnóstico

150 trials con TPESampler es un punto de partida razonable, pero el espacio de búsqueda tiene inconsistencias y la métrica de optimización (F1-weighted) no está alineada con el objetivo real (ROI/value betting).

### Problemas detectados

| # | Problema | Severidad | Evidencia |
|---|----------|-----------|-----------|
| 1 | **Métrica de optimización desalineada** | ALTA | Optimiza F1-weighted pero selecciona modelo por ROI |
| 2 | **min_child_weight fuera de rango** | MEDIA | Rango Optuna [1,8] pero modelo final usa 20 (valor manual) |
| 3 | **Sin pruning** | MEDIA | 150 trials sin MedianPruner desperdicia compute |
| 4 | **150 trials posiblemente insuficiente** | MEDIA | XGBoost tiene ~10 hiperparámetros → espacio amplio |
| 5 | **Sin análisis de sobreoptimización** | MEDIA | No se compara rendimiento in-sample vs out-of-sample de Optuna |

### Mejoras recomendadas

1. **Alinear métrica de Optuna con objetivo**:
   - Opción A: Optimizar directamente ROI simulado (pero ruidoso con pocos datos)
   - Opción B: Optimizar Brier Score (mejor proxy para calibración)
   - Opción C: Métrica compuesta: `0.5 * (1-BrierScore) + 0.5 * ROI_normalizado`

2. **Ampliar y corregir espacio de búsqueda**:
   ```python
   # Corregir min_child_weight
   'min_child_weight': trial.suggest_int('min_child_weight', 5, 30),

   # Agregar gamma al espacio
   'gamma': trial.suggest_float('gamma', 0.0, 2.0),

   # Agregar scale_pos_weight
   'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 3.0),
   ```

3. **Implementar MedianPruner**:
   ```python
   study = optuna.create_study(
       direction='minimize',
       pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=3)
   )
   ```

4. **Aumentar trials a 500 con pruning** (equivalente a ~300 trials completos, más eficiente)

5. **Análisis de sobreoptimización**:
   - Comparar Optuna best score (in-sample) vs test score (out-of-sample)
   - Si gap > 10% → sobreoptimización
   - Solución: aumentar folds en TimeSeriesSplit de 3 a 5

### Experimentos sugeridos

- **Exp-19**: Optimizar Brier Score vs F1-weighted → comparar ROI final
- **Exp-20**: 500 trials con MedianPruner vs 150 sin pruner → eficiencia
- **Exp-21**: TimeSeriesSplit con 5 folds vs 3 folds → estabilidad
- **Exp-22**: Análisis gap in-sample/out-of-sample de los mejores 10 trials

### Prioridad: **MEDIA**

---

## Roadmap de Implementación

### Fase 1: Correcciones Críticas (Semana 1-2)

| # | Tarea | Archivo(s) | Prioridad |
|---|-------|------------|-----------|
| 1.1 | Corregir bug de calibración (guardar modelo calibrado) | `pipeline/02_entrenar_modelo.py` | CRITICA |
| 1.2 | Implementar eliminación de vig en cálculo de edge | `core/sistema_expected_value.py`, `pipeline/02_entrenar_modelo.py` | CRITICA |
| 1.3 | Eliminar features con train/serve mismatch (Prob_Move) | `config.py`, `utils.py` | CRITICA |
| 1.4 | Implementar CLV tracker funcional | `core/clv_tracker.py` | CRITICA |

### Fase 2: Validación Robusta (Semana 3-4)

| # | Tarea | Archivo(s) | Prioridad |
|---|-------|------------|-----------|
| 2.1 | Test de permutación para significancia del ROI | `pipeline/05_validacion_estadistica.py` (nuevo) | ALTA |
| 2.2 | Bootstrap IC 95% del ROI | `pipeline/05_validacion_estadistica.py` | ALTA |
| 2.3 | Simulación Monte Carlo de bankroll (10K runs) | `core/simulacion_montecarlo.py` (nuevo) | ALTA |
| 2.4 | Detección de concept drift (PSI) | `pipeline/06_concept_drift.py` (nuevo) | ALTA |

### Fase 3: Mejora del Sistema de Apuestas (Semana 5-6)

| # | Tarea | Archivo(s) | Prioridad |
|---|-------|------------|-----------|
| 3.1 | Grid search de FACTOR_CONSERVADOR | `herramientas/optimizar_shrinkage.py` (nuevo) | ALTA |
| 3.2 | Análisis de sensibilidad edge_min | `herramientas/sensibilidad_edge.py` (nuevo) | ALTA |
| 3.3 | Resolver doble shrinkage (prob + Kelly) | `core/sistema_expected_value.py` | ALTA |
| 3.4 | Kelly simultáneo sistemático | `core/sistema_expected_value.py` | MEDIA |

### Fase 4: Feature Engineering Avanzado (Semana 7-8)

| # | Tarea | Archivo(s) | Prioridad |
|---|-------|------------|-----------|
| 4.1 | Implementar EWM rolling (decay exponencial) | `utils.py` | MEDIA |
| 4.2 | Agregar features multi-escala (3, 5, 10) | `utils.py`, `config.py` | MEDIA |
| 4.3 | Nuevas features: xG_Diff, Form_Momentum, SoR | `utils.py` | MEDIA |
| 4.4 | Entrenar modelo solo con features independientes (Exp-12) | `pipeline/03_entrenar_sin_cuotas.py` | MEDIA |

### Fase 5: Optimización Avanzada (Semana 9-10)

| # | Tarea | Archivo(s) | Prioridad |
|---|-------|------------|-----------|
| 5.1 | Cambiar Optuna a optimizar Brier Score | `herramientas/buscar_pesos_xgb.py` | MEDIA |
| 5.2 | Implementar MedianPruner + 500 trials | `herramientas/buscar_pesos_xgb.py` | MEDIA |
| 5.3 | SHAP stability analysis por temporada | `pipeline/07_shap_analysis.py` (nuevo) | MEDIA |
| 5.4 | Walk-forward mensual con gap | `pipeline/02_entrenar_modelo.py` | MEDIA |

---

## Métricas de Éxito

Al completar este plan, el sistema debería cumplir:

| Métrica | Objetivo | Estado Actual |
|---------|----------|---------------|
| CLV positivo consistente | CLV > 0 en 60%+ de apuestas | No medido |
| p-value del ROI | p < 0.05 | No calculado |
| Brier Score | < 0.195 | 0.1952 (baseline) |
| ROI walk-forward | > 0% en 5/6 temporadas | 5/6 (mantener) |
| Prob. ruina (Monte Carlo) | < 5% con bankroll 2000EUR | No calculado |
| Max drawdown | < 40% del bankroll | No medido |
| Importancia features independientes | > 40% del total | ~25% (actual) |
| Gap Optuna in/out sample | < 10% | No medido |

---

## Conclusión

El proyecto tiene una base técnica sólida con un pipeline bien estructurado. Los problemas principales son:

1. **El modelo replica al mercado en vez de superarlo** - la dependencia de cuotas es el riesgo fundamental
2. **El ROI reportado no está validado estadísticamente** - falta CLV, permutación, y Monte Carlo
3. **Los parámetros del sistema de apuestas son arbitrarios** - necesitan optimización empírica

La implementación de las Fases 1-2 (correcciones críticas + validación robusta) es **imprescindible** antes de usar el sistema con dinero real. Las Fases 3-5 son mejoras incrementales que aumentarán la robustez y el edge potencial del sistema.
