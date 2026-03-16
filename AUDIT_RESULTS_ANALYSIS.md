# Auditoría de Resultados Post-Implementación (Fases 1-4)

**Fecha de análisis:** 2026-03-14
**Datos prospectivos:** 271 partidos (2025-08-15 → 2026-02-23)
**Modelo evaluado:** XGBoost (35 features, seleccionado por ROI)

---

## Resumen Ejecutivo

Los cambios implementados en las fases 1-4 produjeron **mejoras estructurales significativas** en la arquitectura del sistema, pero los resultados prospectivos revelan que **el edge no es estadísticamente significativo** (p-value = 0.41). El ROI de value bets es +3.39% (vs +24.22% en backtest), confirmando la sospecha de sobreajuste al dataset histórico.

**Veredicto: El sistema mejoró en robustez y limpieza, pero NO ha demostrado edge real explotable.**

---

## 1. Cambios Implementados vs Plan Original

### Fase 1: Correcciones Críticas

| Tarea | Estado | Impacto |
|-------|--------|---------|
| Bug de calibración | CORREGIDO | Modelo guardado ahora evalúa calibración y decide si aplicarla (modelo actual: "Sin Calibrar" — calibración no mejoró Brier) |
| Eliminación closing odds | CORREGIDO | Pinnacle_Sharp_H/A reemplazado por Pinnacle_Open_H/A. Prob_Move ya no se usa como feature con closing data |
| Train/serve mismatch | PARCIAL | Prob_Move eliminado del top-35, pero config aún lista features obsoletas |
| CLV Tracker | IMPLEMENTADO | `core/clv_tracker.py` (273 líneas) — funcional pero sin datos de closing para validar |
| Eliminación de vig | NO IMPLEMENTADO | `sistema_expected_value.py` aún usa `1/cuota` sin normalizar |

### Fase 2: Validación Robusta

| Tarea | Estado | Impacto |
|-------|--------|---------|
| Walk-forward season-by-season | IMPLEMENTADO | 6 temporadas validadas, 5/6 positivas |
| Registro prospectivo | IMPLEMENTADO | 271 partidos tracked en CSV |
| Test permutación / Bootstrap | NO IMPLEMENTADO | Lo ejecuté en esta auditoría (ver resultados abajo) |
| Monte Carlo bankroll | NO IMPLEMENTADO | Simulación básica ejecutada aquí |
| Detección concept drift (PSI) | NO IMPLEMENTADO | — |

### Fase 3: Sistema de Apuestas

| Tarea | Estado | Impacto |
|-------|--------|---------|
| Grid search FACTOR_CONSERVADOR | NO IMPLEMENTADO | Sigue en 0.60 arbitrario |
| Sensibilidad edge_min | NO IMPLEMENTADO | Sigue en 0.05 |
| Resolver doble shrinkage | NO IMPLEMENTADO | Metadata muestra `alpha_shrinkage: 1.0` (desactivado) — MEJORA parcial |
| Kelly simultáneo | IMPLEMENTADO | En código pero impacto no medido |

### Fase 4: Feature Engineering

| Tarea | Estado | Impacto |
|-------|--------|---------|
| Elo Ratings (4 features) | IMPLEMENTADO | 4/4 seleccionadas en top-35 (posiciones 12, 18, 20, 22) |
| EWM rolling (decay exponencial) | NO IMPLEMENTADO | — |
| Multi-escala (3, 5, 10) | PARCIAL | Existe `agregar_features_multi_escala()` con window=10 |
| Nuevas features (xG_Diff, xG_Global) | IMPLEMENTADO | xG_Diff, xG_Global_Diff, AT_xGA_Global, HT_xG_Global en top-35 |
| SoR / Form_Momentum | NO IMPLEMENTADO | — |

**Progreso general estimado: ~40% del plan completado**

---

## 2. Composición del Modelo Actual (35 Features)

| Categoría | Count | % | Cambio vs Pre-Auditoría |
|-----------|-------|---|------------------------|
| Mercado (cuotas) | 14 | 40.0% | **MEJORADO** (era 52%) |
| Independientes (deportivas) | 17 | 48.6% | **MEJORADO** (era ~25%) |
| Otras (descanso, xG global) | 4 | 11.4% | Nuevas |

**Mejora clave**: La proporción mercado/independiente pasó de **52%/25% → 40%/49%**. El modelo ahora tiene mayor peso en features deportivas reales.

### Features de mercado eliminadas (closing → opening):
- Pinnacle_Sharp_H/A → Pinnacle_Open_H/A
- Prob_Move_H/D/A → Eliminados del top-35
- B365CH/CD/CA → Eliminados completamente
- PS_vs_Avg_H → Eliminado (usaba closing)

### Features nuevas incorporadas:
- Elo_Diff, Elo_WinProb_H, HT_Elo, AT_Elo (Elo system)
- xG_Global_Diff, AT_xGA_Global, HT_xG_Global (xG multi-venue)
- HT_Had_Europa (fatiga europea)

---

## 3. Resultados Prospectivos 2025-26 (271 partidos)

### 3.1 Accuracy

| Métrica | Valor | Benchmark |
|---------|-------|-----------|
| Accuracy modelo | 49.08% | Random 33.3% |
| Accuracy mercado (favorito) | 50.18% | — |
| **Diferencia** | **-1.11pp** | Modelo pierde vs mercado |

| Predicción | Accuracy | N |
|------------|----------|---|
| Home | 55.91% | 127 |
| Away | 45.90% | 122 |
| Draw | 27.27% | 22 |

### 3.2 Calibración

| Métrica | Prospectivo | Backtest | Delta |
|---------|-------------|----------|-------|
| Brier Score | **0.2042** | 0.1952 | +0.0090 (peor) |
| Log Loss | **1.0214** | 0.9829 | +0.0385 (peor) |

**Calibración por bins de probabilidad:**

| Bin | Prob Media | Freq Real | N | Delta |
|-----|-----------|-----------|---|-------|
| [0.00-0.20) | 0.1714 | 0.0893 | 56 | -0.0821 (sobreestima) |
| [0.20-0.30) | 0.2592 | 0.2606 | 307 | +0.0013 (excelente) |
| [0.30-0.40) | 0.3414 | 0.3370 | 276 | -0.0044 (excelente) |
| [0.40-0.50) | 0.4500 | 0.4301 | 93 | -0.0199 (bueno) |
| [0.50-0.60) | 0.5455 | 0.6061 | 66 | +0.0605 (subestima) |
| [0.60-0.80) | 0.6493 | 0.8667 | 15 | +0.2173 (subestima mucho) |

**Diagnóstico**: Calibración excelente en el rango [0.20-0.50] donde cae la mayoría de predicciones. Subestima probabilidades altas (>0.50) → el shrinkage conservador es **demasiado agresivo** para eventos probables.

### 3.3 Value Bets

| Métrica | Valor |
|---------|-------|
| Value bets identificadas | 174/271 (64.2%) |
| ROI value bets | **+3.39%** |
| Win rate | 25.9% (45/174) |
| P&L neto | +5.90 unidades |
| Edge promedio | 8.25% |

| Tipo | N Bets | Win Rate | ROI |
|------|--------|----------|-----|
| **Draw** | 97 | 28.9% | **+13.03%** |
| Away | 69 | 23.2% | -3.54% |
| Home | 8 | 12.5% | -53.75% |

**Hallazgo clave**: Todo el valor viene de **draws**. El modelo identifica draws infravalorados por el mercado con ROI +13.03%.

| Rango Edge | N Bets | ROI |
|------------|--------|-----|
| 5%-8% | 95 | **-9.66%** |
| 8%-10% | 50 | +3.50% |
| 10%-15% | 22 | **+35.45%** |
| 15%+ | 7 | **+79.00%** |

**Hallazgo**: Edges bajos (5-8%) no son rentables. Solo edges >10% producen ROI positivo significativo. Esto sugiere que el **UMBRAL_EDGE_MINIMO debería subir a 0.10**, no bajar a 0.05.

### 3.4 Evolución Temporal

| Mes | Bets | Wins | Win% | ROI | P&L |
|-----|------|------|------|-----|-----|
| 2025-08 | 19 | 4 | 21.1% | -21.05% | -4.00 |
| 2025-09 | 21 | 8 | 38.1% | **+37.86%** | +7.95 |
| 2025-10 | 20 | 5 | 25.0% | +18.90% | +3.78 |
| 2025-11 | 19 | 4 | 21.1% | -36.42% | -6.92 |
| 2025-12 | 30 | 8 | 26.7% | +9.87% | +2.96 |
| 2026-01 | 32 | 8 | 25.0% | +5.16% | +1.65 |
| 2026-02 | 33 | 8 | 24.2% | +1.45% | +0.48 |

**Patrón**: Alta varianza mensual. 5/7 meses positivos, 2 meses con drawdowns significativos.

### 3.5 Drawdown y Bankroll

| Métrica | Valor |
|---------|-------|
| P&L final | +5.90 unidades |
| P&L máximo alcanzado | +11.73 unidades |
| Max Drawdown | 15.96 unidades |
| Max DD % del apostado | 9.2% |

**Simulación con bankroll real (2000 EUR, flat 20 EUR):**

| Métrica | Valor |
|---------|-------|
| Bankroll final | 2,118 EUR |
| Beneficio | +118 EUR |
| Min bankroll alcanzado | 1,856 EUR |
| Max Drawdown EUR | 144 EUR (7.2%) |

---

## 4. Significancia Estadística

### Bootstrap (10,000 resamples)

| Métrica | Valor |
|---------|-------|
| ROI observado | +3.39% |
| IC 95% Bootstrap | **[-23.15%, +30.83%]** |
| P-value (ROI ≤ 0) | **0.4089** |
| Significativo al 5% | **NO** |
| Significativo al 10% | **NO** |

**Conclusión estadística**: El ROI positivo de +3.39% **no es estadísticamente distinguible de cero**. El intervalo de confianza incluye ampliamente valores negativos. Se necesitan más datos (>500 bets) o un edge más fuerte para alcanzar significancia.

---

## 5. Comparación Backtest vs Prospectivo

| Métrica | Backtest (Train) | Walk-Forward (Avg) | Prospectivo (OOS) |
|---------|------------------|--------------------|-------------------|
| ROI | +24.22% | +17.54% | **+3.39%** |
| Log Loss | 0.9829 | 0.9843 | **1.0214** |
| Brier Score | 0.1952 | — | **0.2042** |
| Accuracy | 52.37% | — | **49.08%** |

**Degradación clara**: ROI cae de +24% → +17% → +3% conforme pasamos de in-sample a out-of-sample real. Esto es consistente con **sobreajuste moderado al dataset histórico**.

---

## 6. Comparación de Experimentos Optuna

| Configuración | F1 CV | F1 Test | N Features | LR | Depth |
|--------------|-------|---------|------------|-----|-------|
| xgb_nuevas_feats (producción) | 0.5239 | 0.4865 | 45 | 0.00566 | 6 |
| best_params (latest) | — | 0.4912 | — | 0.01625 | 7 |
| xgb_25feat | 0.5371 | 0.4943 | 25 | 0.02723 | 10 |
| rf_25feat | 0.5306 | 0.5030 | 25 | — | — |
| xgb_focused | 0.5114 | 0.4629 | 27 | 0.01087 | 6 |

**Observación**: El modelo con 25 features (RF) tiene **mejor F1 test** (0.5030) que el XGBoost actual (0.4865). La reducción agresiva de features podría mejorar generalización.

---

## 7. Diagnóstico General

### Lo que MEJORÓ con las fases 1-4:

1. **Eliminación de leakage implícito**: Closing odds removidas como features
2. **Mejor balance mercado/independiente**: 52%→40% mercado, 25%→49% independiente
3. **Elo Ratings**: 4 features de alta calidad predictiva incorporadas
4. **xG expandido**: Features globales + venue-specific
5. **Bug de calibración identificado**: Modelo reporta correctamente si calibrado o no
6. **Walk-forward implementado**: Validación temporal robusta
7. **Registro prospectivo**: Tracking en vivo de 271 partidos
8. **CLV tracker**: Infraestructura lista (falta datos de closing)
9. **Descanso/Europa**: Feature HT_Had_Europa en top-35

### Lo que NO mejoró o empeoró:

1. **ROI real es ~3% (no significativo)**: El edge del backtest no se traduce a producción
2. **Calibración degradada OOS**: Brier Score +0.009, Log Loss +0.039
3. **Sin eliminación de vig**: Edge sigue calculándose con probabilidades que incluyen margen
4. **Sin optimización de shrinkage**: Factor 0.60 arbitrario (aunque alpha=1.0 en metadata sugiere que se desactivó para entrenamiento)
5. **Umbral edge demasiado bajo**: 5% no es rentable; datos sugieren que 10%+ sí lo es
6. **Sin Monte Carlo formal**: No hay análisis de riesgo de ruina
7. **Sin SHAP**: No se puede diagnosticar concept drift

---

## 8. Recomendaciones Actualizadas

### PRIORIDAD CRITICA (hacer antes de apostar dinero real)

1. **Subir UMBRAL_EDGE_MINIMO a 0.10**: Los datos prospectivos muestran que edge <8% no es rentable (-9.66% ROI). Solo edge >=10% produce ROI positivo (+35.45%)

2. **Implementar eliminación de vig**: Todos los cálculos de edge están inflados ~5%. Esto es una corrección matemática simple que afecta todas las decisiones

3. **Enfocarse en draws**: El 100% del valor viene de apuestas draw. Considerar un modelo especializado en detección de draws infravalorados

4. **Acumular más datos OOS**: 174 bets no son suficientes para significancia estadística. Se necesitan ~500+ bets para confirmar edge (si existe)

### PRIORIDAD ALTA

5. **Reducir número de features a 25**: Los datos Optuna sugieren que 25 features generalizan mejor que 35-45

6. **Implementar Monte Carlo**: Con los 174 resultados reales, simular 10K trayectorias de bankroll

7. **Calibrar shrinkage empíricamente**: El modelo subestima probabilidades altas (>0.50). El shrinkage actual es demasiado agresivo para favoritos

8. **Obtener datos de closing odds**: Sin CLV real no se puede confirmar si el edge es estructural

### PRIORIDAD MEDIA

9. **EWM rolling**: Reemplazar ventana fija por decay exponencial
10. **Modelo solo-draws**: Entrenar clasificador binario (Draw vs No-Draw) para maximizar la fortaleza detectada
11. **SHAP por temporada**: Diagnosticar si la importancia de features es estable

---

## 9. Conclusión Final

El sistema ha mejorado significativamente en **arquitectura y limpieza** tras las fases 1-4:
- Eliminó leakage implícito de closing odds
- Incorporó features independientes de calidad (Elo, xG global)
- Implementó validación temporal robusta

Sin embargo, los **resultados prospectivos son sobrios**:
- ROI +3.39% no es estadísticamente significativo (p=0.41)
- Degradación clara backtest → producción (+24% → +3%)
- El edge real, si existe, probablemente está en **draws con edge alto (>10%)**

El proyecto es un excelente sistema de análisis y research, pero **no está listo para apuestas con dinero real** hasta que:
1. Se acumulen suficientes datos para significancia estadística
2. Se confirme CLV positivo con datos de closing
3. Se optimicen los parámetros de apuesta con datos reales (no backtest)
