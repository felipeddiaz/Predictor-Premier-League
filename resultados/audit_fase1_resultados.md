# Resultados Fase 1 (ejecuciones)

Fecha: 2026-03-14
Rama: claude/audit-improvement-plan-J5D5V

## Resumen
- Se ejecutaron: entrenamiento (`pipeline/02_entrenar_modelo.py`) y CLV batch (`herramientas/clv_batch.py`).
- El ROI simulado ahora usa probabilidades sin vig (fair) como parte de la modificación en `_roi_simulado`.
- Se generó un baseline en el mismo entorno con la versión anterior (sin vig removal) para comparar.

## Entrenamiento
### Ejecución actual
- Comando: `PYTHONIOENCODING=utf-8 PYTHONPATH=. python pipeline/02_entrenar_modelo.py`
- Estado: exitoso
- Log: `resultados/entrenamiento_fase1.log`

### Resultados principales (con vig removal en ROI)
- Ganador: Random Forest Balanceado
- Log Loss: 0.9845
- Brier Score: 0.1956
- ROI simulado: +21.55%
- Accuracy: 52.10%
- F1 (weighted): 0.4988

### Walk-forward promedio (season-by-season)
- Log Loss: 0.9876
- Brier Score: 0.1958
- ROI simulado: +13.35%

### Baseline (sin vig removal)
- Comando: `PYTHONIOENCODING=utf-8 PYTHONPATH=. python pipeline/02_entrenar_modelo.py`
- Log: `resultados/entrenamiento_baseline.log`
- Ganador: Random Forest Balanceado
- Log Loss: 0.9845
- Brier Score: 0.1956
- ROI simulado: +30.71%
- Walk-forward ROI promedio: -2.30%

### Comparativa (baseline vs actual)
- ROI simulado (ganador): +30.71% → +21.55% (delta -9.16 pp)
- Walk-forward ROI promedio: -2.30% → +13.35% (delta +15.65 pp)
- Log Loss/Brier: sin cambios relevantes (misma selección de modelo)

### Tabla comparativa
| Metrica | Baseline (sin vig) | Actual (vig removal) | Delta |
| --- | --- | --- | --- |
| Log Loss (ganador) | 0.9845 | 0.9845 | +0.0000 |
| Brier (ganador) | 0.1956 | 0.1956 | +0.0000 |
| ROI simulado (ganador) | +30.71% | +21.55% | -9.16 pp |
| Accuracy (ganador) | 52.10% | 52.10% | +0.00 pp |
| F1 weighted (ganador) | 0.4988 | 0.4988 | +0.0000 |
| Walk-forward ROI promedio | -2.30% | +13.35% | +15.65 pp |

## CLV Batch
### Ejecución actual
- Comando: `PYTHONIOENCODING=utf-8 PYTHONPATH=. python herramientas/clv_batch.py`
- Estado: exitoso
- Log: `resultados/clv_batch_fase1.log`

### Resumen CLV
- Total apuestas: 3691
- CLV promedio: -0.0001 (negativo)
- CLV positivo: 49.3%
- Edge vs apertura (avg): +0.0884
- Edge vs cierre (avg): +0.0885
- ROI histórico (simulado): +93.7%

## Dependencias
- Se instalaron manualmente: `joblib`, `scikit-learn` (con wheels).

## Nota sobre comparaciones
Para reportar cambios vs. antes (ROI, Brier, Log Loss, CLV promedio, %CLV positivo), se requiere completar los scripts con el entorno de dependencias activo y, si aplica, conservar un baseline previo.
