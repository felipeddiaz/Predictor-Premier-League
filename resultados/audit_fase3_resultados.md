# Resultados Fase 3

Fecha: 2026-03-14
Rama: claude/audit-improvement-plan-J5D5V

## 3.1 Grid search FACTOR_CONSERVADOR
- Script: `herramientas/optimizar_shrinkage.py`
- Resultados: `resultados/shrinkage_grid.md`, `resultados/shrinkage_grid.csv`
- Mejor alpha (Brier mean season): 1.00 (no shrinkage)
- Nota: evalua con el modelo entrenado actual (no re-entrena por temporada)
- Actualizacion aplicada: `config.py` FACTOR_CONSERVADOR = 1.00

## 3.2 Sensibilidad edge_min
- Script: `herramientas/sensibilidad_edge.py`
- Resultados: `resultados/sensibilidad_edge.md`, `resultados/sensibilidad_edge.csv`
- Max Sharpe observado: edge_min = 0.10 (Sharpe 1.807)
- Trade-off: edge_min alto reduce N de apuestas
- Actualizacion aplicada: `config.py` UMBRAL_EDGE_MINIMO = 0.10

## 3.3 Resolver doble shrinkage
- Archivo: `core/sistema_expected_value.py`
- Cambios:
  - `analizar_apuesta` acepta `prob_modelo_raw` para usar prob sin shrinkage en Kelly (opcional)
  - `kelly_criterion` permite `max_kelly` configurable
  - Kelly simultaneo ahora ajusta `stake_recomendado`

## 3.4 Kelly simultaneo sistematico
- Archivo: `core/sistema_expected_value.py`
- Aplicacion: cuando hay varias apuestas recomendadas, los stakes se reducen por sqrt(N)
  y ahora se refleja en `stake_recomendado`

## Ejecucion posterior (predicciones y backtest rapido)
- Predicciones jornada: `jornada/predecir_jornada_completa.py`
  - Log: `resultados/predicciones_fase3.log`
  - Factor conservador usado: 1.00
  - Edge min usado: 10%
- Entrenamiento/backtest: `pipeline/02_entrenar_modelo.py`
  - Log: `resultados/entrenamiento_fase3.log`
  - ROI simulado (ganador): +21.55%
  - Walk-forward ROI promedio: +13.35%
