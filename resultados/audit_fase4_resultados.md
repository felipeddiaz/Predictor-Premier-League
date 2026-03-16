# Resultados Fase 4

Fecha: 2026-03-14
Rama: claude/audit-improvement-plan-J5D5V

## Cambios implementados
- EWM rolling (span=5) para pts, goles, shots target, xG/xGA
- Multi-escala extendida (window 3 y 10) + Form_Momentum
- SoR (Strength of Recent Schedule) con Elo rivales (rolling 5)
- Calendar_Congestion_Diff (games_15d home - away)

## Entrenamiento modelo principal (con cuotas)
- Log: `resultados/entrenamiento_fase4.log`
- Ganador: Random Forest Balanceado
- Log Loss: 0.9805
- Brier Score: 0.1948
- ROI simulado: +29.82%
- Walk-forward ROI promedio: +11.88%
- Features usadas: 35 (de 121 originales)

## Entrenamiento modelo estructural (sin cuotas)
- Log: `resultados/entrenamiento_sin_cuotas_fase4.log`
- Log Loss: 0.9927
- Brier Score: 0.1976
- ROI simulado: +29.20%
- Recomendacion interna del script: edge >5% como filtro minimo

## Notas
- El shrinkage optimo sigue siendo alpha=1.00 en ambos modelos.
- El modelo estructural mejora ROI vs versiones previas, pero mantiene Brier mayor que el modelo con cuotas.
