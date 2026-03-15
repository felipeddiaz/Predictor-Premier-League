# Resultados Fase 2

Fecha: 2026-03-14
Rama: claude/audit-improvement-plan-J5D5V

## Validacion estadistica (ROI)
- Script: `pipeline/05_validacion_estadistica.py`
- Parametros: `--n_perm 2000 --n_boot 500 --edge_min 0.10`
- Resultados (ver `resultados/validacion_estadistica.md`):
  - ROI real: +0.2982
  - p-value: 0.039
  - ROI perm mean: +0.0741 (std 0.1234)
  - Bootstrap CI 95%: [+0.0307, +0.5649]

## Simulacion Monte Carlo
- Script: `core/simulacion_montecarlo.py`
- Parametros: `--n_sim 5000 --edge_min 0.10`
- Resultados (ver `resultados/simulacion_montecarlo.md`):
  - Prob. ruina: 0.00%
  - Bankroll final mean: 7604.89
  - Max drawdown mean: 31.19% (p95: 43.62%)

## Concept Drift (PSI)
- Script: `pipeline/06_concept_drift.py`
- Base season: 2016-17
- Top features analizadas: Pinnacle_Open_A, Pinnacle_Open_H, Home_Advantage_Prob, HT_Elo, HT_GoalsFor_EWM5, AH_Edge_Home, AT_xGA_EWM5, AT_Elo, Elo_Diff, xG_Diff
- Resumen (ver `resultados/concept_drift_psi.md` y `resultados/concept_drift_psi.csv`):
  - PSI medio por temporada: 0.240 a 0.516
  - PSI max por temporada: 0.631 a 2.688
  - Drift alto en varias temporadas (PSI > 0.25 en varias features)
