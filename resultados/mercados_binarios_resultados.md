# Resultados mercados binarios

Fecha: 2026-03-14

## Scripts nuevos
- `pipeline/04_entrenar_over_under.py`
- `pipeline/07_entrenar_tarjetas.py`
- `pipeline/08_entrenar_corners.py`
- `pipeline/mercados_binarios.py` (base compartida)

## Modelos guardados
- `modelos/modelo_over_under.pkl`
- `modelos/modelo_tarjetas.pkl`
- `modelos/modelo_corners.pkl`

## Resumen de entrenamiento

### Over/Under 2.5 goles
- Ganador: XGBoost
- Holdout: LogLoss 0.6916, Brier 0.2492, ROI +7.18% (usando `B365>2.5`/`B365<2.5`)
- Walk-forward promedio: LogLoss 0.7444, Brier 0.2703, F1 0.5980, Acc 51.84%, ROI +1.40%

### Tarjetas > 3.5
- Ganador: XGBoost
- Holdout: LogLoss 0.6913, Brier 0.2491, ROI N/A (sin cuotas de tarjetas en dataset actual)
- Walk-forward promedio: LogLoss 0.7628, Brier 0.2784, F1 0.5095, Acc 49.30%, ROI N/A

### Corners > 9.5
- Ganador: RF_Balanceado
- Holdout: LogLoss 0.6916, Brier 0.2493, ROI N/A (sin cuotas de corners en dataset actual)
- Walk-forward promedio: LogLoss 0.6941, Brier 0.2504, F1 0.6261, Acc 53.22%, ROI N/A

## Nota
Se usan cuotas solo para evaluacion de ROI. En este dataset solo hay cuotas O/U de goles; por eso tarjetas/corners reportan ROI N/A.
