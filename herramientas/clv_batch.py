"""
Genera CLV tracking en batch usando el CSV local.

Usa el modelo entrenado y las features guardadas para producir
probabilidades historicas y registrar CLV con cuotas de apertura
y cierre (B365H/D/A y B365CH/CD/CA).
"""

import os
import warnings

import pandas as pd
import joblib

from config import ARCHIVO_FEATURES, ARCHIVO_MODELO, ARCHIVO_FEATURES_PKL
from core.clv_tracker import CLVTracker
from utils import (
    agregar_features_tabla,
    agregar_features_cuotas_derivadas,
    agregar_features_asian_handicap,
    agregar_features_rolling_extra,
    agregar_features_multi_escala,
    agregar_features_ewm,
    agregar_features_forma_momentum,
    agregar_features_pinnacle_move,
    agregar_features_arbitro,
    agregar_features_elo,
    agregar_features_sor,
)


def _preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el mismo feature engineering usado en entrenamiento."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = agregar_features_tabla(df)
        df = agregar_features_cuotas_derivadas(df)
        df = agregar_features_asian_handicap(df)
        df = agregar_features_rolling_extra(df)
        df = agregar_features_multi_escala(df)
        df = agregar_features_ewm(df)
        df = agregar_features_forma_momentum(df)
        df = agregar_features_pinnacle_move(df)
        df = agregar_features_arbitro(df)
        df = agregar_features_elo(df)
        df = agregar_features_sor(df)
    return df


def main(archivo_csv: str = ARCHIVO_FEATURES) -> None:
    if not os.path.exists(archivo_csv):
        raise FileNotFoundError(f"No se encontro el CSV: {archivo_csv}")

    if not os.path.exists(ARCHIVO_MODELO):
        raise FileNotFoundError(f"No se encontro el modelo: {ARCHIVO_MODELO}")

    if not os.path.exists(ARCHIVO_FEATURES_PKL):
        raise FileNotFoundError(f"No se encontro features.pkl: {ARCHIVO_FEATURES_PKL}")

    print("=" * 70)
    print("CLV TRACKER BATCH")
    print("=" * 70)

    df = pd.read_csv(archivo_csv)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    df = _preparar_features(df)

    cierre_cols = ["B365CH", "B365CD", "B365CA"]
    if not all(c in df.columns for c in cierre_cols):
        raise ValueError("Faltan columnas de cierre B365CH/B365CD/B365CA en el CSV.")

    modelo = joblib.load(ARCHIVO_MODELO)
    features = joblib.load(ARCHIVO_FEATURES_PKL)

    faltantes = [f for f in features if f not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan features en el CSV: {faltantes}")

    X = df[features]

    try:
        probs = modelo.predict_proba(X)
    except Exception:
        probs = modelo.predict_proba(X.fillna(0))

    if probs.shape[1] != 3:
        raise ValueError("Se esperaban 3 probabilidades (Local/Empate/Visitante).")

    df["prob_0"] = probs[:, 0]
    df["prob_1"] = probs[:, 1]
    df["prob_2"] = probs[:, 2]

    tracker = CLVTracker()
    tracker.registrar_batch_historico(df)
    tracker.guardar()
    tracker.resumen()


if __name__ == "__main__":
    main()
