"""
Grid search de FACTOR_CONSERVADOR usando Brier Score por temporada.

Nota: se usa el modelo entrenado actual para predecir y se evalua
la calibracion por temporada con shrinkage aplicado.
"""

import os
import warnings

import numpy as np
import pandas as pd
import joblib

from config import ARCHIVO_FEATURES, ARCHIVO_MODELO, ARCHIVO_FEATURES_PKL
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


def _asignar_temporada(dates: pd.Series) -> pd.Series:
    return dates.apply(
        lambda d: f"{d.year}-{str(d.year+1)[-2:]}" if d.month >= 8
        else f"{d.year-1}-{str(d.year)[-2:]}"
    )


def _brier_score_multiclase(y_true: np.ndarray, probs: np.ndarray) -> float:
    n = len(y_true)
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def _aplicar_shrinkage(probs: np.ndarray, alpha: float) -> np.ndarray:
    uniforme = np.array([1 / 3, 1 / 3, 1 / 3])
    ajustadas = alpha * probs + (1 - alpha) * uniforme
    return ajustadas / ajustadas.sum(axis=1, keepdims=True)


def _write_md_table(path: str, headers: list, rows: list):
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")


def main():
    if not os.path.exists(ARCHIVO_MODELO) or not os.path.exists(ARCHIVO_FEATURES_PKL):
        raise FileNotFoundError("Modelo o features no encontrados. Ejecuta entrenamiento primero.")

    df = pd.read_csv(ARCHIVO_FEATURES)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df = _preparar_features(df)

    modelo = joblib.load(ARCHIVO_MODELO)
    features = joblib.load(ARCHIVO_FEATURES_PKL)

    faltantes = [f for f in features if f not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan features en el CSV: {faltantes}")

    X = df[features]
    y = df["FTR_numeric"].values

    try:
        probs = modelo.predict_proba(X)
    except Exception:
        probs = modelo.predict_proba(X.fillna(0))

    df["Season"] = _asignar_temporada(df["Date"])
    temporadas = sorted(df["Season"].unique())

    alphas = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    resultados = []

    for alpha in alphas:
        probs_adj = _aplicar_shrinkage(probs, alpha)
        brier_global = _brier_score_multiclase(y, probs_adj)

        brier_por_temp = []
        for season in temporadas:
            idx = df["Season"] == season
            if idx.sum() == 0:
                continue
            brier_season = _brier_score_multiclase(y[idx], probs_adj[idx])
            brier_por_temp.append(brier_season)

        brier_mean = float(np.mean(brier_por_temp)) if brier_por_temp else brier_global

        resultados.append({
            "alpha": alpha,
            "brier_global": brier_global,
            "brier_mean_season": brier_mean,
        })

    resultados = sorted(resultados, key=lambda r: r["brier_mean_season"])
    mejor = resultados[0]

    os.makedirs("resultados", exist_ok=True)
    ruta_csv = os.path.join("resultados", "shrinkage_grid.csv")
    pd.DataFrame(resultados).to_csv(ruta_csv, index=False)

    ruta_md = os.path.join("resultados", "shrinkage_grid.md")
    rows = [
        [
            f"{r['alpha']:.2f}",
            f"{r['brier_global']:.4f}",
            f"{r['brier_mean_season']:.4f}",
        ]
        for r in resultados
    ]
    _write_md_table(ruta_md, ["alpha", "brier_global", "brier_mean_season"], rows)

    print("Grid search completado")
    print(f"Mejor alpha: {mejor['alpha']:.2f} (Brier mean season {mejor['brier_mean_season']:.4f})")
    print(f"Resultados: {ruta_md}")


if __name__ == "__main__":
    main()
