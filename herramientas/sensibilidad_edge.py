"""
Analisis de sensibilidad del umbral de edge.

Evalua ROI, numero de apuestas, drawdown y Sharpe para distintos edge_min.
"""

import os
import warnings

import numpy as np
import pandas as pd
import joblib

from config import ARCHIVO_FEATURES, ARCHIVO_MODELO, ARCHIVO_FEATURES_PKL, TEST_SIZE
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


def _split_temporal(df: pd.DataFrame, test_size: float = TEST_SIZE):
    n_total = len(df)
    n_test = int(n_total * test_size)
    idx_split = n_total - n_test
    return df.iloc[:idx_split], df.iloc[idx_split:]


def _roi_y_drawdown(y_true, probs, df_cuotas, edge_min):
    cuota_cols = ["B365H", "B365D", "B365A"]
    cuotas = df_cuotas[cuota_cols].values

    retorno_por_bet = []
    bankroll = 100.0
    peak = bankroll
    max_dd = 0.0

    for i in range(len(y_true)):
        prob_raw = 1.0 / cuotas[i]
        if np.any(np.isnan(prob_raw)) or np.any(cuotas[i] <= 1.0):
            continue

        total = prob_raw.sum()
        if total <= 0:
            continue

        prob_mercado = prob_raw / total
        edges = probs[i] - prob_mercado
        pick = int(np.argmax(edges))
        edge = edges[pick]
        if edge < edge_min:
            continue

        if y_true.iloc[i] == pick:
            ret = cuotas[i][pick] - 1.0
        else:
            ret = -1.0

        retorno_por_bet.append(ret)
        bankroll += ret
        if bankroll < 0:
            bankroll = 0.0

        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    if len(retorno_por_bet) == 0:
        return None

    retorno_por_bet = np.array(retorno_por_bet)
    roi = retorno_por_bet.sum() / len(retorno_por_bet)
    return {
        "roi": float(roi),
        "n_bets": int(len(retorno_por_bet)),
        "max_dd": float(max_dd),
        "mean_ret": float(retorno_por_bet.mean()),
        "std_ret": float(retorno_por_bet.std()),
    }


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

    _, df_test = _split_temporal(df)
    X_test = df_test[features]
    y_test = df_test["FTR_numeric"]

    try:
        probs = modelo.predict_proba(X_test)
    except Exception:
        probs = modelo.predict_proba(X_test.fillna(0))

    edge_grid = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15]
    resultados = []

    for edge_min in edge_grid:
        r = _roi_y_drawdown(y_test, probs, df_test, edge_min)
        if r is None:
            resultados.append({
                "edge_min": edge_min,
                "roi": None,
                "n_bets": 0,
                "max_dd": None,
                "sharpe": None,
            })
            continue

        sharpe = None
        if r["std_ret"] > 0:
            sharpe = (r["mean_ret"] / r["std_ret"]) * np.sqrt(r["n_bets"])

        resultados.append({
            "edge_min": edge_min,
            "roi": r["roi"],
            "n_bets": r["n_bets"],
            "max_dd": r["max_dd"],
            "sharpe": sharpe,
        })

    os.makedirs("resultados", exist_ok=True)
    ruta_csv = os.path.join("resultados", "sensibilidad_edge.csv")
    pd.DataFrame(resultados).to_csv(ruta_csv, index=False)

    ruta_md = os.path.join("resultados", "sensibilidad_edge.md")
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write("# Sensibilidad del umbral de edge\n\n")
        f.write("| edge_min | roi | n_bets | max_dd | sharpe |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for r in resultados:
            roi = "-" if r["roi"] is None else f"{r['roi']:+.4f}"
            max_dd = "-" if r["max_dd"] is None else f"{r['max_dd']:.4f}"
            sharpe = "-" if r["sharpe"] is None else f"{r['sharpe']:.3f}"
            f.write(f"| {r['edge_min']:.2f} | {roi} | {r['n_bets']} | {max_dd} | {sharpe} |\n")

    print(f"Resultados guardados en {ruta_md}")


if __name__ == "__main__":
    main()
