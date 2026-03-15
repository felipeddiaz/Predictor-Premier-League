"""
Simulacion Monte Carlo de bankroll basada en apuestas historicas.
"""

from __future__ import annotations

import math
import os
import warnings
import argparse

import numpy as np
import pandas as pd
import joblib

from config import (
    ARCHIVO_FEATURES,
    ARCHIVO_MODELO,
    ARCHIVO_FEATURES_PKL,
    TEST_SIZE,
    KELLY_FRACTION,
    STAKE_MAXIMO_PCT,
    BANKROLL_DEFAULT,
)
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


def _kelly_fraction(prob_modelo: float, cuota: float, kelly_fraction: float) -> float:
    b = cuota - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - prob_modelo
    kelly_full = (b * prob_modelo - q) / b
    kelly_full = max(0.0, kelly_full)
    return min(kelly_full * kelly_fraction, STAKE_MAXIMO_PCT)


def _generar_apuestas(df_test: pd.DataFrame, probs: np.ndarray, edge_minimo=0.10):
    cuota_cols = ["B365H", "B365D", "B365A"]
    if not all(c in df_test.columns for c in cuota_cols):
        raise ValueError("Faltan cuotas B365H/B365D/B365A en el dataset.")

    cuotas = df_test[cuota_cols].values
    apuestas = []

    for i in range(len(df_test)):
        prob_raw = 1.0 / cuotas[i]
        if np.any(np.isnan(prob_raw)) or np.any(cuotas[i] <= 1.0):
            continue
        total = prob_raw.sum()
        if total <= 0:
            continue

        prob_mercado = prob_raw / total
        edges = probs[i] - prob_mercado
        idx = int(np.argmax(edges))
        edge = edges[idx]
        if edge < edge_minimo:
            continue

        apuestas.append({
            "idx": i,
            "resultado_real": int(df_test.iloc[i]["FTR_numeric"]),
            "prob_modelo": float(probs[i][idx]),
            "cuota": float(cuotas[i][idx]),
            "edge": float(edge),
            "pick": idx,
        })

    return apuestas


def simular_bankroll(apuestas: list[dict],
                     n_sim=10000,
                     bankroll_inicial=BANKROLL_DEFAULT,
                     kelly_fraction=KELLY_FRACTION,
                     seed=42):
    rng = np.random.default_rng(seed)

    max_drawdowns = []
    finales = []
    ruinas = 0

    for _ in range(n_sim):
        bankroll = float(bankroll_inicial)
        peak = bankroll
        max_dd = 0.0

        orden = rng.permutation(len(apuestas))
        for j in orden:
            a = apuestas[j]
            f = _kelly_fraction(a["prob_modelo"], a["cuota"], kelly_fraction)
            if f <= 0:
                continue
            stake = bankroll * f
            if stake <= 0:
                continue

            if a["pick"] == a["resultado_real"]:
                bankroll += stake * (a["cuota"] - 1.0)
            else:
                bankroll -= stake

            if bankroll <= 0:
                bankroll = 0
                break

            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        if bankroll <= 0:
            ruinas += 1

        max_drawdowns.append(max_dd)
        finales.append(bankroll)

    finales = np.array(finales)
    max_drawdowns = np.array(max_drawdowns)

    return {
        "n_sim": n_sim,
        "ruina_pct": (ruinas / n_sim) * 100.0,
        "final_mean": float(finales.mean()),
        "final_std": float(finales.std()),
        "final_p5": float(np.percentile(finales, 5)),
        "final_p50": float(np.percentile(finales, 50)),
        "final_p95": float(np.percentile(finales, 95)),
        "max_dd_mean": float(max_drawdowns.mean()),
        "max_dd_p95": float(np.percentile(max_drawdowns, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sim", type=int, default=10000)
    parser.add_argument("--edge_min", type=float, default=0.10)
    parser.add_argument("--bankroll", type=float, default=BANKROLL_DEFAULT)
    args = parser.parse_args()

    print("=" * 70)
    print("SIMULACION MONTE CARLO DE BANKROLL")
    print("=" * 70)

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

    try:
        probs = modelo.predict_proba(X_test)
    except Exception:
        probs = modelo.predict_proba(X_test.fillna(0))

    apuestas = _generar_apuestas(df_test, probs, edge_minimo=args.edge_min)
    if not apuestas:
        raise RuntimeError("No se generaron apuestas (edge_minimo alto o datos insuficientes).")

    resultados = simular_bankroll(apuestas, n_sim=args.n_sim, bankroll_inicial=args.bankroll)

    os.makedirs("resultados", exist_ok=True)
    ruta_md = os.path.join("resultados", "simulacion_montecarlo.md")

    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write("# Simulacion Monte Carlo de bankroll\n\n")
        f.write(f"- Simulaciones: {resultados['n_sim']}\n")
        f.write(f"- Prob. ruina: {resultados['ruina_pct']:.2f}%\n")
        f.write(f"- Bankroll final (mean): {resultados['final_mean']:.2f}\n")
        f.write(f"- Bankroll final (std): {resultados['final_std']:.2f}\n")
        f.write(f"- Bankroll final (p5/p50/p95): {resultados['final_p5']:.2f} / {resultados['final_p50']:.2f} / {resultados['final_p95']:.2f}\n")
        f.write(f"- Max drawdown (mean): {resultados['max_dd_mean']:.2%}\n")
        f.write(f"- Max drawdown (p95): {resultados['max_dd_p95']:.2%}\n")

    print(f"Resultados guardados en {ruta_md}")


if __name__ == "__main__":
    main()
