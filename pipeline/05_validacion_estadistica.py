"""
Validacion estadistica del ROI.

Incluye:
  - Test de permutacion para significancia del ROI.
  - Bootstrap para intervalo de confianza del ROI.

Usa el modelo y features guardados en modelos/.
"""

import os
import warnings
import argparse

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


def _roi_simulado(y_true, probs, df_cuotas, edge_minimo=0.10):
    cuota_cols = ["B365H", "B365D", "B365A"]
    if not all(c in df_cuotas.columns for c in cuota_cols):
        return None

    cuotas = df_cuotas[cuota_cols].values
    apostado = 0.0
    ganancia = 0.0

    for i in range(len(y_true)):
        prob_raw = 1.0 / cuotas[i]
        if np.any(np.isnan(prob_raw)) or np.any(cuotas[i] <= 1.0):
            continue

        total = prob_raw.sum()
        if total <= 0:
            continue

        prob_mercado = prob_raw / total
        edges = probs[i] - prob_mercado
        mejor_resultado = int(np.argmax(edges))
        mejor_edge = edges[mejor_resultado]

        if mejor_edge >= edge_minimo:
            apostado += 1.0
            if y_true.iloc[i] == mejor_resultado:
                ganancia += cuotas[i][mejor_resultado] - 1.0
            else:
                ganancia -= 1.0

    if apostado == 0:
        return None

    return ganancia / apostado


def _split_temporal(df: pd.DataFrame, test_size: float = TEST_SIZE):
    n_total = len(df)
    n_test = int(n_total * test_size)
    idx_split = n_total - n_test
    return df.iloc[:idx_split], df.iloc[idx_split:]


def test_permutacion(y_true, probs, df_cuotas, n_perm=10000, edge_minimo=0.10, seed=42):
    rng = np.random.default_rng(seed)
    roi_real = _roi_simulado(y_true, probs, df_cuotas, edge_minimo=edge_minimo)
    if roi_real is None:
        return None

    rois = []
    y_vals = y_true.values
    for _ in range(n_perm):
        y_perm = rng.permutation(y_vals)
        roi = _roi_simulado(pd.Series(y_perm, index=y_true.index), probs, df_cuotas, edge_minimo=edge_minimo)
        if roi is not None:
            rois.append(roi)

    rois = np.array(rois)
    p_value = float((rois >= roi_real).mean()) if len(rois) > 0 else None
    return {
        "roi_real": float(roi_real),
        "p_value": p_value,
        "roi_perm_mean": float(np.mean(rois)) if len(rois) > 0 else None,
        "roi_perm_std": float(np.std(rois)) if len(rois) > 0 else None,
        "n_perm": int(n_perm),
        "n_valid": int(len(rois)),
    }


def bootstrap_roi(y_true, probs, df_cuotas, n_boot=1000, edge_minimo=0.10, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rois = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        roi = _roi_simulado(y_true.iloc[idx].reset_index(drop=True),
                            probs[idx],
                            df_cuotas.iloc[idx].reset_index(drop=True),
                            edge_minimo=edge_minimo)
        if roi is not None:
            rois.append(roi)

    rois = np.array(rois)
    if len(rois) == 0:
        return None

    ci_low, ci_high = np.percentile(rois, [2.5, 97.5])
    return {
        "roi_mean": float(np.mean(rois)),
        "roi_std": float(np.std(rois)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_boot": int(n_boot),
        "n_valid": int(len(rois)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--edge_min", type=float, default=0.10)
    args = parser.parse_args()

    print("=" * 70)
    print("VALIDACION ESTADISTICA DEL ROI")
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

    df_train, df_test = _split_temporal(df)
    X_test = df_test[features]
    y_test = df_test["FTR_numeric"]

    try:
        probs = modelo.predict_proba(X_test)
    except Exception:
        probs = modelo.predict_proba(X_test.fillna(0))

    resultados_perm = test_permutacion(y_test, probs, df_test, n_perm=args.n_perm, edge_minimo=args.edge_min)
    resultados_boot = bootstrap_roi(y_test, probs, df_test, n_boot=args.n_boot, edge_minimo=args.edge_min)

    os.makedirs("resultados", exist_ok=True)
    ruta_md = os.path.join("resultados", "validacion_estadistica.md")

    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write("# Validacion estadistica del ROI\n\n")
        f.write("## Permutacion\n")
        if resultados_perm is None:
            f.write("- Sin ROI valido (no hubo apuestas).\n")
        else:
            f.write(f"- ROI real: {resultados_perm['roi_real']:+.4f}\n")
            f.write(f"- p-value: {resultados_perm['p_value']}\n")
            f.write(f"- ROI perm mean: {resultados_perm['roi_perm_mean']:+.4f}\n")
            f.write(f"- ROI perm std: {resultados_perm['roi_perm_std']:.4f}\n")
            f.write(f"- Permutaciones validas: {resultados_perm['n_valid']}/{resultados_perm['n_perm']}\n")

        f.write("\n## Bootstrap\n")
        if resultados_boot is None:
            f.write("- Sin ROI valido (no hubo apuestas).\n")
        else:
            f.write(f"- ROI mean: {resultados_boot['roi_mean']:+.4f}\n")
            f.write(f"- ROI std: {resultados_boot['roi_std']:.4f}\n")
            f.write(f"- CI 95%: [{resultados_boot['ci_low']:+.4f}, {resultados_boot['ci_high']:+.4f}]\n")
            f.write(f"- Resamples validos: {resultados_boot['n_valid']}/{resultados_boot['n_boot']}\n")

    print(f"Resultados guardados en {ruta_md}")


if __name__ == "__main__":
    main()
