# -*- coding: utf-8 -*-
"""
Analiza importancia de features para los 3 modelos binarios nuevos:
  - Over/Under 2.5
  - Tarjetas > 3.5
  - Corners > 9.5

Uso:
    python backup/analizar_features.py

Salida:
  - Imprime en terminal la importancia (%) por feature para cada modelo.
  - Guarda CSVs en resultados/feature_importance_binarios_backup/
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from config import (
    ARCHIVO_FEATURES,
    ARCHIVO_MODELO_OU,
    ARCHIVO_FEATURES_OU,
    ARCHIVO_MODELO_TARJETAS,
    ARCHIVO_FEATURES_TARJETAS,
    ARCHIVO_MODELO_CORNERS,
    ARCHIVO_FEATURES_CORNERS,
    TEST_SIZE,
    RANDOM_SEED,
)
from utils import (
    agregar_xg_rolling,
    agregar_features_tabla,
    agregar_features_forma_momentum,
    agregar_features_descanso,
    agregar_features_arbitro,
    agregar_features_elo,
    agregar_features_sor,
    agregar_features_goles_binarias,
    agregar_features_tarjetas_binarias,
    agregar_features_corners_binarias,
)

warnings.filterwarnings("ignore")


def _preparar_base(df: pd.DataFrame) -> pd.DataFrame:
    df = agregar_xg_rolling(df)
    df = agregar_features_tabla(df)
    df = agregar_features_forma_momentum(df)
    df = agregar_features_descanso(df)
    df = agregar_features_arbitro(df)
    df = agregar_features_elo(df)
    df = agregar_features_sor(df)
    return df


def _target_ou(df: pd.DataFrame) -> pd.Series:
    return ((df["FTHG"].fillna(0) + df["FTAG"].fillna(0)) > 2.5).astype(int)


def _target_tarjetas(df: pd.DataFrame) -> pd.Series:
    return ((df["HY"].fillna(0) + df["AY"].fillna(0)) > 3.5).astype(int)


def _target_corners(df: pd.DataFrame) -> pd.Series:
    return ((df["HC"].fillna(0) + df["AC"].fillna(0)) > 9.5).astype(int)


MARKETS = {
    "over_under": {
        "model_path": ARCHIVO_MODELO_OU,
        "features_path": ARCHIVO_FEATURES_OU,
        "fe_fn": agregar_features_goles_binarias,
        "target_fn": _target_ou,
    },
    "tarjetas": {
        "model_path": ARCHIVO_MODELO_TARJETAS,
        "features_path": ARCHIVO_FEATURES_TARJETAS,
        "fe_fn": agregar_features_tarjetas_binarias,
        "target_fn": _target_tarjetas,
    },
    "corners": {
        "model_path": ARCHIVO_MODELO_CORNERS,
        "features_path": ARCHIVO_FEATURES_CORNERS,
        "fe_fn": agregar_features_corners_binarias,
        "target_fn": _target_corners,
    },
}


def _cargar_dataset() -> pd.DataFrame:
    df = pd.read_csv(ARCHIVO_FEATURES)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed", dayfirst=True)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def _needs_fillna(modelo) -> bool:
    return "randomforest" in modelo.__class__.__name__.lower()


def _imprimir_importancias(nombre: str, df_imp: pd.DataFrame):
    print("\n" + "=" * 86)
    print(f"IMPORTANCIA DE FEATURES (%) - {nombre.upper()}")
    print("=" * 86)
    print(f"{'#':<3} {'Feature':<34} {'Intrinsic%':>11} {'Perm%+':>10} {'PermMean':>11}")
    print("-" * 86)
    for i, (_, row) in enumerate(df_imp.iterrows(), 1):
        print(
            f"{i:<3} {row['feature']:<34} "
            f"{row['intrinsic_pct']:>10.2f}% {row['perm_pct_pos']:>9.2f}% {row['perm_mean']:>11.6f}"
        )

    drops = df_imp[df_imp['perm_mean'] < 0]['feature'].tolist()
    print("\nFeatures sugeridas para excluir (perm_mean < 0):")
    if not drops:
        print("  - Ninguna")
    else:
        for feat in drops:
            print(f"  - {feat}")


def _analizar_market(nombre: str, cfg: dict, out_dir: str):
    if not os.path.exists(cfg["model_path"]) or not os.path.exists(cfg["features_path"]):
        raise FileNotFoundError(f"Faltan archivos para {nombre}: {cfg['model_path']} / {cfg['features_path']}")

    df = _cargar_dataset()
    df = _preparar_base(df)
    df = cfg["fe_fn"](df)
    y = cfg["target_fn"](df)

    features = joblib.load(cfg["features_path"])
    features = [f for f in features if f in df.columns]

    X = df[features]
    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

    modelo = joblib.load(cfg["model_path"])
    X_eval = X_test.fillna(0) if _needs_fillna(modelo) else X_test

    if hasattr(modelo, "feature_importances_"):
        intrinsic = np.array(modelo.feature_importances_, dtype=float)
    else:
        intrinsic = np.zeros(len(features), dtype=float)

    perm = permutation_importance(
        modelo,
        X_eval,
        y_test,
        scoring="neg_log_loss",
        n_repeats=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    df_imp = pd.DataFrame(
        {
            "feature": features,
            "intrinsic": intrinsic,
            "perm_mean": perm.importances_mean,
            "perm_std": perm.importances_std,
        }
    )

    total_intr = float(df_imp["intrinsic"].sum())
    df_imp["intrinsic_pct"] = (df_imp["intrinsic"] / total_intr * 100.0) if total_intr > 0 else 0.0

    perm_pos = df_imp["perm_mean"].clip(lower=0)
    total_perm_pos = float(perm_pos.sum())
    df_imp["perm_pct_pos"] = (perm_pos / total_perm_pos * 100.0) if total_perm_pos > 0 else 0.0

    df_imp = df_imp.sort_values("intrinsic_pct", ascending=False).reset_index(drop=True)

    _imprimir_importancias(nombre, df_imp)

    out_csv = os.path.join(out_dir, f"feature_importance_{nombre}.csv")
    df_imp.to_csv(out_csv, index=False)
    print(f"\nCSV guardado: {out_csv}")


def main():
    out_dir = os.path.join("resultados", "feature_importance_binarios_backup")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 86)
    print("ANALISIS DE FEATURES - MODELOS BINARIOS (BACKUP/analizar_features.py)")
    print("=" * 86)

    for market, cfg in MARKETS.items():
        _analizar_market(market, cfg, out_dir)

    print("\nListo. Revisar tambien los CSV en resultados/feature_importance_binarios_backup/")


if __name__ == "__main__":
    main()
