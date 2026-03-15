# -*- coding: utf-8 -*-
"""
Evalua importancia de features para los nuevos modelos binarios:
  - Over/Under 2.5
  - Tarjetas >3.5
  - Corners >9.5

Genera reportes por mercado y recomendaciones de exclusion de features
de bajo valor usando importancia por permutacion en holdout temporal.
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from config import (
    ARCHIVO_FEATURES,
    ARCHIVO_FEATURES_MERCADOS,
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


def _cargar_dataset_binarios() -> pd.DataFrame:
    """Usa RESTAURADO como base y mergea cuotas O/U desde con_features."""
    df = pd.read_csv(ARCHIVO_FEATURES)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed", dayfirst=True)
    df = df.dropna(subset=["Date"]).copy()

    if os.path.exists(ARCHIVO_FEATURES_MERCADOS):
        odds_cols = [
            "Date", "HomeTeam", "AwayTeam",
            "B365>2.5", "B365<2.5", "P>2.5", "P<2.5",
            "B365C>2.5", "B365C<2.5", "PC>2.5", "PC<2.5",
        ]
        df_odds = pd.read_csv(ARCHIVO_FEATURES_MERCADOS)
        keep = [c for c in odds_cols if c in df_odds.columns]
        if {"Date", "HomeTeam", "AwayTeam"}.issubset(keep):
            df_odds = df_odds[keep].copy()
            df_odds["Date"] = pd.to_datetime(df_odds["Date"], errors="coerce", format="mixed", dayfirst=True)
            df_odds = df_odds.dropna(subset=["Date"])
            df_odds = df_odds.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"], keep="last")
            df = df.merge(df_odds, on=["Date", "HomeTeam", "AwayTeam"], how="left")

    return df.sort_values("Date").reset_index(drop=True)


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
        "modelo": ARCHIVO_MODELO_OU,
        "features": ARCHIVO_FEATURES_OU,
        "fe_fn": agregar_features_goles_binarias,
        "target_fn": _target_ou,
    },
    "tarjetas": {
        "modelo": ARCHIVO_MODELO_TARJETAS,
        "features": ARCHIVO_FEATURES_TARJETAS,
        "fe_fn": agregar_features_tarjetas_binarias,
        "target_fn": _target_tarjetas,
    },
    "corners": {
        "modelo": ARCHIVO_MODELO_CORNERS,
        "features": ARCHIVO_FEATURES_CORNERS,
        "fe_fn": agregar_features_corners_binarias,
        "target_fn": _target_corners,
    },
}


def _needs_fillna(modelo) -> bool:
    name = modelo.__class__.__name__.lower()
    return "randomforest" in name


def _evaluar_market(nombre: str, cfg: dict, out_dir: str) -> dict:
    if not os.path.exists(cfg["modelo"]) or not os.path.exists(cfg["features"]):
        raise FileNotFoundError(f"Faltan artefactos de {nombre}: {cfg['modelo']} / {cfg['features']}")

    df = _cargar_dataset_binarios()

    df = _preparar_base(df)
    df = cfg["fe_fn"](df)
    y = cfg["target_fn"](df)

    features = joblib.load(cfg["features"])
    features = [f for f in features if f in df.columns]
    X = df[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)

    modelo = joblib.load(cfg["modelo"])
    X_test_eval = X_test.fillna(0) if _needs_fillna(modelo) else X_test

    probs = modelo.predict_proba(X_test_eval)[:, 1]
    preds = (probs >= 0.5).astype(int)

    ll = log_loss(y_test, probs, labels=[0, 1])
    brier = brier_score_loss(y_test, probs)
    acc = accuracy_score(y_test, preds)

    perm = permutation_importance(
        modelo,
        X_test_eval,
        y_test,
        scoring="neg_log_loss",
        n_repeats=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    if hasattr(modelo, "feature_importances_"):
        intrinsic = np.array(modelo.feature_importances_)
    else:
        intrinsic = np.zeros(len(features), dtype=float)

    df_imp = pd.DataFrame(
        {
            "feature": features,
            "intrinsic": intrinsic,
            "perm_mean": perm.importances_mean,
            "perm_std": perm.importances_std,
        }
    )

    total_intrinsic = float(df_imp["intrinsic"].sum())
    if total_intrinsic > 0:
        df_imp["intrinsic_pct"] = (df_imp["intrinsic"] / total_intrinsic) * 100
    else:
        df_imp["intrinsic_pct"] = 0.0

    perm_pos = df_imp["perm_mean"].clip(lower=0)
    perm_pos_total = float(perm_pos.sum())
    if perm_pos_total > 0:
        df_imp["perm_pct_pos"] = (perm_pos / perm_pos_total) * 100
    else:
        df_imp["perm_pct_pos"] = 0.0

    df_imp = df_imp.sort_values("perm_mean", ascending=False).reset_index(drop=True)
    df_imp["rank"] = np.arange(1, len(df_imp) + 1)

    # Regla de exclusion sugerida:
    # - no tocar el top de ranking
    # - excluir principalmente cola con permutacion consistentemente no positiva
    n_feats = len(df_imp)
    tail_cut = int(np.ceil(n_feats * 0.70))
    zero_tail_cut = int(np.ceil(n_feats * 0.80))
    df_imp["drop_recommended"] = (
        (
            (df_imp["perm_mean"] < 0)
            & ((df_imp["perm_mean"] + df_imp["perm_std"]) <= 0)
            & (df_imp["rank"] >= tail_cut)
        )
        |
        (
            (df_imp["perm_mean"] == 0)
            & (df_imp["rank"] >= zero_tail_cut)
        )
    )

    # Keep set sugerido: todo lo no-drop
    keep_features = df_imp.loc[~df_imp["drop_recommended"], "feature"].tolist()
    drop_features = df_imp.loc[df_imp["drop_recommended"], "feature"].tolist()

    csv_path = os.path.join(out_dir, f"feature_importance_{nombre}.csv")
    df_imp.to_csv(csv_path, index=False)

    keep_path = os.path.join(out_dir, f"features_keep_{nombre}.txt")
    with open(keep_path, "w", encoding="utf-8") as f:
        for feat in keep_features:
            f.write(f"{feat}\n")

    drop_path = os.path.join(out_dir, f"features_drop_{nombre}.txt")
    with open(drop_path, "w", encoding="utf-8") as f:
        for feat in drop_features:
            f.write(f"{feat}\n")

    return {
        "market": nombre,
        "log_loss": ll,
        "brier": brier,
        "accuracy": acc,
        "n_features": len(features),
        "n_keep": len(keep_features),
        "n_drop": len(drop_features),
        "csv": csv_path,
        "keep_file": keep_path,
        "drop_file": drop_path,
        "df_importance": df_imp,
    }


def main():
    out_dir = os.path.join("resultados", "feature_importance_binarios")
    os.makedirs(out_dir, exist_ok=True)

    resumen = []
    for market, cfg in MARKETS.items():
        print("=" * 72)
        print(f"Evaluando importancia: {market}")
        print("=" * 72)
        r = _evaluar_market(market, cfg, out_dir)
        resumen.append(r)
        print(
            f"LogLoss={r['log_loss']:.4f} | Brier={r['brier']:.4f} | Acc={r['accuracy']:.2%} | "
            f"Features={r['n_features']} | Keep={r['n_keep']} | Drop={r['n_drop']}"
        )
        print("\nImportancia por feature (%):")
        print(f"{'#':<3} {'Feature':<32} {'Intrinsic%':>11} {'Perm%+':>10} {'PermMean':>10}")
        print("-" * 74)
        for i, (_, row) in enumerate(r["df_importance"].iterrows(), 1):
            print(
                f"{i:<3} {row['feature']:<32} {row['intrinsic_pct']:>10.2f}% "
                f"{row['perm_pct_pos']:>9.2f}% {row['perm_mean']:>10.6f}"
            )
        print()

    resumen_df = pd.DataFrame(resumen)
    resumen_csv = os.path.join(out_dir, "resumen_importancia_binarios.csv")
    resumen_df.to_csv(resumen_csv, index=False)

    md_path = os.path.join(out_dir, "resumen_importancia_binarios.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Resumen importancia de features (mercados binarios)\n\n")
        f.write("| Mercado | LogLoss | Brier | Accuracy | Features | Keep | Drop |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for _, r in resumen_df.iterrows():
            f.write(
                f"| {r['market']} | {r['log_loss']:.4f} | {r['brier']:.4f} | {r['accuracy']:.2%} | "
                f"{int(r['n_features'])} | {int(r['n_keep'])} | {int(r['n_drop'])} |\n"
            )
        f.write("\n")
        for _, r in resumen_df.iterrows():
            f.write(f"- `{r['market']}`:\n")
            f.write(f"  - csv importancia: `{r['csv']}`\n")
            f.write(f"  - keep list: `{r['keep_file']}`\n")
            f.write(f"  - drop list: `{r['drop_file']}`\n")

    print("\nReportes generados:")
    print(f"- {resumen_csv}")
    print(f"- {md_path}")


if __name__ == "__main__":
    main()
