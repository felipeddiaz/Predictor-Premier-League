"""
Deteccion de concept drift con PSI por temporada.
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


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf

    e_counts, _ = np.histogram(expected, bins=quantiles)
    a_counts, _ = np.histogram(actual, bins=quantiles)

    e_pct = e_counts / max(len(expected), 1)
    a_pct = a_counts / max(len(actual), 1)

    eps = 1e-6
    e_pct = np.clip(e_pct, eps, None)
    a_pct = np.clip(a_pct, eps, None)

    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def main():
    print("=" * 70)
    print("DETECCION DE CONCEPT DRIFT (PSI)")
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

    df["Season"] = _asignar_temporada(df["Date"])
    temporadas = sorted(df["Season"].unique())
    base_season = temporadas[0]

    # Top features por importancia si el modelo la expone
    if hasattr(modelo, "feature_importances_"):
        importances = modelo.feature_importances_
        df_imp = pd.DataFrame({"Feature": features, "Importance": importances})
        top_features = df_imp.sort_values("Importance", ascending=False)["Feature"].head(10).tolist()
    else:
        top_features = features[:10]

    registros = []
    df_base = df[df["Season"] == base_season]

    for season in temporadas[1:]:
        df_season = df[df["Season"] == season]
        for feat in top_features:
            psi_val = psi(df_base[feat].values, df_season[feat].values, bins=10)
            registros.append({
                "season": season,
                "feature": feat,
                "psi": psi_val,
            })

    psi_df = pd.DataFrame(registros)
    os.makedirs("resultados", exist_ok=True)
    ruta_csv = os.path.join("resultados", "concept_drift_psi.csv")
    psi_df.to_csv(ruta_csv, index=False)

    resumen = (
        psi_df.groupby("season")["psi"]
        .agg(["mean", "max"])
        .reset_index()
        .sort_values("season")
    )

    def _write_md_table(f, df_table: pd.DataFrame):
        headers = "| " + " | ".join(df_table.columns) + " |\n"
        sep = "| " + " | ".join(["---"] * len(df_table.columns)) + " |\n"
        f.write(headers)
        f.write(sep)
        for _, row in df_table.iterrows():
            f.write("| " + " | ".join([str(x) for x in row.values]) + " |\n")

    ruta_md = os.path.join("resultados", "concept_drift_psi.md")
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write("# Concept Drift (PSI)\n\n")
        f.write(f"- Base season: {base_season}\n")
        f.write(f"- Top features: {', '.join(top_features)}\n\n")
        f.write("## Resumen por temporada\n")
        _write_md_table(f, resumen)
        f.write("\n## PSI por feature y temporada\n")
        _write_md_table(f, psi_df)

    print(f"PSI guardado en {ruta_csv}")
    print(f"Resumen guardado en {ruta_md}")


if __name__ == "__main__":
    main()
