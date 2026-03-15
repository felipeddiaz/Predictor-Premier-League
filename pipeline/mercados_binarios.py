# -*- coding: utf-8 -*-
"""
Entrenamiento genérico para mercados binarios (O/U, tarjetas, corners).
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import (
    ARCHIVO_FEATURES_MERCADOS,
    RANDOM_SEED,
    TEST_SIZE,
    UMBRAL_EDGE_BINARIO,
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


def _asignar_temporada(dates: pd.Series) -> pd.Series:
    return dates.apply(
        lambda d: f"{d.year}-{str(d.year+1)[-2:]}" if d.month >= 8
        else f"{d.year-1}-{str(d.year)[-2:]}"
    )


def _roi_binario(y_true, p_over, df_eval, over_col=None, under_col=None, edge_min=UMBRAL_EDGE_BINARIO):
    if over_col is None or under_col is None:
        return None
    if over_col not in df_eval.columns or under_col not in df_eval.columns:
        return None

    odds_over = pd.to_numeric(df_eval[over_col], errors="coerce").values
    odds_under = pd.to_numeric(df_eval[under_col], errors="coerce").values
    bankroll = 0.0
    bets = 0

    for i in range(len(y_true)):
        oo = odds_over[i]
        ou = odds_under[i]
        if np.isnan(oo) or np.isnan(ou) or oo <= 1.0 or ou <= 1.0:
            continue

        imp_o = 1.0 / oo
        imp_u = 1.0 / ou
        total = imp_o + imp_u
        if total <= 0:
            continue

        fair_o = imp_o / total
        fair_u = imp_u / total
        p_o = float(p_over[i])
        p_u = 1.0 - p_o

        edge_o = p_o - fair_o
        edge_u = p_u - fair_u

        if max(edge_o, edge_u) < edge_min:
            continue

        bets += 1
        apostar_over = edge_o >= edge_u
        real_over = int(y_true.iloc[i]) == 1
        if apostar_over:
            bankroll += (oo - 1.0) if real_over else -1.0
        else:
            bankroll += (ou - 1.0) if not real_over else -1.0

    if bets == 0:
        return None
    return bankroll / bets


def _entrenar_candidatos(X_train, y_train, X_test, y_test, roi_kwargs):
    modelos = {}

    rf_base = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=4,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf_base.fit(X_train.fillna(0), y_train)
    modelos["RF_Basico"] = rf_base

    rf_bal = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf_bal.fit(X_train.fillna(0), y_train)
    modelos["RF_Balanceado"] = rf_bal

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=8,
        gamma=0.5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=40,
    )
    sw = compute_sample_weight(class_weight="balanced", y=y_train)
    xgb.fit(X_train, y_train, sample_weight=sw, eval_set=[(X_test, y_test)], verbose=False)
    modelos["XGBoost"] = xgb

    resultados = {}
    for nombre, modelo in modelos.items():
        x_eval = X_test if nombre == "XGBoost" else X_test.fillna(0)
        probs = modelo.predict_proba(x_eval)[:, 1]
        pred = (probs >= 0.5).astype(int)
        roi = _roi_binario(y_test, probs, **roi_kwargs)
        resultados[nombre] = {
            "modelo": modelo,
            "log_loss": log_loss(y_test, probs, labels=[0, 1]),
            "brier": brier_score_loss(y_test, probs),
            "f1": f1_score(y_test, pred),
            "acc": accuracy_score(y_test, pred),
            "roi": roi,
        }
    return resultados


def _seleccionar_modelo(resultados):
    elegibles = {k: v for k, v in resultados.items() if v["roi"] is None or v["roi"] >= 0}
    pool = elegibles if elegibles else resultados
    mejor_key = min(pool.items(), key=lambda x: (x[1]["log_loss"], x[1]["brier"]))[0]
    return mejor_key, resultados[mejor_key]


def _walk_forward(df, features, target_col, modelo_key, roi_kwargs):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["_Season"] = _asignar_temporada(df["Date"])
    seasons = [s for s in sorted(df["_Season"].unique()) if s >= "2020-21"]

    resultados = []
    for season in seasons:
        m_train = df["_Season"] < season
        m_test = df["_Season"] == season
        if m_train.sum() == 0 or m_test.sum() == 0:
            continue

        X_train = df.loc[m_train, features]
        y_train = df.loc[m_train, target_col]
        X_test = df.loc[m_test, features]
        y_test = df.loc[m_test, target_col]
        df_eval = df.loc[m_test]

        if modelo_key == "XGBoost":
            m = XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=2.0,
                min_child_weight=8,
                gamma=0.5,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                eval_metric="logloss",
            )
            sw = compute_sample_weight(class_weight="balanced", y=y_train)
            m.fit(X_train, y_train, sample_weight=sw)
            probs = m.predict_proba(X_test)[:, 1]
            pred = (probs >= 0.5).astype(int)
        else:
            m = RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                class_weight="balanced" if modelo_key == "RF_Balanceado" else None,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
            m.fit(X_train.fillna(0), y_train)
            probs = m.predict_proba(X_test.fillna(0))[:, 1]
            pred = (probs >= 0.5).astype(int)

        roi = _roi_binario(y_test, probs, df_eval=df_eval, over_col=roi_kwargs.get("over_col"), under_col=roi_kwargs.get("under_col"))
        resultados.append({
            "season": season,
            "log_loss": log_loss(y_test, probs, labels=[0, 1]),
            "brier": brier_score_loss(y_test, probs),
            "f1": f1_score(y_test, pred),
            "acc": accuracy_score(y_test, pred),
            "roi": roi,
        })
    return resultados


def _preparar_base(df: pd.DataFrame) -> pd.DataFrame:
    df = agregar_xg_rolling(df)
    df = agregar_features_tabla(df)
    df = agregar_features_forma_momentum(df)
    df = agregar_features_descanso(df)
    df = agregar_features_arbitro(df)
    df = agregar_features_elo(df)
    df = agregar_features_sor(df)
    return df


def entrenar_mercado_binario(nombre_mercado: str, target_col: str, features: list,
                             over_col: str | None, under_col: str | None,
                             archivo_modelo: str, archivo_features: str,
                             target_builder=None):
    print("=" * 70)
    print(f"ENTRENAMIENTO MERCADO: {nombre_mercado}")
    print("=" * 70)

    df = pd.read_csv(ARCHIVO_FEATURES_MERCADOS)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    df = _preparar_base(df)
    if nombre_mercado == "OVER_UNDER":
        df = agregar_features_goles_binarias(df)
    elif nombre_mercado == "TARJETAS":
        df = agregar_features_tarjetas_binarias(df)
    elif nombre_mercado == "CORNERS":
        df = agregar_features_corners_binarias(df)

    if target_builder is not None:
        df[target_col] = target_builder(df)

    features_ok = [f for f in features if f in df.columns]
    X = df[features_ok]
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False)
    df_eval = df.loc[X_test.index]

    roi_kwargs = {"df_eval": df_eval, "over_col": over_col, "under_col": under_col}
    resultados = _entrenar_candidatos(X_train, y_train, X_test, y_test, roi_kwargs)

    print(f"\n{'Modelo':<15} {'LogLoss':>8} {'Brier':>8} {'F1':>8} {'Acc':>8} {'ROI':>8}")
    print("-" * 62)
    for k, r in resultados.items():
        roi_s = f"{r['roi']:+.2%}" if r['roi'] is not None else "N/A"
        print(f"{k:<15} {r['log_loss']:>8.4f} {r['brier']:>8.4f} {r['f1']:>8.4f} {r['acc']:>7.2%} {roi_s:>8}")

    mejor_key, mejor = _seleccionar_modelo(resultados)
    print(f"\nGANADOR: {mejor_key} | LogLoss={mejor['log_loss']:.4f} Brier={mejor['brier']:.4f} ROI={mejor['roi'] if mejor['roi'] is not None else 'N/A'}")

    wf = _walk_forward(df, features_ok, target_col, mejor_key, {"over_col": over_col, "under_col": under_col})
    if wf:
        print("\nWalk-forward:")
        print(f"{'Season':<10} {'LogLoss':>8} {'Brier':>8} {'F1':>8} {'Acc':>8} {'ROI':>8}")
        print("-" * 58)
        for r in wf:
            roi_s = f"{r['roi']:+.2%}" if r['roi'] is not None else "N/A"
            print(f"{r['season']:<10} {r['log_loss']:>8.4f} {r['brier']:>8.4f} {r['f1']:>8.4f} {r['acc']:>7.2%} {roi_s:>8}")
        avg_roi_vals = [r['roi'] for r in wf if r['roi'] is not None]
        avg_roi = np.mean(avg_roi_vals) if avg_roi_vals else None
        avg_roi_s = f"{avg_roi:+.2%}" if avg_roi is not None else "N/A"
        print("-" * 58)
        print(f"{'PROMEDIO':<10} {np.mean([r['log_loss'] for r in wf]):>8.4f} {np.mean([r['brier'] for r in wf]):>8.4f} "
              f"{np.mean([r['f1'] for r in wf]):>8.4f} {np.mean([r['acc'] for r in wf]):>7.2%} {avg_roi_s:>8}")

    modelo_final = mejor['modelo']
    joblib.dump(modelo_final, archivo_modelo)
    joblib.dump(features_ok, archivo_features)
    print(f"\nModelo guardado: {archivo_modelo}")
    print(f"Features guardadas: {archivo_features} ({len(features_ok)})")
