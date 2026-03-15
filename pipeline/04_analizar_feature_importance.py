# -*- coding: utf-8 -*-
"""
ANÁLISIS DE IMPORTANCIA DE FEATURES — Todos los modelos.

Carga los modelos entrenados y extrae feature_importances_ directamente.
Analiza: modelo principal (1X2), over/under, tarjetas, corners.
Identifica features redundantes y de baja importancia.
"""
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RUTA_MODELOS

warnings.filterwarnings('ignore')


def _extraer_importancias(modelo, features):
    """Extrae feature_importances_ de un modelo (RF, XGB o CalibratedClassifierCV)."""
    if hasattr(modelo, 'feature_importances_'):
        return modelo.feature_importances_

    # CalibratedClassifierCV: extraer del estimador base
    if hasattr(modelo, 'calibrated_classifiers_'):
        importancias = []
        for cc in modelo.calibrated_classifiers_:
            if hasattr(cc.estimator, 'feature_importances_'):
                importancias.append(cc.estimator.feature_importances_)
        if importancias:
            return np.mean(importancias, axis=0)

    if hasattr(modelo, 'estimator') and hasattr(modelo.estimator, 'feature_importances_'):
        return modelo.estimator.feature_importances_

    return None


def _analizar_modelo(nombre, modelo, features):
    """Analiza un modelo y retorna DataFrame de importancias."""
    imp = _extraer_importancias(modelo, features)
    if imp is None:
        print(f"  No se pudieron extraer importancias de {nombre}")
        return None

    df = pd.DataFrame({
        'Feature': features,
        'Importancia': imp,
    }).sort_values('Importancia', ascending=False)

    total = df['Importancia'].sum()
    df['Porcentaje'] = (df['Importancia'] / total) * 100 if total > 0 else 0
    df['Porcentaje_Acumulado'] = df['Porcentaje'].cumsum()
    df['Modelo'] = nombre
    return df


MODELOS_A_ANALIZAR = [
    ('1X2 Principal', 'modelo_final_optimizado.pkl', 'features.pkl'),
    ('1X2 Sin Cuotas', 'modelo_value_betting.pkl', 'features_value_betting.pkl'),
    ('Over/Under 2.5', 'modelo_over_under.pkl', 'features_over_under.pkl'),
    ('Tarjetas 3.5', 'modelo_tarjetas.pkl', 'features_tarjetas.pkl'),
    ('Corners 9.5', 'modelo_corners.pkl', 'features_corners.pkl'),
]


if __name__ == '__main__':
    print("=" * 70)
    print("   ANÁLISIS DE IMPORTANCIA DE FEATURES — TODOS LOS MODELOS")
    print("=" * 70)

    resultados_todos = []

    for nombre, modelo_file, features_file in MODELOS_A_ANALIZAR:
        ruta_modelo = os.path.join(RUTA_MODELOS, modelo_file)
        ruta_features = os.path.join(RUTA_MODELOS, features_file)

        if not os.path.exists(ruta_modelo) or not os.path.exists(ruta_features):
            print(f"\n  {nombre}: archivos no encontrados, saltando")
            continue

        modelo = joblib.load(ruta_modelo)
        features = joblib.load(ruta_features)

        print(f"\n{'=' * 70}")
        print(f"  {nombre} ({len(features)} features)")
        print(f"{'=' * 70}")

        df_imp = _analizar_modelo(nombre, modelo, features)
        if df_imp is None:
            continue

        resultados_todos.append(df_imp)

        # Ranking
        print(f"\n{'#':<4} {'Feature':<35} {'%':<8} {'% Acum':<8}")
        print("-" * 60)
        for i, (_, row) in enumerate(df_imp.iterrows(), 1):
            marker = " *" if row['Porcentaje'] < 1.0 else ""
            print(f"{i:<4} {row['Feature']:<35} {row['Porcentaje']:<8.2f} {row['Porcentaje_Acumulado']:<8.1f}{marker}")

        # Top features (50% acumulado)
        top_50 = df_imp[df_imp['Porcentaje_Acumulado'] <= 50]
        if len(top_50) == 0:
            top_50 = df_imp.head(1)
        print(f"\n  Top {len(top_50)} features acumulan ~50% de importancia")

        # Features con poca importancia
        low = df_imp[df_imp['Porcentaje'] < 1.0]
        if len(low) > 0:
            print(f"  {len(low)} features aportan < 1% cada una (total: {low['Porcentaje'].sum():.1f}%)")
            for _, row in low.iterrows():
                print(f"    - {row['Feature']} ({row['Porcentaje']:.2f}%)")

    # ========================================================================
    # ANÁLISIS CRUZADO: features de baja importancia en TODOS los modelos
    # ========================================================================
    if resultados_todos:
        print(f"\n{'=' * 70}")
        print("  ANÁLISIS CRUZADO — Features de baja importancia")
        print(f"{'=' * 70}")

        df_all = pd.concat(resultados_todos, ignore_index=True)

        # Features que aparecen en al menos un modelo con < 1%
        low_features = df_all[df_all['Porcentaje'] < 1.0].groupby('Feature').agg(
            modelos_low=('Modelo', 'count'),
            modelos_total=('Modelo', lambda x: len(x)),
            pct_promedio=('Porcentaje', 'mean'),
        ).sort_values('pct_promedio')

        # Features < 1% en TODOS los modelos donde aparecen
        candidatas_eliminar = low_features[low_features['modelos_low'] == low_features['modelos_total']]

        if len(candidatas_eliminar) > 0:
            print(f"\n  {len(candidatas_eliminar)} features son < 1% en TODOS los modelos donde aparecen:")
            for feat, row in candidatas_eliminar.iterrows():
                print(f"    - {feat} (promedio: {row['pct_promedio']:.2f}%, en {int(row['modelos_total'])} modelo(s))")
            print("\n  Estas features son candidatas a eliminarse del entrenamiento.")
        else:
            print("\n  No hay features < 1% en todos sus modelos.")

        # Redundancia (correlación) — solo para modelo principal
        print(f"\n{'=' * 70}")
        print("  FEATURES REDUNDANTES (Correlación > 0.85)")
        print(f"{'=' * 70}")

        from config import ARCHIVO_FEATURES
        if os.path.exists(ARCHIVO_FEATURES):
            df_data = pd.read_csv(ARCHIVO_FEATURES)
            # Usar features del modelo sin cuotas (más amplio)
            feats_vb = joblib.load(os.path.join(RUTA_MODELOS, 'features_value_betting.pkl')) if os.path.exists(os.path.join(RUTA_MODELOS, 'features_value_betting.pkl')) else []
            feats_ok = [f for f in feats_vb if f in df_data.columns]
            if feats_ok:
                X = df_data[feats_ok].apply(pd.to_numeric, errors='coerce').fillna(0)
                corr = X.corr().abs()
                pares = []
                cols = list(corr.columns)
                for ci in range(len(cols)):
                    for cj in range(ci + 1, len(cols)):
                        if corr.iloc[ci, cj] > 0.85:
                            pares.append((cols[ci], cols[cj], corr.iloc[ci, cj]))
                if pares:
                    pares.sort(key=lambda x: -x[2])
                    print(f"\n  {len(pares)} pares con correlación > 0.85:")
                    for f1, f2, c in pares[:20]:
                        print(f"    {f1:<30} <-> {f2:<30} ({c:.2f})")
                    if len(pares) > 20:
                        print(f"    ... y {len(pares) - 20} pares más")
                else:
                    print("\n  No hay features altamente correlacionadas.")

        # Guardar CSV consolidado
        csv_path = os.path.join(RUTA_MODELOS, 'feature_importance_analysis.csv')
        df_all.to_csv(csv_path, index=False)
        print(f"\n  Guardado: {csv_path}")

        # Gráfico por modelo
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        for idx, df_imp in enumerate(resultados_todos[:4]):
            ax = axes[idx]
            nombre_modelo = df_imp['Modelo'].iloc[0]
            top_15 = df_imp.head(15)
            colors = ['#e74c3c' if p < 1.0 else '#2ecc71' if p > 5.0 else '#3498db'
                       for p in top_15['Porcentaje']]
            ax.barh(range(len(top_15)), list(top_15['Porcentaje']), color=colors)
            ax.set_yticks(range(len(top_15)))
            ax.set_yticklabels(list(top_15['Feature']), fontsize=8)
            ax.set_xlabel('Importancia (%)')
            ax.set_title(nombre_modelo, fontweight='bold')
            ax.invert_yaxis()

        for idx in range(len(resultados_todos), 4):
            axes[idx].set_visible(False)

        plt.suptitle('Importancia de Features por Modelo\n(Verde >5% | Azul 1-5% | Rojo <1%)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        img_path = os.path.join(RUTA_MODELOS, 'feature_importance_analysis.png')
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Guardado: {img_path}")

    print(f"\n{'=' * 70}")
    print("   ANÁLISIS COMPLETADO")
    print(f"{'=' * 70}")
