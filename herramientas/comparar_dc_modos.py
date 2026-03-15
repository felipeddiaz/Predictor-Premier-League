# -*- coding: utf-8 -*-
"""
Comparacion walk-forward de los 3 modos de Doble Oportunidad:
  - desactivado: solo 1X2
  - fallback: DC cuando 1X2 no pasa (original)
  - inteligente: DC basado en diagnostico_dc() (nuevo)

Ejecutar:
    python -m herramientas.comparar_dc_modos
"""

import pandas as pd
import numpy as np
import warnings

from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import (
    ARCHIVO_FEATURES,
    PARAMS_XGB,
    PARAMS_XGB_VB,
    PESOS_XGB,
    ALL_FEATURES,
    FEATURES_ESTRUCTURALES,
    RANDOM_SEED,
    UMBRAL_EDGE_MINIMO,
    UMBRAL_EDGE_DC,
    MARGEN_DC_SINTETICO,
    CUOTA_MAXIMA,
    PROBABILIDAD_MINIMA,
)
from utils import (
    agregar_xg_rolling,
    agregar_features_tabla,
    agregar_features_cuotas_derivadas,
    agregar_features_asian_handicap,
    agregar_features_rolling_extra,
    agregar_features_multi_escala,
    agregar_features_ewm,
    agregar_features_forma_momentum,
    agregar_features_descanso,
    agregar_features_pinnacle_move,
    agregar_features_arbitro,
    agregar_features_elo,
    agregar_features_sor,
    agregar_features_interaccion,
)
from core.sistema_expected_value import eliminar_vig, diagnostico_dc

warnings.filterwarnings('ignore')


def _asignar_temporada(dates):
    return dates.apply(
        lambda d: f"{d.year}-{str(d.year+1)[-2:]}" if d.month >= 8
        else f"{d.year-1}-{str(d.year)[-2:]}"
    )


def cargar_y_preparar():
    df = pd.read_csv(ARCHIVO_FEATURES)
    df['Date'] = pd.to_datetime(df['Date'])

    # FE compartido (con y sin cuotas)
    df = agregar_xg_rolling(df)
    df = agregar_features_tabla(df)
    df = agregar_features_multi_escala(df)
    df = agregar_features_ewm(df)
    df = agregar_features_forma_momentum(df)
    df = agregar_features_descanso(df)
    df = agregar_features_arbitro(df)
    df = agregar_features_elo(df)
    df = agregar_features_sor(df)
    df = agregar_features_interaccion(df)

    # FE solo para modelo con cuotas
    df = agregar_features_cuotas_derivadas(df)
    df = agregar_features_asian_handicap(df)
    df = agregar_features_rolling_extra(df)
    df = agregar_features_pinnacle_move(df)

    df['_Season'] = _asignar_temporada(df['Date'])
    features_con = [f for f in ALL_FEATURES if f in df.columns]
    features_sin = [f for f in FEATURES_ESTRUCTURALES if f in df.columns]
    return df, features_con, features_sin


def simular_temporada(y_test, probs, cuotas_df, modo_dc, umbral_dc_override=None):
    """Simula una temporada con flat stakes en un modo DC dado."""
    umbral_dc_eff = umbral_dc_override if umbral_dc_override is not None else UMBRAL_EDGE_DC
    cuotas = cuotas_df[['B365H', 'B365D', 'B365A']].values
    _clase_pierde = {'1X': 2, 'X2': 0, '12': 1}

    apostado = 0.0
    ganancia = 0.0
    n_1x2 = 0
    n_dc = 0
    detalle = []  # Para analisis por apuesta

    for i in range(len(y_test)):
        cuota_h, cuota_d, cuota_a = cuotas[i]
        if np.any(np.isnan([cuota_h, cuota_d, cuota_a])) or np.any(cuotas[i] <= 1.0):
            continue

        prob_mercado = 1.0 / cuotas[i]
        fair = prob_mercado / prob_mercado.sum()

        # --- Candidato 1X2 ---
        edges_1x2 = probs[i] - fair
        mejor_i = int(np.argmax(edges_1x2))
        mejor_edge_1x2 = edges_1x2[mejor_i]

        # Filtros 1X2
        tiene_1x2 = (
            mejor_edge_1x2 >= UMBRAL_EDGE_MINIMO
            and cuotas[i][mejor_i] <= CUOTA_MAXIMA
            and probs[i][mejor_i] >= PROBABILIDAD_MINIMA
        )

        # --- Candidato DC ---
        tiene_dc = False
        dc_mercado = None
        dc_edge = 0.0
        dc_cuota = 0.0
        dc_clase_pierde = -1

        if modo_dc != 'desactivado':
            if modo_dc == 'inteligente':
                diag = diagnostico_dc(probs[i])
                if diag['usar_dc'] and diag.get('confianza_dc') in ('alta', 'media'):
                    mercado = diag['mercado']
                    clase_p = _clase_pierde[mercado]

                    if mercado == '1X':
                        p_mod = probs[i][0] + probs[i][1]
                        p_fair = fair[0] + fair[1]
                    elif mercado == 'X2':
                        p_mod = probs[i][1] + probs[i][2]
                        p_fair = fair[1] + fair[2]
                    else:
                        p_mod = probs[i][0] + probs[i][2]
                        p_fair = fair[0] + fair[2]

                    edge_dc = p_mod - p_fair
                    if edge_dc >= umbral_dc_eff and p_fair > 0:
                        tiene_dc = True
                        dc_mercado = mercado
                        dc_edge = edge_dc
                        dc_cuota = (1.0 / p_fair) * MARGEN_DC_SINTETICO
                        dc_clase_pierde = clase_p

            elif modo_dc == 'fallback' and not tiene_1x2:
                dc_opciones = [
                    ('1X', probs[i][0] + probs[i][1], fair[0] + fair[1], 2),
                    ('X2', probs[i][1] + probs[i][2], fair[1] + fair[2], 0),
                    ('12', probs[i][0] + probs[i][2], fair[0] + fair[2], 1),
                ]
                for nombre, p_mod, p_fair, clase_p in dc_opciones:
                    edge_dc_cand = p_mod - p_fair
                    if edge_dc_cand >= umbral_dc_eff and edge_dc_cand > dc_edge and p_fair > 0:
                        tiene_dc = True
                        dc_mercado = nombre
                        dc_edge = edge_dc_cand
                        dc_cuota = (1.0 / p_fair) * MARGEN_DC_SINTETICO
                        dc_clase_pierde = clase_p

        # --- Decidir ---
        resultado_real = y_test.iloc[i]

        if tiene_1x2 and tiene_dc:
            if mejor_edge_1x2 >= dc_edge:
                apostado += 1.0; n_1x2 += 1
                gano = resultado_real == mejor_i
                ganancia += cuotas[i][mejor_i] - 1.0 if gano else -1.0
            else:
                apostado += 1.0; n_dc += 1
                gano = resultado_real != dc_clase_pierde
                ganancia += dc_cuota - 1.0 if gano else -1.0
        elif tiene_1x2:
            apostado += 1.0; n_1x2 += 1
            gano = resultado_real == mejor_i
            ganancia += cuotas[i][mejor_i] - 1.0 if gano else -1.0
        elif tiene_dc:
            apostado += 1.0; n_dc += 1
            gano = resultado_real != dc_clase_pierde
            ganancia += dc_cuota - 1.0 if gano else -1.0

    if apostado == 0:
        return {'roi': 0, 'n_total': 0, 'n_1x2': 0, 'n_dc': 0, 'ganancia': 0}

    return {
        'roi': ganancia / apostado,
        'n_total': int(apostado),
        'n_1x2': n_1x2,
        'n_dc': n_dc,
        'ganancia': ganancia,
    }


def run_comparacion(df, features, params_xgb, label, test_seasons, modos, dc_umbrales):
    """Corre walk-forward para un set de features dado."""
    print(f"\n{'#' * 90}")
    print(f"  MODELO: {label}")
    print(f"  Features: {len(features)}")
    print(f"{'#' * 90}")

    all_results = {m: [] for m in modos}

    for test_season in test_seasons:
        train_mask = df['_Season'] < test_season
        test_mask = df['_Season'] == test_season

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        df_train = df.loc[train_mask]
        df_test = df.loc[test_mask].copy()

        X_train = df_train[features]
        y_train = df_train['FTR_numeric']
        X_test = df_test[features]
        y_test = df_test['FTR_numeric'].reset_index(drop=True)

        sample_weights = compute_sample_weight(class_weight=PESOS_XGB, y=y_train)
        # Remove early_stopping_rounds for fit without eval_set
        params_fit = {k: v for k, v in params_xgb.items()
                      if k != 'early_stopping_rounds'}
        modelo = XGBClassifier(**params_fit)
        modelo.fit(X_train, y_train, sample_weight=sample_weights)
        probs = modelo.predict_proba(X_test)

        cuotas_df = df_test[['B365H', 'B365D', 'B365A']].reset_index(drop=True)

        for modo in modos:
            if modo.startswith('inteligente_'):
                umbral_dc_override = float(modo.split('_')[1])
                res = simular_temporada(y_test, probs, cuotas_df, 'inteligente',
                                        umbral_dc_override=umbral_dc_override)
            else:
                res = simular_temporada(y_test, probs, cuotas_df, modo)
            res['season'] = test_season
            all_results[modo].append(res)

    # Imprimir resumen
    print(f"\n  {'Modo':<25} {'Bets':>6} {'1X2':>5} {'DC':>5} {'ROI':>8} {'P&L':>10} {'Win%':>8}")
    print(f"  {'─' * 75}")
    for modo in modos:
        if modo.startswith('inteligente_'):
            u = modo.split('_')[1]
            nombre_corto = f'DC Intel >={float(u):.0%}'
        else:
            nombre_corto = {'desactivado': 'Solo 1X2', 'fallback': 'DC Fallback'}.get(modo, modo)

        total_bets = sum(r['n_total'] for r in all_results[modo])
        total_1x2 = sum(r['n_1x2'] for r in all_results[modo])
        total_dc = sum(r['n_dc'] for r in all_results[modo])
        total_ganancia = sum(r['ganancia'] for r in all_results[modo])
        roi = total_ganancia / total_bets if total_bets > 0 else 0
        n_positive = sum(1 for r in all_results[modo] if r['roi'] > 0 and r['n_total'] > 0)
        n_seasons = sum(1 for r in all_results[modo] if r['n_total'] > 0)
        win_seasons = f"{n_positive}/{n_seasons}"

        print(f"  {nombre_corto:<25} {total_bets:>6} {total_1x2:>5} {total_dc:>5} "
              f"{roi:>+7.1%} {total_ganancia:>+10.2f} {win_seasons:>8}")

    # Detalle por temporada para los mejores
    for modo in modos:
        if modo.startswith('inteligente_'):
            u = modo.split('_')[1]
            nombre = f'DC INTELIGENTE (edge>={float(u):.0%})'
        else:
            _nombres = {'desactivado': 'SOLO 1X2', 'fallback': 'DC FALLBACK'}
            nombre = _nombres.get(modo, modo)

        total_dc = sum(r['n_dc'] for r in all_results[modo])
        if total_dc == 0 and modo != 'desactivado' and modo != 'fallback':
            continue  # Skip modes with no DC bets in detail

        print(f"\n  {nombre}:")
        print(f"  {'Season':<12} {'Bets':>5} {'1X2':>5} {'DC':>5} {'ROI':>8} {'P&L':>8}")
        for r in all_results[modo]:
            roi_str = f"{r['roi']:+.1%}" if r['n_total'] > 0 else "N/A"
            print(f"  {r['season']:<12} {r['n_total']:>5} {r['n_1x2']:>5} "
                  f"{r['n_dc']:>5} {roi_str:>8} {r['ganancia']:>+8.2f}")

    return all_results


def main():
    print("Cargando datos y features...")
    df, features_con, features_sin = cargar_y_preparar()

    temporadas = sorted(df['_Season'].unique())
    test_seasons = [s for s in temporadas if s >= '2020-21' and s <= '2025-26']

    dc_umbrales = [0.03, 0.05, 0.08, 0.10, 0.15]
    modos = ['desactivado', 'fallback'] + [f'inteligente_{u}' for u in dc_umbrales]

    print("\n" + "=" * 90)
    print("COMPARACION WALK-FORWARD: MODOS DC + BARRIDO UMBRALES")
    print("=" * 90)
    print(f"  Edge 1X2: {UMBRAL_EDGE_MINIMO:.0%} | Edge DC config: {UMBRAL_EDGE_DC:.0%} | "
          f"Cuota max: {CUOTA_MAXIMA} | Prob min: {PROBABILIDAD_MINIMA:.0%}")
    print(f"  Umbrales DC inteligente: {dc_umbrales}")
    print(f"  Temporadas OOS: {', '.join(test_seasons)}")

    # --- Modelo CON cuotas ---
    run_comparacion(df, features_con, PARAMS_XGB,
                    "CON CUOTAS (ALL_FEATURES)", test_seasons, modos, dc_umbrales)

    # --- Modelo SIN cuotas ---
    run_comparacion(df, features_sin, PARAMS_XGB_VB,
                    "SIN CUOTAS (FEATURES_ESTRUCTURALES)", test_seasons, modos, dc_umbrales)

    print(f"\n{'=' * 90}")
    print("LISTO.")


if __name__ == '__main__':
    main()
