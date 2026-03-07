# -*- coding: utf-8 -*-
"""
Fase 5 — Simulacion y validacion final

5.1  Backtest walk-forward con ROI real (season-by-season)
5.2  Monte Carlo de bankroll (10,000 simulaciones)
5.3  Benchmark: apostar siempre al favorito del mercado
5.4  Registro prospectivo (predicciones 2025-26 ANTES de resultados)

Ejecutar:
    python -m herramientas.simulacion_final
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import log_loss, brier_score_loss
from xgboost import XGBClassifier

from config import (
    ARCHIVO_FEATURES,
    RUTA_MODELOS,
    PARAMS_XGB,
    PESOS_XGB,
    ALL_FEATURES,
    FEATURES_BASE,
    FEATURES_CUOTAS,
    FEATURES_CUOTAS_DERIVADAS,
    FEATURES_XG,
    FEATURES_XG_GLOBAL,
    FEATURES_MULTI_ESCALA,
    FEATURES_H2H,
    FEATURES_H2H_DERIVADAS,
    FEATURES_TABLA,
    FEATURES_ASIAN_HANDICAP,
    FEATURES_ROLLING_EXTRA,
    FEATURES_PINNACLE,
    FEATURES_REFEREE,
    FEATURES_FORMA_MOMENTUM,
    FEATURES_DESCANSO,
    RANDOM_SEED,
    UMBRAL_EDGE_MINIMO,
    KELLY_FRACTION,
    STAKE_MAXIMO_PCT,
    BANKROLL_DEFAULT,
    CUOTA_MAXIMA,
    PROBABILIDAD_MINIMA,
)
from utils import (
    agregar_features_tabla,
    agregar_features_cuotas_derivadas,
    agregar_features_asian_handicap,
    agregar_features_rolling_extra,
    agregar_features_multi_escala,
    agregar_features_forma_momentum,
    agregar_features_pinnacle_move,
    agregar_features_arbitro,
)
from core.sistema_expected_value import eliminar_vig, kelly_criterion

warnings.filterwarnings('ignore')

RESULTADO_NOMBRES = ['Home', 'Draw', 'Away']


# ============================================================================
# HELPERS
# ============================================================================

def _asignar_temporada(dates: pd.Series) -> pd.Series:
    """Asigna temporada tipo '2020-21' basandose en la fecha."""
    return dates.apply(
        lambda d: f"{d.year}-{str(d.year+1)[-2:]}" if d.month >= 8
        else f"{d.year-1}-{str(d.year)[-2:]}"
    )


def _brier_multiclase(y_true, probs):
    bs = 0.0
    for clase in range(3):
        y_bin = (y_true == clase).astype(int)
        bs += brier_score_loss(y_bin, probs[:, clase])
    return bs / 3


def cargar_y_preparar():
    """Carga el dataset y aplica feature engineering."""
    df = pd.read_csv(ARCHIVO_FEATURES)
    df['Date'] = pd.to_datetime(df['Date'])

    df = agregar_features_tabla(df)
    df = agregar_features_cuotas_derivadas(df)
    df = agregar_features_asian_handicap(df)
    df = agregar_features_rolling_extra(df)
    df = agregar_features_multi_escala(df)
    df = agregar_features_forma_momentum(df)
    df = agregar_features_pinnacle_move(df)
    df = agregar_features_arbitro(df)

    df['_Season'] = _asignar_temporada(df['Date'])

    features = [f for f in ALL_FEATURES if f in df.columns]
    return df, features


# ============================================================================
# 5.1 — BACKTEST WALK-FORWARD CON ROI REAL
# ============================================================================

def backtest_walkforward(df, features, bankroll_inicial=None,
                         edge_minimo=None, verbose=True):
    """
    Walk-forward season-by-season backtest con value betting real.

    Para cada temporada OOS (2020-21 en adelante):
      1. Entrena XGBoost en todas las temporadas anteriores
      2. Genera probabilidades para la temporada OOS
      3. Elimina vig de las cuotas
      4. Aplica filtros: edge > umbral, cuota < max, prob > min
      5. Aplica Kelly sizing con ajuste simultaneo (sqrt(N) por jornada)
      6. Calcula ROI, yield, max drawdown, Sharpe ratio

    Returns:
        dict con resultados por temporada y agregados
    """
    if bankroll_inicial is None:
        bankroll_inicial = BANKROLL_DEFAULT
    if edge_minimo is None:
        edge_minimo = UMBRAL_EDGE_MINIMO

    temporadas = sorted(df['_Season'].unique())
    test_seasons = [s for s in temporadas if s >= '2020-21' and s <= '2024-25']

    if verbose:
        print("=" * 80)
        print("5.1 — BACKTEST WALK-FORWARD CON ROI REAL")
        print("=" * 80)
        print(f"   Bankroll inicial: ${bankroll_inicial}")
        print(f"   Edge minimo: {edge_minimo:.0%}")
        print(f"   Kelly fraction: {KELLY_FRACTION}")
        print(f"   Temporadas OOS: {', '.join(test_seasons)}")

    resultados_temporada = []
    todas_apuestas = []

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
        y_test = df_test['FTR_numeric']

        # Entrenar XGBoost
        sample_weights = compute_sample_weight(class_weight=PESOS_XGB, y=y_train)
        modelo = XGBClassifier(**PARAMS_XGB)
        modelo.fit(X_train, y_train, sample_weight=sample_weights)

        probs = modelo.predict_proba(X_test)
        ll = log_loss(y_test, probs)
        bs = _brier_multiclase(y_test, probs)

        # Value betting con vig removal
        cuota_cols = ['B365H', 'B365D', 'B365A']
        apuestas_season = []
        bankroll = bankroll_inicial

        # Agrupar por fecha (jornada) para Kelly simultaneo
        df_test = df_test.reset_index(drop=True)
        df_test['_probs_0'] = probs[:, 0]
        df_test['_probs_1'] = probs[:, 1]
        df_test['_probs_2'] = probs[:, 2]

        for fecha, grupo in df_test.groupby('Date'):
            candidatos_jornada = []

            for idx_local, row in grupo.iterrows():
                cuota_h = row['B365H']
                cuota_d = row['B365D']
                cuota_a = row['B365A']

                if pd.isna(cuota_h) or pd.isna(cuota_d) or pd.isna(cuota_a):
                    continue
                if cuota_h <= 1.0 or cuota_d <= 1.0 or cuota_a <= 1.0:
                    continue

                # Eliminar vig
                fair_h, fair_d, fair_a = eliminar_vig(cuota_h, cuota_d, cuota_a)
                fair_probs = [fair_h, fair_d, fair_a]
                model_probs = [row['_probs_0'], row['_probs_1'], row['_probs_2']]
                cuotas = [cuota_h, cuota_d, cuota_a]
                resultado_real = int(row['FTR_numeric'])

                # Encontrar mejor apuesta (mayor edge positivo)
                edges = [model_probs[i] - fair_probs[i] for i in range(3)]
                mejor_i = int(np.argmax(edges))
                mejor_edge = edges[mejor_i]

                # Filtros de calidad
                if mejor_edge < edge_minimo:
                    continue
                if cuotas[mejor_i] > CUOTA_MAXIMA:
                    continue
                if model_probs[mejor_i] < PROBABILIDAD_MINIMA:
                    continue

                # Kelly sizing
                kelly_info = kelly_criterion(
                    model_probs[mejor_i], cuotas[mejor_i],
                    kelly_fraction=KELLY_FRACTION
                )

                candidatos_jornada.append({
                    'fecha': fecha,
                    'home': row.get('HomeTeam', ''),
                    'away': row.get('AwayTeam', ''),
                    'apuesta': RESULTADO_NOMBRES[mejor_i],
                    'prob_modelo': model_probs[mejor_i],
                    'prob_fair': fair_probs[mejor_i],
                    'cuota': cuotas[mejor_i],
                    'edge': mejor_edge,
                    'kelly_safe': kelly_info['kelly_safe'],
                    'resultado_real': resultado_real,
                    'gano': resultado_real == mejor_i,
                    'season': test_season,
                })

            # Ajustar Kelly para apuestas simultaneas
            n_simultaneas = len(candidatos_jornada)
            if n_simultaneas > 1:
                factor = 1.0 / np.sqrt(n_simultaneas)
                for c in candidatos_jornada:
                    c['kelly_safe'] *= factor

            # Ejecutar apuestas
            for c in candidatos_jornada:
                stake = bankroll * c['kelly_safe']
                stake = min(stake, bankroll * STAKE_MAXIMO_PCT)
                stake = max(stake, 0)

                if stake <= 0 or bankroll <= 0:
                    continue

                if c['gano']:
                    pnl = stake * (c['cuota'] - 1)
                else:
                    pnl = -stake

                bankroll += pnl

                c['stake'] = stake
                c['pnl'] = pnl
                c['bankroll_after'] = bankroll
                apuestas_season.append(c)

        # Metricas de la temporada
        if apuestas_season:
            total_staked = sum(a['stake'] for a in apuestas_season)
            total_pnl = sum(a['pnl'] for a in apuestas_season)
            roi = total_pnl / total_staked if total_staked > 0 else 0
            n_bets = len(apuestas_season)
            n_wins = sum(1 for a in apuestas_season if a['gano'])
            hit_rate = n_wins / n_bets

            # Max drawdown (sobre bankroll de la temporada)
            peak = bankroll_inicial
            max_dd = 0
            running = bankroll_inicial
            for a in sorted(apuestas_season, key=lambda x: x['fecha']):
                running += a['pnl']
                if running > peak:
                    peak = running
                dd = (peak - running) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

            # Sharpe ratio (sobre returns por apuesta)
            returns = [a['pnl'] / a['stake'] if a['stake'] > 0 else 0
                       for a in apuestas_season]
            avg_ret = np.mean(returns)
            std_ret = np.std(returns) if len(returns) > 1 else 1.0
            sharpe = avg_ret / std_ret if std_ret > 0 else 0

            # Yield = profit / total_staked
            yield_pct = total_pnl / total_staked * 100 if total_staked > 0 else 0
        else:
            total_staked = total_pnl = roi = n_bets = n_wins = hit_rate = 0
            max_dd = sharpe = yield_pct = 0

        season_result = {
            'season': test_season,
            'train_size': train_mask.sum(),
            'test_size': test_mask.sum(),
            'log_loss': ll,
            'brier_score': bs,
            'n_bets': n_bets,
            'n_wins': n_wins,
            'hit_rate': hit_rate,
            'total_staked': total_staked,
            'total_pnl': total_pnl,
            'roi': roi,
            'yield_pct': yield_pct,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'bankroll_final': bankroll,
        }
        resultados_temporada.append(season_result)
        todas_apuestas.extend(apuestas_season)

    # Imprimir resultados
    if verbose and resultados_temporada:
        print(f"\n{'Season':<12} {'Train':>6} {'Test':>5} "
              f"{'LL':>7} {'Brier':>7} {'Bets':>5} {'Wins':>5} "
              f"{'Hit%':>6} {'ROI':>8} {'Yield':>7} {'MaxDD':>7} {'Sharpe':>7}")
        print("-" * 100)

        for r in resultados_temporada:
            print(f"{r['season']:<12} {r['train_size']:>6} {r['test_size']:>5} "
                  f"{r['log_loss']:>7.4f} {r['brier_score']:>7.4f} "
                  f"{r['n_bets']:>5} {r['n_wins']:>5} "
                  f"{r['hit_rate']:>5.1%} {r['roi']:>+7.1%} "
                  f"{r['yield_pct']:>6.1f}% {r['max_drawdown']:>6.1%} "
                  f"{r['sharpe']:>7.3f}")

        # Agregados
        total_bets = sum(r['n_bets'] for r in resultados_temporada)
        total_wins = sum(r['n_wins'] for r in resultados_temporada)
        total_staked_all = sum(r['total_staked'] for r in resultados_temporada)
        total_pnl_all = sum(r['total_pnl'] for r in resultados_temporada)

        print("-" * 100)
        avg_ll = np.mean([r['log_loss'] for r in resultados_temporada])
        avg_bs = np.mean([r['brier_score'] for r in resultados_temporada])
        overall_roi = total_pnl_all / total_staked_all if total_staked_all > 0 else 0
        overall_yield = total_pnl_all / total_staked_all * 100 if total_staked_all > 0 else 0
        overall_hit = total_wins / total_bets if total_bets > 0 else 0

        # Aggregate: worst per-season drawdown (seasons have independent bankrolls)
        if todas_apuestas:
            max_dd_all = max(r['max_drawdown'] for r in resultados_temporada)

            returns_all = [a['pnl'] / a['stake'] if a['stake'] > 0 else 0
                           for a in todas_apuestas]
            sharpe_all = (np.mean(returns_all) / np.std(returns_all)
                          if np.std(returns_all) > 0 else 0)
        else:
            max_dd_all = sharpe_all = 0

        print(f"{'TOTAL':<12} {'':>6} {'':>5} "
              f"{avg_ll:>7.4f} {avg_bs:>7.4f} "
              f"{total_bets:>5} {total_wins:>5} "
              f"{overall_hit:>5.1%} {overall_roi:>+7.1%} "
              f"{overall_yield:>6.1f}% {max_dd_all:>6.1%} "
              f"{sharpe_all:>7.3f}")

        print(f"\n   P&L total: ${total_pnl_all:+.2f}")
        print(f"   Bankroll final: ${bankroll_inicial + total_pnl_all:.2f} "
              f"(inicio: ${bankroll_inicial})")

    return {
        'por_temporada': resultados_temporada,
        'apuestas': todas_apuestas,
        'bankroll_inicial': bankroll_inicial,
    }


# ============================================================================
# 5.2 — MONTE CARLO DE BANKROLL
# ============================================================================

def monte_carlo_bankroll(apuestas, bankroll_inicial=None, n_sims=10_000,
                         verbose=True):
    """
    Monte Carlo simulacion de bankroll.

    Toma las apuestas observadas del backtest y resamples con reemplazo
    para simular 10,000 trayectorias de bankroll.

    Calcula:
      - Probabilidad de ruina (bankroll <= 0)
      - Max drawdown al percentil 95
      - ROI con intervalo de confianza 95%
    """
    if bankroll_inicial is None:
        bankroll_inicial = BANKROLL_DEFAULT

    if not apuestas:
        print("   No hay apuestas para simular.")
        return None

    if verbose:
        print("\n" + "=" * 80)
        print("5.2 — MONTE CARLO DE BANKROLL")
        print("=" * 80)
        print(f"   Apuestas observadas: {len(apuestas)}")
        print(f"   Simulaciones: {n_sims:,}")
        print(f"   Bankroll inicial: ${bankroll_inicial}")

    # Extraer returns relativos (pnl/stake) de las apuestas reales
    returns = np.array([a['pnl'] / a['stake'] if a['stake'] > 0 else 0
                        for a in apuestas])
    stakes_pct = np.array([a['stake'] / bankroll_inicial for a in apuestas])
    avg_stake_pct = np.mean(stakes_pct)

    n_bets = len(apuestas)
    rng = np.random.RandomState(RANDOM_SEED)

    # Resultados de las simulaciones
    final_bankrolls = np.zeros(n_sims)
    max_drawdowns = np.zeros(n_sims)
    ruin_count = 0

    for sim in range(n_sims):
        # Resample con reemplazo: misma cantidad de apuestas
        indices = rng.randint(0, n_bets, size=n_bets)
        sim_returns = returns[indices]

        bankroll = bankroll_inicial
        peak = bankroll_inicial
        max_dd = 0

        for ret in sim_returns:
            stake = bankroll * avg_stake_pct
            stake = min(stake, bankroll * STAKE_MAXIMO_PCT)
            pnl = stake * ret
            bankroll += pnl

            if bankroll <= 0:
                ruin_count += 1
                bankroll = 0
                max_dd = 1.0
                break

            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak
            if dd > max_dd:
                max_dd = dd

        final_bankrolls[sim] = bankroll
        max_drawdowns[sim] = max_dd

    # Estadisticas
    prob_ruin = ruin_count / n_sims
    dd_95 = np.percentile(max_drawdowns, 95)
    dd_50 = np.percentile(max_drawdowns, 50)

    rois = (final_bankrolls - bankroll_inicial) / bankroll_inicial
    roi_mean = np.mean(rois)
    roi_ci_lower = np.percentile(rois, 2.5)
    roi_ci_upper = np.percentile(rois, 97.5)
    roi_median = np.median(rois)

    bankroll_median = np.median(final_bankrolls)
    bankroll_5th = np.percentile(final_bankrolls, 5)
    bankroll_95th = np.percentile(final_bankrolls, 95)

    if verbose:
        print(f"\n   RESULTADOS ({n_sims:,} simulaciones):")
        print(f"   {'─' * 50}")
        print(f"   Probabilidad de ruina:    {prob_ruin:.2%}")
        print(f"   Max drawdown (mediana):   {dd_50:.1%}")
        print(f"   Max drawdown (P95):       {dd_95:.1%}")
        print(f"   {'─' * 50}")
        print(f"   ROI medio:                {roi_mean:+.2%}")
        print(f"   ROI mediana:              {roi_median:+.2%}")
        print(f"   ROI IC 95%:               [{roi_ci_lower:+.2%}, {roi_ci_upper:+.2%}]")
        print(f"   {'─' * 50}")
        print(f"   Bankroll final (mediana):  ${bankroll_median:.0f}")
        print(f"   Bankroll final (P5-P95):   ${bankroll_5th:.0f} — ${bankroll_95th:.0f}")

        if prob_ruin > 0.05:
            print(f"\n   ALERTA: Probabilidad de ruina > 5%. Riesgo elevado.")
        elif prob_ruin > 0:
            print(f"\n   Riesgo de ruina presente pero bajo ({prob_ruin:.2%}).")
        else:
            print(f"\n   Riesgo de ruina: 0% en {n_sims:,} simulaciones.")

        if roi_ci_lower > 0:
            print(f"   IC 95% enteramente positivo -> evidencia de edge.")
        elif roi_ci_upper < 0:
            print(f"   IC 95% enteramente negativo -> NO hay edge.")
        else:
            print(f"   IC 95% cruza cero -> edge incierto.")

    return {
        'prob_ruin': prob_ruin,
        'max_dd_median': dd_50,
        'max_dd_p95': dd_95,
        'roi_mean': roi_mean,
        'roi_median': roi_median,
        'roi_ci_lower': roi_ci_lower,
        'roi_ci_upper': roi_ci_upper,
        'bankroll_median': bankroll_median,
        'bankroll_5th': bankroll_5th,
        'bankroll_95th': bankroll_95th,
        'final_bankrolls': final_bankrolls,
        'max_drawdowns': max_drawdowns,
    }


# ============================================================================
# 5.3 — BENCHMARK: APOSTAR SIEMPRE AL FAVORITO DEL MERCADO
# ============================================================================

def benchmark_favorito(df, features, bankroll_inicial=None, verbose=True):
    """
    Baseline: apostar siempre al favorito del mercado (cuota mas baja).

    Si el modelo no supera consistentemente esta estrategia trivial,
    no hay edge real.
    """
    if bankroll_inicial is None:
        bankroll_inicial = BANKROLL_DEFAULT

    if verbose:
        print("\n" + "=" * 80)
        print("5.3 — BENCHMARK: APOSTAR SIEMPRE AL FAVORITO")
        print("=" * 80)

    temporadas = sorted(df['_Season'].unique())
    test_seasons = [s for s in temporadas if s >= '2020-21' and s <= '2024-25']

    resultados_fav = []
    todas_apuestas_fav = []

    for test_season in test_seasons:
        test_mask = df['_Season'] == test_season
        df_test = df.loc[test_mask].copy().reset_index(drop=True)

        apuestas_season = []
        bankroll = bankroll_inicial

        for _, row in df_test.iterrows():
            cuota_h = row['B365H']
            cuota_d = row['B365D']
            cuota_a = row['B365A']

            if pd.isna(cuota_h) or pd.isna(cuota_d) or pd.isna(cuota_a):
                continue
            if cuota_h <= 1.0 or cuota_d <= 1.0 or cuota_a <= 1.0:
                continue

            cuotas = [cuota_h, cuota_d, cuota_a]
            # Favorito = cuota mas baja
            fav_i = int(np.argmin(cuotas))
            cuota_fav = cuotas[fav_i]

            # Apuesta fija (flat bet): mismo % del bankroll
            stake = bankroll * 0.02  # 2% del bankroll (comparable)
            stake = min(stake, bankroll * STAKE_MAXIMO_PCT)

            resultado_real = int(row['FTR_numeric'])
            gano = resultado_real == fav_i

            if gano:
                pnl = stake * (cuota_fav - 1)
            else:
                pnl = -stake

            bankroll += pnl

            apuestas_season.append({
                'fecha': row['Date'],
                'home': row.get('HomeTeam', ''),
                'away': row.get('AwayTeam', ''),
                'apuesta': RESULTADO_NOMBRES[fav_i],
                'cuota': cuota_fav,
                'resultado_real': resultado_real,
                'gano': gano,
                'stake': stake,
                'pnl': pnl,
                'bankroll_after': bankroll,
                'season': test_season,
            })

        # Metricas
        if apuestas_season:
            total_staked = sum(a['stake'] for a in apuestas_season)
            total_pnl = sum(a['pnl'] for a in apuestas_season)
            roi = total_pnl / total_staked if total_staked > 0 else 0
            n_bets = len(apuestas_season)
            n_wins = sum(1 for a in apuestas_season if a['gano'])
            hit_rate = n_wins / n_bets
            yield_pct = total_pnl / total_staked * 100 if total_staked > 0 else 0
        else:
            total_staked = total_pnl = roi = n_bets = n_wins = 0
            hit_rate = yield_pct = 0

        resultados_fav.append({
            'season': test_season,
            'n_bets': n_bets,
            'n_wins': n_wins,
            'hit_rate': hit_rate,
            'roi': roi,
            'yield_pct': yield_pct,
            'total_pnl': total_pnl,
        })
        todas_apuestas_fav.extend(apuestas_season)

    if verbose and resultados_fav:
        print(f"\n{'Season':<12} {'Bets':>5} {'Wins':>5} {'Hit%':>6} "
              f"{'ROI':>8} {'Yield':>7} {'P&L':>10}")
        print("-" * 60)

        for r in resultados_fav:
            print(f"{r['season']:<12} {r['n_bets']:>5} {r['n_wins']:>5} "
                  f"{r['hit_rate']:>5.1%} {r['roi']:>+7.1%} "
                  f"{r['yield_pct']:>6.1f}% {r['total_pnl']:>+9.2f}")

        total_bets = sum(r['n_bets'] for r in resultados_fav)
        total_wins = sum(r['n_wins'] for r in resultados_fav)
        total_staked_all = sum(a['stake'] for a in todas_apuestas_fav)
        total_pnl_all = sum(r['total_pnl'] for r in resultados_fav)
        overall_roi = total_pnl_all / total_staked_all if total_staked_all > 0 else 0

        print("-" * 60)
        print(f"{'TOTAL':<12} {total_bets:>5} {total_wins:>5} "
              f"{total_wins/total_bets:>5.1%} {overall_roi:>+7.1%} "
              f"{overall_roi*100:>6.1f}% {total_pnl_all:>+9.2f}")

    return {
        'por_temporada': resultados_fav,
        'apuestas': todas_apuestas_fav,
    }


# ============================================================================
# 5.4 — REGISTRO PROSPECTIVO (2025-26)
# ============================================================================

def registro_prospectivo(df, features, verbose=True):
    """
    Registra predicciones para la temporada 2025-26 ANTES de que se jueguen.

    Entrena en todas las temporadas anteriores a 2025-26, genera
    predicciones para los partidos de 2025-26, y las guarda con timestamp.

    Para partidos ya jugados, calcula precision retrospectiva.
    Para partidos futuros, registra las predicciones como prospectivas.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("5.4 — REGISTRO PROSPECTIVO (2025-26)")
        print("=" * 80)

    train_mask = df['_Season'] < '2025-26'
    target_mask = df['_Season'] == '2025-26'

    if target_mask.sum() == 0:
        print("   No hay partidos de 2025-26 en el dataset.")
        return None

    df_train = df.loc[train_mask]
    df_target = df.loc[target_mask].copy().reset_index(drop=True)

    X_train = df_train[features]
    y_train = df_train['FTR_numeric']
    X_target = df_target[features]

    if verbose:
        print(f"   Entrenamiento: {len(df_train)} partidos (hasta 2024-25)")
        print(f"   Objetivo: {len(df_target)} partidos (2025-26)")

    # Entrenar modelo
    sample_weights = compute_sample_weight(class_weight=PESOS_XGB, y=y_train)
    modelo = XGBClassifier(**PARAMS_XGB)
    modelo.fit(X_train, y_train, sample_weight=sample_weights)

    probs = modelo.predict_proba(X_target)
    preds = modelo.predict(X_target)

    # Construir registro
    registros = []
    for i, (_, row) in enumerate(df_target.iterrows()):
        cuota_h = row.get('B365H', np.nan)
        cuota_d = row.get('B365D', np.nan)
        cuota_a = row.get('B365A', np.nan)

        # Determinar si hay value bet
        value_bet = None
        edge_value = 0
        if not pd.isna(cuota_h) and cuota_h > 1.0:
            fair_h, fair_d, fair_a = eliminar_vig(cuota_h, cuota_d, cuota_a)
            edges = [probs[i, 0] - fair_h, probs[i, 1] - fair_d, probs[i, 2] - fair_a]
            mejor_i = int(np.argmax(edges))
            if edges[mejor_i] >= UMBRAL_EDGE_MINIMO:
                value_bet = RESULTADO_NOMBRES[mejor_i]
                edge_value = edges[mejor_i]

        tiene_resultado = not pd.isna(row.get('FTR_numeric', np.nan))

        registro = {
            'timestamp_prediccion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fecha_partido': row['Date'],
            'home': row.get('HomeTeam', ''),
            'away': row.get('AwayTeam', ''),
            'prob_home': round(probs[i, 0], 4),
            'prob_draw': round(probs[i, 1], 4),
            'prob_away': round(probs[i, 2], 4),
            'prediccion': RESULTADO_NOMBRES[int(preds[i])],
            'B365H': cuota_h,
            'B365D': cuota_d,
            'B365A': cuota_a,
            'value_bet': value_bet if value_bet else '',
            'edge': round(edge_value, 4) if edge_value > 0 else '',
            'resultado_real': RESULTADO_NOMBRES[int(row['FTR_numeric'])] if tiene_resultado else '',
            'acierto': (int(preds[i]) == int(row['FTR_numeric'])) if tiene_resultado else '',
        }
        registros.append(registro)

    df_reg = pd.DataFrame(registros)

    # Guardar
    archivo = os.path.join(RUTA_MODELOS, 'registro_prospectivo_2025_26.csv')
    df_reg.to_csv(archivo, index=False)

    if verbose:
        # Partidos ya jugados
        jugados = df_reg[df_reg['resultado_real'] != '']
        no_jugados = df_reg[df_reg['resultado_real'] == '']

        print(f"\n   Partidos ya jugados: {len(jugados)}")
        print(f"   Partidos pendientes: {len(no_jugados)}")

        if len(jugados) > 0:
            aciertos = jugados['acierto'].sum()
            acc = aciertos / len(jugados)
            print(f"\n   Precision en partidos jugados: {int(aciertos)}/{len(jugados)} ({acc:.1%})")

            # Value bets jugadas
            vb_jugados = jugados[jugados['value_bet'] != '']
            if len(vb_jugados) > 0:
                # Calcular ROI de value bets
                vb_pnl = 0
                vb_count = 0
                for _, r in vb_jugados.iterrows():
                    resultado_map = {'Home': 0, 'Draw': 1, 'Away': 2}
                    cuotas_map = {'Home': r['B365H'], 'Draw': r['B365D'], 'Away': r['B365A']}
                    apuesta_cuota = cuotas_map.get(r['value_bet'], 0)
                    if r['resultado_real'] == r['value_bet']:
                        vb_pnl += apuesta_cuota - 1
                    else:
                        vb_pnl -= 1
                    vb_count += 1
                vb_roi = vb_pnl / vb_count if vb_count > 0 else 0
                print(f"   Value bets jugadas: {vb_count}, ROI: {vb_roi:+.1%}")

        print(f"\n   Guardado: {archivo}")

    return df_reg


# ============================================================================
# COMPARACION MODELO vs FAVORITO
# ============================================================================

def comparar_modelo_vs_favorito(resultado_modelo, resultado_favorito, verbose=True):
    """Compara los resultados del modelo vs la estrategia de apostar al favorito."""
    if not verbose:
        return

    print("\n" + "=" * 80)
    print("COMPARACION: MODELO vs APOSTAR AL FAVORITO")
    print("=" * 80)

    seasons_modelo = {r['season']: r for r in resultado_modelo['por_temporada']}
    seasons_fav = {r['season']: r for r in resultado_favorito['por_temporada']}

    print(f"\n{'Season':<12} {'Modelo ROI':>11} {'Favorito ROI':>13} {'Diferencia':>11} {'Mejor':>10}")
    print("-" * 60)

    modelo_wins = 0
    total_seasons = 0

    for season in sorted(set(seasons_modelo.keys()) & set(seasons_fav.keys())):
        m = seasons_modelo[season]
        f = seasons_fav[season]
        diff = m['roi'] - f['roi']
        mejor = "MODELO" if diff > 0 else "FAVORITO"

        print(f"{season:<12} {m['roi']:>+10.1%} {f['roi']:>+12.1%} "
              f"{diff:>+10.1%} {mejor:>10}")

        if diff > 0:
            modelo_wins += 1
        total_seasons += 1

    # Totales
    total_staked_modelo = sum(a['stake'] for a in resultado_modelo['apuestas'])
    total_pnl_modelo = sum(a['pnl'] for a in resultado_modelo['apuestas'])
    roi_modelo = total_pnl_modelo / total_staked_modelo if total_staked_modelo > 0 else 0

    total_staked_fav = sum(a['stake'] for a in resultado_favorito['apuestas'])
    total_pnl_fav = sum(a['pnl'] for a in resultado_favorito['apuestas'])
    roi_fav = total_pnl_fav / total_staked_fav if total_staked_fav > 0 else 0

    print("-" * 60)
    diff_total = roi_modelo - roi_fav
    print(f"{'TOTAL':<12} {roi_modelo:>+10.1%} {roi_fav:>+12.1%} "
          f"{diff_total:>+10.1%} {'MODELO' if diff_total > 0 else 'FAVORITO':>10}")

    print(f"\n   Modelo supera al favorito en {modelo_wins}/{total_seasons} temporadas")

    if modelo_wins >= total_seasons * 0.6 and diff_total > 0:
        print("   CONCLUSION: El modelo muestra edge sobre la estrategia de favorito.")
    elif diff_total > 0:
        print("   CONCLUSION: El modelo supera al favorito en agregado, pero no consistentemente.")
    else:
        print("   CONCLUSION: El modelo NO supera al favorito. No hay evidencia de edge real.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("   FASE 5 — SIMULACION Y VALIDACION FINAL")
    print("=" * 80)

    df, features = cargar_y_preparar()
    print(f"   Dataset: {len(df)} partidos, {len(features)} features")
    print(f"   Temporadas: {', '.join(sorted(df['_Season'].unique()))}")

    # 5.1 — Backtest walk-forward
    resultado_bt = backtest_walkforward(df, features)

    # 5.2 — Monte Carlo
    mc_result = monte_carlo_bankroll(resultado_bt['apuestas'],
                                      resultado_bt['bankroll_inicial'])

    # 5.3 — Benchmark vs favorito
    resultado_fav = benchmark_favorito(df, features)

    # Comparacion
    comparar_modelo_vs_favorito(resultado_bt, resultado_fav)

    # 5.4 — Registro prospectivo
    registro_prospectivo(df, features)

    print("\n" + "=" * 80)
    print("   FASE 5 COMPLETADA")
    print("=" * 80)


if __name__ == '__main__':
    main()
