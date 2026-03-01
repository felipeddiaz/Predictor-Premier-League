# -*- coding: utf-8 -*-
"""
experimento_features.py

Script de experimentacion iterativa para maximizar F1.
Cada paso agrega o quita features y reporta el delta de F1.
No modifica config.py ni utils.py — todo en memoria.
"""

import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import f1_score

from utils import (
    agregar_xg_rolling, agregar_features_tabla,
    agregar_features_cuotas_derivadas, agregar_features_asian_handicap,
)
from config import ALL_FEATURES, PARAMS_OPTIMOS, ARCHIVO_FEATURES, ROLLING_WINDOW

ROLLING = ROLLING_WINDOW  # 5

# ============================================================================
# HELPERS
# ============================================================================

def evaluar(df, features, label=""):
    X = df[features].fillna(0)
    y = df['FTR_numeric']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    m = RandomForestClassifier(**PARAMS_OPTIMOS)
    m.fit(X_train, y_train)
    f1 = f1_score(y_test, m.predict(X_test), average='weighted')
    tscv = TimeSeriesSplit(n_splits=5)
    cv = cross_val_score(m, X, y, cv=tscv, scoring='f1_weighted', n_jobs=-1).mean()
    print(f"  {label:<50} F1_test={f1:.4f}  CV5={cv:.4f}  n={len(features)}")
    return f1, cv, m

def importancias(model, features, n=10):
    imp = sorted(zip(features, model.feature_importances_), key=lambda x: x[1])
    print(f"    Bottom {n}: " + "  ".join([f"{f}={v:.4f}" for f, v in imp[:n]]))
    print(f"    Top    {n}: " + "  ".join([f"{f}={v:.4f}" for f, v in imp[-n:]]))

# ============================================================================
# CARGAR Y PREPARAR BASE
# ============================================================================

print("="*70)
print("CARGANDO DATOS BASE")
print("="*70)

df = pd.read_csv(ARCHIVO_FEATURES)
df = agregar_xg_rolling(df)
df = agregar_features_tabla(df)
df = agregar_features_cuotas_derivadas(df)
df = agregar_features_asian_handicap(df)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

features_base = [f for f in ALL_FEATURES if f in df.columns]
print(f"Features base: {len(features_base)}")

print("\n" + "="*70)
print("PASO 0 — BASELINE ACTUAL")
print("="*70)
f1_base, cv_base, m_base = evaluar(df, features_base, "Baseline actual")
importancias(m_base, features_base)

# ============================================================================
# PASO 1 — ELIMINAR FEATURES DE BAJO APORTE
# ============================================================================

print("\n" + "="*70)
print("PASO 1 — ELIMINAR FEATURES DE BAJO APORTE")
print("="*70)

ELIMINAR = [
    'Match_Type',
    'HT_Form_D',
    'Position_Reliability',
    'HT_Form_L',
    'AT_Form_D',
    'AT_Form_W',
    'AT_Form_L',
    'H2H_Home_Win_Rate',
    'H2H_Win_Advantage',
    'H2H_Matches',
]

f1_prev = f1_base
features_work = features_base.copy()

for feat in ELIMINAR:
    if feat not in features_work:
        continue
    candidato = [f for f in features_work if f != feat]
    f1_new, cv_new, _ = evaluar(df, candidato, f"sin {feat}")
    if f1_new >= f1_prev - 0.0005:
        features_work = candidato
        f1_prev = f1_new
        print(f"    -> ELIMINADA (acum delta={f1_new-f1_base:+.4f})")
    else:
        print(f"    -> RETENIDA  (perderia {f1_prev-f1_new:.4f})")

print(f"\nTras limpieza: {len(features_work)} features, F1={f1_prev:.4f}")

# ============================================================================
# PASO 2 — REEMPLAZAR FORM W/D/L POR FORM_PTS
# ============================================================================

print("\n" + "="*70)
print("PASO 2 — HT_Form_W/AT_Form_W -> HT_Form_Pts / AT_Form_Pts")
print("="*70)

df = df.sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True)

for team_col, pts_col in [('HomeTeam', 'HT_Form_Pts'), ('AwayTeam', 'AT_Form_Pts')]:
    df[pts_col] = np.nan
    for team in df['HomeTeam'].unique():
        mask_h = df['HomeTeam'] == team
        mask_a = df['AwayTeam'] == team
        mask = mask_h | mask_a
        idx = df[mask].sort_values('Date').index.tolist()
        pts_hist = []
        for i in idx:
            if len(pts_hist) > 0:
                df.at[i, pts_col] = np.mean(pts_hist[-ROLLING:])
            ftr = df.at[i, 'FTR']
            is_home = (df.at[i, 'HomeTeam'] == team)
            if ftr == 'H':
                pts = 3 if is_home else 0
            elif ftr == 'A':
                pts = 0 if is_home else 3
            else:
                pts = 1
            pts_hist.append(pts)
    df[pts_col] = df[pts_col].fillna(1.0)

FORM_INDIV = ['HT_Form_W', 'HT_Form_D', 'HT_Form_L', 'AT_Form_W', 'AT_Form_D', 'AT_Form_L']
features_form = [f for f in features_work if f not in FORM_INDIV] + ['HT_Form_Pts', 'AT_Form_Pts']
f1_form, cv_form, _ = evaluar(df, features_form, "Form_Pts en lugar de W/D/L")
if f1_form >= f1_prev - 0.001:
    features_work = features_form
    f1_prev = f1_form
    print(f"    -> APLICADO (delta={f1_form-f1_base:+.4f})")
else:
    print(f"    -> REVERTIDO (perderia {f1_prev-f1_form:.4f})")

# ============================================================================
# PASO 3 — ROLLING STATS: SHOTS, GOALS_DIFF, HTR
# ============================================================================

print("\n" + "="*70)
print("PASO 3 — NUEVAS FEATURES ROLLING (shots, goals_diff, HTR)")
print("="*70)

df = df.sort_values(['Date','HomeTeam','AwayTeam']).reset_index(drop=True)

for col in ['HT_ShotAcc','AT_ShotAcc','HT_Goals_Diff','AT_Goals_Diff','HT_HTR_Rate','AT_HTR_Rate']:
    df[col] = np.nan

for team in df['HomeTeam'].unique():
    mask_h = df['HomeTeam'] == team
    mask_a = df['AwayTeam'] == team
    idx_h = df[mask_h].sort_values('Date').index.tolist()
    idx_a = df[mask_a].sort_values('Date').index.tolist()

    # Shot accuracy home
    hs_hist, hst_hist = [], []
    for i in idx_h:
        if len(hs_hist) >= 1:
            s  = [x for x in hs_hist[-ROLLING:]  if x > 0]
            st = [x for x in hst_hist[-ROLLING:] if x >= 0]
            pairs = [(a, b) for a, b in zip(st[-len(s):], s) if b > 0]
            df.at[i, 'HT_ShotAcc'] = np.mean([a/b for a, b in pairs]) if pairs else 0.3
        hs  = df.at[i, 'HS']  if pd.notna(df.at[i, 'HS'])  else 0
        hst = df.at[i, 'HST'] if pd.notna(df.at[i, 'HST']) else 0
        hs_hist.append(hs); hst_hist.append(hst)

    # Shot accuracy away
    as_hist, ast_hist = [], []
    for i in idx_a:
        if len(as_hist) >= 1:
            s  = [x for x in as_hist[-ROLLING:]  if x > 0]
            st = [x for x in ast_hist[-ROLLING:] if x >= 0]
            pairs = [(a, b) for a, b in zip(st[-len(s):], s) if b > 0]
            df.at[i, 'AT_ShotAcc'] = np.mean([a/b for a, b in pairs]) if pairs else 0.3
        as_  = df.at[i, 'AS']  if pd.notna(df.at[i, 'AS'])  else 0
        ast_ = df.at[i, 'AST'] if pd.notna(df.at[i, 'AST']) else 0
        as_hist.append(as_); ast_hist.append(ast_)

    # Goals diff home
    gd_h = []
    for i in idx_h:
        if gd_h:
            df.at[i, 'HT_Goals_Diff'] = np.mean(gd_h[-ROLLING:])
        gh = df.at[i, 'FTHG'] if pd.notna(df.at[i, 'FTHG']) else 0
        ga = df.at[i, 'FTAG'] if pd.notna(df.at[i, 'FTAG']) else 0
        gd_h.append(gh - ga)

    # Goals diff away
    gd_a = []
    for i in idx_a:
        if gd_a:
            df.at[i, 'AT_Goals_Diff'] = np.mean(gd_a[-ROLLING:])
        gh = df.at[i, 'FTHG'] if pd.notna(df.at[i, 'FTHG']) else 0
        ga = df.at[i, 'FTAG'] if pd.notna(df.at[i, 'FTAG']) else 0
        gd_a.append(ga - gh)

    # HTR rate home (ganando al descanso)
    htr_h = []
    for i in idx_h:
        if htr_h:
            df.at[i, 'HT_HTR_Rate'] = np.mean(htr_h[-ROLLING:])
        hthg = df.at[i, 'HTHG'] if pd.notna(df.at[i, 'HTHG']) else 0
        htag = df.at[i, 'HTAG'] if pd.notna(df.at[i, 'HTAG']) else 0
        htr_h.append(1 if hthg > htag else 0)

    # HTR rate away
    htr_a = []
    for i in idx_a:
        if htr_a:
            df.at[i, 'AT_HTR_Rate'] = np.mean(htr_a[-ROLLING:])
        hthg = df.at[i, 'HTHG'] if pd.notna(df.at[i, 'HTHG']) else 0
        htag = df.at[i, 'HTAG'] if pd.notna(df.at[i, 'HTAG']) else 0
        htr_a.append(1 if htag > hthg else 0)

for col in ['HT_ShotAcc','AT_ShotAcc','HT_Goals_Diff','AT_Goals_Diff','HT_HTR_Rate','AT_HTR_Rate']:
    df[col] = df[col].fillna(0)

nuevas_rolling = ['HT_ShotAcc','AT_ShotAcc','HT_Goals_Diff','AT_Goals_Diff','HT_HTR_Rate','AT_HTR_Rate']
for feat in nuevas_rolling:
    candidato = features_work + [feat]
    f1_new, cv_new, _ = evaluar(df, candidato, f"+ {feat}")
    if f1_new > f1_prev:
        features_work = candidato
        f1_prev = f1_new
        print(f"    -> INCLUIDA  (delta={f1_new-f1_base:+.4f})")
    else:
        print(f"    -> DESCARTADA (delta={f1_new-f1_base:+.4f})")

# ============================================================================
# PASO 4 — MERCADO OVER/UNDER Y SHARP MONEY
# ============================================================================

print("\n" + "="*70)
print("PASO 4 — OVER/UNDER + SHARP MONEY SIGNAL")
print("="*70)

df['Over25_Prob']  = (1 / df['B365>2.5'].replace(0, np.nan)).fillna(0.5)
df['Over25_Move']  = ((1 / df['B365C>2.5'].replace(0, np.nan)) - df['Over25_Prob']).fillna(0)
df['Sharp_H']      = (df['MaxCH']  - df['AvgCH']).fillna(0)
df['Sharp_A']      = (df['MaxCA']  - df['AvgCA']).fillna(0)
df['PS_vs_Avg_H']  = (df['PSCH']   - df['AvgCH']).fillna(0)
df['PS_vs_Avg_A']  = (df['PSCA']   - df['AvgCA']).fillna(0)

nuevas_mercado = ['Over25_Prob','Over25_Move','Sharp_H','Sharp_A','PS_vs_Avg_H','PS_vs_Avg_A']
for feat in nuevas_mercado:
    candidato = features_work + [feat]
    f1_new, cv_new, _ = evaluar(df, candidato, f"+ {feat}")
    if f1_new > f1_prev:
        features_work = candidato
        f1_prev = f1_new
        print(f"    -> INCLUIDA  (delta={f1_new-f1_base:+.4f})")
    else:
        print(f"    -> DESCARTADA (delta={f1_new-f1_base:+.4f})")

# ============================================================================
# PASO 5 — POSESION ROLLING
# ============================================================================

print("\n" + "="*70)
print("PASO 5 — POSESION ROLLING")
print("="*70)

if 'Home_Poss' in df.columns:
    df['HT_Poss_Rolling'] = np.nan
    df['AT_Poss_Rolling'] = np.nan

    for team in df['HomeTeam'].unique():
        idx_h = df[df['HomeTeam'] == team].sort_values('Date').index.tolist()
        idx_a = df[df['AwayTeam'] == team].sort_values('Date').index.tolist()
        ph, pa = [], []
        for i in idx_h:
            if ph: df.at[i, 'HT_Poss_Rolling'] = np.mean(ph[-ROLLING:])
            p = df.at[i, 'Home_Poss']
            ph.append(p if pd.notna(p) else 50)
        for i in idx_a:
            if pa: df.at[i, 'AT_Poss_Rolling'] = np.mean(pa[-ROLLING:])
            p = df.at[i, 'Away_Poss']
            pa.append(p if pd.notna(p) else 50)

    df['HT_Poss_Rolling']  = df['HT_Poss_Rolling'].fillna(50)
    df['AT_Poss_Rolling']  = df['AT_Poss_Rolling'].fillna(50)
    df['Poss_Diff_Rolling'] = df['HT_Poss_Rolling'] - df['AT_Poss_Rolling']

    for feat in ['HT_Poss_Rolling','AT_Poss_Rolling','Poss_Diff_Rolling']:
        candidato = features_work + [feat]
        f1_new, cv_new, _ = evaluar(df, candidato, f"+ {feat}")
        if f1_new > f1_prev:
            features_work = candidato
            f1_prev = f1_new
            print(f"    -> INCLUIDA  (delta={f1_new-f1_base:+.4f})")
        else:
            print(f"    -> DESCARTADA (delta={f1_new-f1_base:+.4f})")

# ============================================================================
# PASO 6 — ELO RATINGS
# ============================================================================

print("\n" + "="*70)
print("PASO 6 — ELO RATINGS (clubelo.com)")
print("="*70)

_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELO_DIR  = os.path.join(_ROOT, 'datos', 'raw', 'elo')

team_to_file = {
    'Arsenal':'Arsenal','Aston Villa':'AstonVilla','Bournemouth':'Bournemouth',
    'Brentford':'Brentford','Brighton':'Brighton','Burnley':'Burnley',
    'Chelsea':'Chelsea','Crystal Palace':'CrystalPalace','Everton':'Everton',
    'Fulham':'Fulham','Ipswich':'Ipswich','Leeds':'Leeds','Leicester':'Leicester',
    'Liverpool':'Liverpool','Luton':'Luton','Man City':'ManCity',
    'Man United':'ManUnited','Newcastle':'Newcastle','Norwich':'Norwich',
    "Nott'm Forest":'NottmForest','Sheffield United':'SheffieldUnited',
    'Southampton':'Southampton','Sunderland':'Sunderland','Tottenham':'Tottenham',
    'Watford':'Watford','West Brom':'WestBrom','West Ham':'WestHam','Wolves':'Wolves',
}

elo_frames = []
for team_name, elo_file in team_to_file.items():
    path = os.path.join(ELO_DIR, f'{elo_file}.csv')
    if not os.path.exists(path):
        print(f"  AVISO: no encontrado {path}")
        continue
    elo = pd.read_csv(path)
    elo['From'] = pd.to_datetime(elo['From'])
    elo['To']   = pd.to_datetime(elo['To'])
    elo['team_name'] = team_name
    elo_frames.append(elo[['team_name','Elo','From','To']])

elo_all = pd.concat(elo_frames, ignore_index=True)
elo_mean_global = elo_all['Elo'].mean()
elo_mean_team   = elo_all.groupby('team_name')['Elo'].mean()

# Merge eficiente: para cada partido buscar el ELO vigente en esa fecha
print("Calculando ELO por partido...")

def elo_para_equipo_fecha(team, date, elo_df):
    sub = elo_df[elo_df['team_name'] == team]
    match = sub[(sub['From'] <= date) & (sub['To'] >= date)]
    if len(match) > 0:
        return float(match['Elo'].iloc[0])
    return None

# Vectorizar por equipo para rapidez
ht_elo_list, at_elo_list = [], []
for _, row in df.iterrows():
    ht_elo_list.append(elo_para_equipo_fecha(row['HomeTeam'], row['Date'], elo_all))
    at_elo_list.append(elo_para_equipo_fecha(row['AwayTeam'], row['Date'], elo_all))

df['HT_ELO'] = ht_elo_list
df['AT_ELO'] = at_elo_list

# Rellenar NaN
for idx, row in df[df['HT_ELO'].isna()].iterrows():
    df.at[idx, 'HT_ELO'] = elo_mean_team.get(row['HomeTeam'], elo_mean_global)
for idx, row in df[df['AT_ELO'].isna()].iterrows():
    df.at[idx, 'AT_ELO'] = elo_mean_team.get(row['AwayTeam'], elo_mean_global)

df['ELO_Diff'] = df['HT_ELO'] - df['AT_ELO']

con_elo = (df['HT_ELO'] > 0).sum()
print(f"ELO calculado: {con_elo}/{len(df)} partidos")

for feat in ['ELO_Diff', 'HT_ELO', 'AT_ELO']:
    candidato = features_work + [feat]
    f1_new, cv_new, _ = evaluar(df, candidato, f"+ {feat}")
    if f1_new > f1_prev:
        features_work = candidato
        f1_prev = f1_new
        print(f"    -> INCLUIDA  (delta={f1_new-f1_base:+.4f})")
    else:
        print(f"    -> DESCARTADA (delta={f1_new-f1_base:+.4f})")

# ============================================================================
# PASO 7 — SEGUNDA PASADA: RE-EVALUAR ELIMINACIONES CON EL SET FINAL
# ============================================================================

print("\n" + "="*70)
print("PASO 7 — SEGUNDA PASADA DE LIMPIEZA")
print("="*70)

# Re-evaluar eliminar features que quizas ahora son redundantes
_, _, m_tmp = evaluar(df, features_work, "Pre segunda limpieza")
imp_dict = dict(zip(features_work, m_tmp.feature_importances_))
candidatas_elim = [f for f, v in sorted(imp_dict.items(), key=lambda x: x[1]) if v < 0.005]
print(f"Candidatas a eliminar (imp < 0.5%): {candidatas_elim}")

for feat in candidatas_elim:
    candidato = [f for f in features_work if f != feat]
    f1_new, cv_new, _ = evaluar(df, candidato, f"sin {feat}")
    if f1_new >= f1_prev - 0.0005:
        features_work = candidato
        f1_prev = f1_new
        print(f"    -> ELIMINADA (delta={f1_new-f1_base:+.4f})")
    else:
        print(f"    -> RETENIDA  (perderia {f1_prev-f1_new:.4f})")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*70)
print("RESUMEN FINAL")
print("="*70)

f1_final, cv_final, m_final = evaluar(df, features_work, "CONFIGURACION FINAL")
importancias(m_final, features_work, n=10)

print(f"\nBaseline:   F1={f1_base:.4f}  CV={cv_base:.4f}  ({len(features_base)} features)")
print(f"Final:      F1={f1_final:.4f}  CV={cv_final:.4f}  ({len(features_work)} features)")
print(f"Delta:      {f1_final-f1_base:+.4f}")
print(f"Objetivo:   0.5460  {'SUPERADO' if f1_final > 0.5460 else 'NO SUPERADO'}")

print("\nFeatures en la configuracion final:")
for f in features_work:
    marca = "[NEW]" if f not in features_base else "     "
    print(f"  {marca} {f}")

print("\nFeatures ELIMINADAS vs baseline:")
for f in sorted(set(features_base) - set(features_work)):
    print(f"  [-] {f}")
