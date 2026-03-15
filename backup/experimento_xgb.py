# -*- coding: utf-8 -*-
"""
experimento_xgb.py — Probar XGBoost + todas las features nuevas + RandomSearch rapido
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from utils import (
    agregar_xg_rolling, agregar_features_tabla,
    agregar_features_cuotas_derivadas, agregar_features_asian_handicap,
)
from config import ALL_FEATURES, PARAMS_OPTIMOS, PARAMS_XGB, PESOS_OPTIMOS, ARCHIVO_FEATURES, ROLLING_WINDOW

ROLLING = ROLLING_WINDOW

# ============================================================================
# CARGAR + FEATURES
# ============================================================================
print("Cargando datos...")
df = pd.read_csv(ARCHIVO_FEATURES)
df = agregar_xg_rolling(df)
df = agregar_features_tabla(df)
df = agregar_features_cuotas_derivadas(df)
df = agregar_features_asian_handicap(df)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

# Goals diff + HTR + ShotAcc rolling
for col in ['HT_Goals_Diff','AT_Goals_Diff','HT_HTR_Rate','AT_HTR_Rate','HT_ShotAcc','AT_ShotAcc']:
    df[col] = np.nan

for team in df['HomeTeam'].unique():
    idx_h = df[df['HomeTeam'] == team].sort_values('Date').index.tolist()
    idx_a = df[df['AwayTeam'] == team].sort_values('Date').index.tolist()
    gd_h, gd_a, htr_h, htr_a = [], [], [], []
    hs_h, hst_h, as_a, ast_a = [], [], [], []

    for i in idx_h:
        if gd_h:  df.at[i, 'HT_Goals_Diff'] = np.mean(gd_h[-ROLLING:])
        if htr_h: df.at[i, 'HT_HTR_Rate']   = np.mean(htr_h[-ROLLING:])
        if hs_h and any(x > 0 for x in hs_h[-ROLLING:]):
            pairs = [(b, a) for a, b in zip(hs_h[-ROLLING:], hst_h[-ROLLING:]) if b > 0]
            df.at[i, 'HT_ShotAcc'] = np.mean([a/b for a, b in pairs]) if pairs else 0.3
        gh  = df.at[i, 'FTHG'] if pd.notna(df.at[i, 'FTHG']) else 0
        ga  = df.at[i, 'FTAG'] if pd.notna(df.at[i, 'FTAG']) else 0
        gd_h.append(gh - ga)
        hthg = df.at[i, 'HTHG'] if pd.notna(df.at[i, 'HTHG']) else 0
        htag = df.at[i, 'HTAG'] if pd.notna(df.at[i, 'HTAG']) else 0
        htr_h.append(1 if hthg > htag else 0)
        hs  = df.at[i, 'HS']  if pd.notna(df.at[i, 'HS'])  else 0
        hst = df.at[i, 'HST'] if pd.notna(df.at[i, 'HST']) else 0
        hs_h.append(hs); hst_h.append(hst)

    for i in idx_a:
        if gd_a:  df.at[i, 'AT_Goals_Diff'] = np.mean(gd_a[-ROLLING:])
        if htr_a: df.at[i, 'AT_HTR_Rate']   = np.mean(htr_a[-ROLLING:])
        if as_a and any(x > 0 for x in as_a[-ROLLING:]):
            pairs = [(b, a) for a, b in zip(as_a[-ROLLING:], ast_a[-ROLLING:]) if b > 0]
            df.at[i, 'AT_ShotAcc'] = np.mean([a/b for a, b in pairs]) if pairs else 0.3
        gh  = df.at[i, 'FTHG'] if pd.notna(df.at[i, 'FTHG']) else 0
        ga  = df.at[i, 'FTAG'] if pd.notna(df.at[i, 'FTAG']) else 0
        gd_a.append(ga - gh)
        hthg = df.at[i, 'HTHG'] if pd.notna(df.at[i, 'HTHG']) else 0
        htag = df.at[i, 'HTAG'] if pd.notna(df.at[i, 'HTAG']) else 0
        htr_a.append(1 if htag > hthg else 0)
        asv = df.at[i, 'AS']  if pd.notna(df.at[i, 'AS'])  else 0
        ast = df.at[i, 'AST'] if pd.notna(df.at[i, 'AST']) else 0
        as_a.append(asv); ast_a.append(ast)

for col in ['HT_Goals_Diff','AT_Goals_Diff','HT_HTR_Rate','AT_HTR_Rate','HT_ShotAcc','AT_ShotAcc']:
    df[col] = df[col].fillna(0)

# Mercado over/under + sharp money
df['Over25_Prob'] = (1 / df['B365>2.5'].replace(0, np.nan)).fillna(0.5)
df['Over25_Move'] = ((1 / df['B365C>2.5'].replace(0, np.nan)) - df['Over25_Prob']).fillna(0)
df['Sharp_H']     = (df['MaxCH'] - df['AvgCH']).fillna(0)
df['Sharp_A']     = (df['MaxCA'] - df['AvgCA']).fillna(0)
df['PS_vs_Avg_H'] = (df['PSCH']  - df['AvgCH']).fillna(0)
df['PS_vs_Avg_A'] = (df['PSCA']  - df['AvgCA']).fillna(0)

# Posesion rolling
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
    df['HT_Poss_Rolling']   = df['HT_Poss_Rolling'].fillna(50)
    df['AT_Poss_Rolling']   = df['AT_Poss_Rolling'].fillna(50)
    df['Poss_Diff_Rolling'] = df['HT_Poss_Rolling'] - df['AT_Poss_Rolling']

# ELO
ELO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datos', 'raw', 'elo')
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
for tn, ef in team_to_file.items():
    path = os.path.join(ELO_DIR, f'{ef}.csv')
    if os.path.exists(path):
        e = pd.read_csv(path)
        e['From'] = pd.to_datetime(e['From'])
        e['To']   = pd.to_datetime(e['To'])
        e['team_name'] = tn
        elo_frames.append(e[['team_name','Elo','From','To']])
elo_all = pd.concat(elo_frames, ignore_index=True)
elo_mean_team   = elo_all.groupby('team_name')['Elo'].mean()
elo_mean_global = elo_all['Elo'].mean()

print("Calculando ELO...")
ht_elo, at_elo = [], []
for _, row in df.iterrows():
    d = row['Date']
    for team, lst in [(row['HomeTeam'], ht_elo), (row['AwayTeam'], at_elo)]:
        sub = elo_all[(elo_all['team_name'] == team) & (elo_all['From'] <= d) & (elo_all['To'] >= d)]
        lst.append(float(sub['Elo'].iloc[0]) if len(sub) > 0 else None)

df['HT_ELO'] = ht_elo
df['AT_ELO'] = at_elo
for idx, row in df[df['HT_ELO'].isna()].iterrows():
    df.at[idx, 'HT_ELO'] = elo_mean_team.get(row['HomeTeam'], elo_mean_global)
for idx, row in df[df['AT_ELO'].isna()].iterrows():
    df.at[idx, 'AT_ELO'] = elo_mean_team.get(row['AwayTeam'], elo_mean_global)
df['ELO_Diff'] = df['HT_ELO'] - df['AT_ELO']

print("Todos los features calculados.")

# ============================================================================
# DEFINIR SETS DE FEATURES
# ============================================================================

features_base   = [f for f in ALL_FEATURES if f in df.columns]
features_nuevas = ['HT_Goals_Diff','AT_Goals_Diff','AT_HTR_Rate','PS_vs_Avg_H',
                   'HT_HTR_Rate','HT_ShotAcc','AT_ShotAcc',
                   'Over25_Prob','Over25_Move','Sharp_H','Sharp_A','PS_vs_Avg_A',
                   'HT_Poss_Rolling','AT_Poss_Rolling','Poss_Diff_Rolling',
                   'ELO_Diff','HT_ELO','AT_ELO']
features_todas  = list(dict.fromkeys(features_base + [f for f in features_nuevas if f in df.columns]))

print(f"Features base: {len(features_base)}  |  Features todas: {len(features_todas)}")

tscv = TimeSeriesSplit(n_splits=5)

def test_model(X, y, label, model_fn):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
    m = model_fn(y_tr)
    m.fit(X_tr, y_tr) if not hasattr(m, '_needs_sw') else None
    f1 = f1_score(y_te, m.predict(X_te), average='weighted')
    print(f"  {label:<55} F1={f1:.4f}")
    return f1, m

print("\n" + "="*70)
print("COMPARATIVA DE MODELOS Y FEATURES")
print("="*70)

results = {}

# RF base
X = df[features_base].fillna(0); y = df['FTR_numeric']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
m = RandomForestClassifier(**PARAMS_OPTIMOS); m.fit(X_tr, y_tr)
f1 = f1_score(y_te, m.predict(X_te), average='weighted')
print(f"  {'RF base (54 features)':<55} F1={f1:.4f}")
results['RF_base'] = f1

# RF todas las features
X = df[features_todas].fillna(0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
m = RandomForestClassifier(**PARAMS_OPTIMOS); m.fit(X_tr, y_tr)
f1 = f1_score(y_te, m.predict(X_te), average='weighted')
print(f"  {'RF todas features':<55} F1={f1:.4f}")
results['RF_todas'] = f1

# XGBoost base
sw = compute_sample_weight(class_weight=PESOS_OPTIMOS, y=y_tr)
X = df[features_base].fillna(0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
sw = compute_sample_weight(class_weight=PESOS_OPTIMOS, y=y_tr)
m = XGBClassifier(**PARAMS_XGB); m.fit(X_tr, y_tr, sample_weight=sw)
f1 = f1_score(y_te, m.predict(X_te), average='weighted')
print(f"  {'XGB base features PARAMS_XGB':<55} F1={f1:.4f}")
results['XGB_base'] = f1

# XGBoost todas las features
X = df[features_todas].fillna(0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
sw = compute_sample_weight(class_weight=PESOS_OPTIMOS, y=y_tr)
m = XGBClassifier(**PARAMS_XGB); m.fit(X_tr, y_tr, sample_weight=sw)
f1 = f1_score(y_te, m.predict(X_te), average='weighted')
print(f"  {'XGB todas features PARAMS_XGB':<55} F1={f1:.4f}")
results['XGB_todas'] = f1

# XGBoost RandomSearch con features buenas (las que pasaron el filtro)
features_buenas = [f for f in features_base if f not in ['H2H_Matches']] + \
                  ['HT_Goals_Diff','AT_Goals_Diff','AT_HTR_Rate','PS_vs_Avg_H','ELO_Diff']
features_buenas = [f for f in dict.fromkeys(features_buenas) if f in df.columns]

X = df[features_buenas].fillna(0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
sw = compute_sample_weight(class_weight=PESOS_OPTIMOS, y=y_tr)

print(f"\n  RandomSearch XGB con {len(features_buenas)} features (50 iters)...")
param_dist = {
    'n_estimators':     [200, 300, 400, 500, 600],
    'max_depth':        [3, 4, 5, 6, 7, 8],
    'learning_rate':    [0.02, 0.03, 0.05, 0.08, 0.10, 0.15],
    'subsample':        [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'reg_alpha':        [0, 0.05, 0.1, 0.5, 1.0, 2.0],
    'reg_lambda':       [0.5, 1.0, 1.5, 2.0, 3.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma':            [0, 0.1, 0.2, 0.5],
}
xgb_base_m = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
rs = RandomizedSearchCV(
    xgb_base_m, param_dist, n_iter=50, cv=tscv,
    scoring='f1_weighted', random_state=42, n_jobs=-1, verbose=0
)
rs.fit(X_tr, y_tr, sample_weight=sw)
f1_rs = f1_score(y_te, rs.best_estimator_.predict(X_te), average='weighted')
print(f"  {'XGB RandomSearch (50 iters)':<55} F1={f1_rs:.4f}")
print(f"  Mejores params: {rs.best_params_}")
results['XGB_RS'] = f1_rs

# ============================================================================
# RESUMEN
# ============================================================================
print("\n" + "="*70)
print("RESUMEN")
print("="*70)
best_name = max(results, key=results.get)
best_f1   = results[best_name]
for name, f1 in results.items():
    mark = " <-- MEJOR" if name == best_name else ""
    print(f"  {name:<35} F1={f1:.4f}{mark}")
print(f"\nObjetivo: 0.5460  Mejor: {best_f1:.4f}  {'*** SUPERADO ***' if best_f1 > 0.5460 else 'no superado'}")
print(f"Best params XGB RS: {rs.best_params_}")
