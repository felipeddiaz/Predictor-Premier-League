# -*- coding: utf-8 -*-
"""
ANÁLISIS DE FEATURES
Script independiente para analizar qué features aportan más al modelo.
NO modifica nada, solo analiza y reporta.
Incluye análisis de CUOTAS puras y derivadas.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

PESOS_PERSONALIZADOS = {0: 1.0, 1: 2.4, 2: 1.3}

# ============================================================================
# CARGAR DATOS DIRECTAMENTE
# ============================================================================

print("="*70)
print("   ANÁLISIS DE IMPORTANCIA DE FEATURES")
print("="*70)

# Cargar datos
df = pd.read_csv('./datos/procesados/premier_league_RESTAURADO.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
print(f"\n✅ Datos cargados: {len(df)} partidos")

# ============================================================================
# DEFINIR TODAS LAS FEATURES A ANALIZAR (incluyendo cuotas)
# ============================================================================

# Cuotas puras
features_cuotas = ['B365H', 'B365D', 'B365A', 'B365CH', 'B365CD', 'B365CA']

# Crear probabilidades implícitas si hay cuotas
if all(col in df.columns for col in ['B365H', 'B365D', 'B365A']):
    print("\n🔧 Calculando probabilidades implícitas de cuotas...")
    
    prob_h = 1 / df['B365H']
    prob_d = 1 / df['B365D']
    prob_a = 1 / df['B365A']
    total = prob_h + prob_d + prob_a
    
    df['Prob_H'] = prob_h / total
    df['Prob_D'] = prob_d / total
    df['Prob_A'] = prob_a / total
    df['Prob_Diff_HA'] = df['Prob_H'] - df['Prob_A']
    df['Prob_Ratio_HA'] = df['Prob_H'] / (df['Prob_A'] + 0.01)
    df['Prob_Max'] = df[['Prob_H', 'Prob_D', 'Prob_A']].max(axis=1)
    df['Prob_Spread'] = df['Prob_Max'] - df[['Prob_H', 'Prob_D', 'Prob_A']].min(axis=1)
    
    if all(col in df.columns for col in ['B365CH', 'B365CD', 'B365CA']):
        probc_h = 1 / df['B365CH']
        probc_d = 1 / df['B365CD']
        probc_a = 1 / df['B365CA']
        totalc = probc_h + probc_d + probc_a
        
        df['ProbC_H'] = probc_h / totalc
        df['ProbC_D'] = probc_d / totalc
        df['ProbC_A'] = probc_a / totalc
        df['Prob_Move_H'] = df['ProbC_H'] - df['Prob_H']
        df['Prob_Move_D'] = df['ProbC_D'] - df['Prob_D']
        df['Prob_Move_A'] = df['ProbC_A'] - df['Prob_A']
    
    print("   ✅ Probabilidades calculadas")

# Features de probabilidades
features_prob = ['Prob_H', 'Prob_D', 'Prob_A', 'Prob_Diff_HA', 'Prob_Ratio_HA',
                 'Prob_Max', 'Prob_Spread', 'ProbC_H', 'ProbC_D', 'ProbC_A',
                 'Prob_Move_H', 'Prob_Move_D', 'Prob_Move_A']

# Features base
features_base = ['HT_AvgGoals', 'AT_AvgGoals', 'HT_AvgShotsTarget', 'AT_AvgShotsTarget',
                 'HT_Form_W', 'HT_Form_D', 'HT_Form_L', 'AT_Form_W', 'AT_Form_D', 'AT_Form_L']

# Features H2H
features_h2h = ['H2H_Available', 'H2H_Matches', 'H2H_Home_Wins', 'H2H_Draws', 'H2H_Away_Wins',
                'H2H_Home_Goals_Avg', 'H2H_Away_Goals_Avg', 'H2H_Home_Win_Rate', 'H2H_BTTS_Rate',
                'H2H_Goal_Diff', 'H2H_Win_Advantage', 'H2H_Total_Goals_Avg', 'H2H_Home_Consistent']

# Todas las features disponibles
all_possible = features_cuotas + features_prob + features_base + features_h2h
features = [f for f in all_possible if f in df.columns]

print(f"\n📊 Features disponibles para análisis: {len(features)}")
print(f"   • Cuotas puras: {len([f for f in features_cuotas if f in features])}")
print(f"   • Probabilidades: {len([f for f in features_prob if f in features])}")
print(f"   • Base: {len([f for f in features_base if f in features])}")
print(f"   • H2H: {len([f for f in features_h2h if f in features])}")

# Preparar datos
X = df[features].fillna(0)
y = df['FTR_numeric']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ============================================================================
# ENTRENAR MODELO PARA ANALIZAR IMPORTANCIA
# ============================================================================

print("\n🔧 Entrenando modelo para análisis...")

modelo = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    class_weight=PESOS_PERSONALIZADOS,
    random_state=SEED,
    n_jobs=1
)
modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)
acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average='weighted')
print(f"✅ Modelo entrenado - Accuracy: {acc:.2%} | F1: {f1:.4f}")

# ============================================================================
# 1. IMPORTANCIA SEGÚN EL MODELO
# ============================================================================

print("\n" + "="*70)
print("1. IMPORTANCIA SEGÚN EL MODELO")
print("="*70)

importancias = modelo.feature_importances_

# Crear DataFrame
df_imp = pd.DataFrame({
    'Feature': features,
    'Importancia': importancias
}).sort_values('Importancia', ascending=False)

# Calcular porcentaje
total_imp = df_imp['Importancia'].sum()
df_imp['Porcentaje'] = (df_imp['Importancia'] / total_imp) * 100
df_imp['Porcentaje_Acumulado'] = df_imp['Porcentaje'].cumsum()

print(f"\n📊 RANKING COMPLETO DE FEATURES ({len(features)} total):")
print("-"*75)
print(f"{'#':<4} {'Feature':<35} {'Importancia':<12} {'%':<8} {'% Acum':<8}")
print("-"*75)

for i, (_, row) in enumerate(df_imp.iterrows(), 1):
    print(f"{i:<4} {row['Feature']:<35} {row['Importancia']:<12.4f} {row['Porcentaje']:<8.2f} {row['Porcentaje_Acumulado']:<8.1f}")

# ============================================================================
# 2. RESUMEN POR CATEGORÍA
# ============================================================================

print("\n" + "="*70)
print("2. RESUMEN POR CATEGORÍA DE FEATURES")
print("="*70)

categorias = {
    'Cuotas Puras': features_cuotas,
    'Probabilidades': features_prob,
    'Base': features_base,
    'H2H': features_h2h
}

resumen_cat = {}
for cat, feat_list in categorias.items():
    features_cat = [f for f in feat_list if f in features]
    if features_cat:
        imp_cat = df_imp[df_imp['Feature'].isin(features_cat)]['Porcentaje'].sum()
        resumen_cat[cat] = {
            'n_features': len(features_cat),
            'porcentaje': imp_cat,
            'features': features_cat
        }

print(f"\n{'Categoría':<20} {'# Features':<12} {'% Importancia':<15}")
print("-"*50)
for cat, data in sorted(resumen_cat.items(), key=lambda x: -x[1]['porcentaje']):
    print(f"{cat:<20} {data['n_features']:<12} {data['porcentaje']:<15.2f}%")

# ============================================================================
# 3. COMPARACIÓN: CUOTAS PURAS vs PROBABILIDADES
# ============================================================================

print("\n" + "="*70)
print("3. COMPARACIÓN: CUOTAS PURAS vs PROBABILIDADES DERIVADAS")
print("="*70)

cuotas_en_modelo = [f for f in features_cuotas if f in features]
probs_en_modelo = [f for f in features_prob if f in features]

if cuotas_en_modelo:
    imp_cuotas = df_imp[df_imp['Feature'].isin(cuotas_en_modelo)]['Porcentaje'].sum()
    print(f"\n📊 CUOTAS PURAS ({len(cuotas_en_modelo)} features): {imp_cuotas:.2f}%")
    for f in cuotas_en_modelo:
        pct = df_imp[df_imp['Feature'] == f]['Porcentaje'].values
        if len(pct) > 0:
            print(f"   • {f:<20} {pct[0]:.2f}%")

if probs_en_modelo:
    imp_probs = df_imp[df_imp['Feature'].isin(probs_en_modelo)]['Porcentaje'].sum()
    print(f"\n📊 PROBABILIDADES DERIVADAS ({len(probs_en_modelo)} features): {imp_probs:.2f}%")
    for f in probs_en_modelo:
        pct = df_imp[df_imp['Feature'] == f]['Porcentaje'].values
        if len(pct) > 0:
            print(f"   • {f:<20} {pct[0]:.2f}%")

if cuotas_en_modelo and probs_en_modelo:
    print(f"\n🏆 VEREDICTO:")
    if imp_cuotas > imp_probs:
        print(f"   Las CUOTAS PURAS aportan más ({imp_cuotas:.1f}% vs {imp_probs:.1f}%)")
    else:
        print(f"   Las PROBABILIDADES DERIVADAS aportan más ({imp_probs:.1f}% vs {imp_cuotas:.1f}%)")

# ============================================================================
# 4. TOP FEATURES
# ============================================================================

print("\n" + "="*70)
print("4. TOP FEATURES (Acumulan 50% de importancia)")
print("="*70)

top_50 = df_imp[df_imp['Porcentaje_Acumulado'] <= 50]
if len(top_50) == 0:
    top_50 = df_imp.head(1)

print(f"\n🏆 Las siguientes {len(top_50)} features acumulan ~50% de la importancia:")
print("-"*50)
for _, row in top_50.iterrows():
    print(f"   • {row['Feature']:<30} ({row['Porcentaje']:.2f}%)")

# ============================================================================
# 5. FEATURES CON POCA IMPORTANCIA
# ============================================================================

print("\n" + "="*70)
print("5. FEATURES CON POCA IMPORTANCIA (<1%)")
print("="*70)

low_imp = df_imp[df_imp['Porcentaje'] < 1.0]
print(f"\n⚠️ {len(low_imp)} features aportan menos del 1% cada una:")
print("-"*50)
for _, row in low_imp.iterrows():
    print(f"   • {row['Feature']:<30} ({row['Porcentaje']:.2f}%)")

print(f"\n   📊 Estas {len(low_imp)} features juntas aportan: {low_imp['Porcentaje'].sum():.1f}%")

# ============================================================================
# 6. FEATURES REDUNDANTES
# ============================================================================

print("\n" + "="*70)
print("6. FEATURES REDUNDANTES (Correlación > 0.85)")
print("="*70)

corr_matrix = X.corr().abs()

redundantes = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.85:
            f1, f2 = corr_matrix.columns[i], corr_matrix.columns[j]
            imp1 = df_imp[df_imp['Feature'] == f1]['Porcentaje'].values
            imp2 = df_imp[df_imp['Feature'] == f2]['Porcentaje'].values
            imp1 = imp1[0] if len(imp1) > 0 else 0
            imp2 = imp2[0] if len(imp2) > 0 else 0
            
            redundantes.append({
                'Feature_1': f1,
                'Feature_2': f2,
                'Correlacion': corr_matrix.iloc[i, j],
                'Imp_1': imp1,
                'Imp_2': imp2,
                'Eliminar': f1 if imp1 < imp2 else f2
            })

if redundantes:
    df_red = pd.DataFrame(redundantes).sort_values('Correlacion', ascending=False)
    print(f"\n⚠️ {len(df_red)} pares de features muy correlacionadas:")
    print("-"*85)
    print(f"{'Feature 1':<25} {'Feature 2':<25} {'Corr':<8} {'Sugerencia Eliminar':<25}")
    print("-"*85)
    for _, row in df_red.head(25).iterrows():
        print(f"{row['Feature_1']:<25} {row['Feature_2']:<25} {row['Correlacion']:<8.2f} {row['Eliminar']:<25}")
    
    if len(df_red) > 25:
        print(f"   ... y {len(df_red) - 25} pares más")
else:
    print("\n✅ No hay features altamente correlacionadas")

# ============================================================================
# 7. GUARDAR RESULTADOS
# ============================================================================

print("\n" + "="*70)
print("7. GUARDANDO RESULTADOS")
print("="*70)

os.makedirs('./modelos', exist_ok=True)

df_imp.to_csv('./modelos/feature_importance_analysis.csv', index=False)
print(f"✅ Guardado: modelos/feature_importance_analysis.csv")

# Gráfico
plt.figure(figsize=(12, 10))
top_20 = df_imp.head(20)
colors = ['#2ecc71' if f in features_cuotas else '#3498db' if f in features_prob else '#95a5a6' 
          for f in top_20['Feature']]
plt.barh(range(len(top_20)), top_20['Porcentaje'], color=colors)
plt.yticks(range(len(top_20)), top_20['Feature'])
plt.xlabel('Porcentaje de Importancia (%)')
plt.title('Top 20 Features por Importancia\n(Verde=Cuotas, Azul=Probabilidades, Gris=Otras)', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

for i, (_, row) in enumerate(top_20.iterrows()):
    plt.text(row['Porcentaje'] + 0.1, i, f"{row['Porcentaje']:.1f}%", va='center')

plt.tight_layout()
plt.savefig('./modelos/feature_importance_analysis.png', dpi=150, bbox_inches='tight')
print(f"✅ Guardado: modelos/feature_importance_analysis.png")
plt.close()

print("\n" + "="*70)
print("   ANÁLISIS COMPLETADO")
print("="*70)