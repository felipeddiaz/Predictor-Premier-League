# -*- coding: utf-8 -*-
"""
Análisis de Importancia de Features
Identifica las features que menos aportan al modelo para considerar eliminarlas.

USO:
    python analizar_features.py

OUTPUT:
    - Ranking completo de features por importancia
    - Features candidatas a eliminar (< umbral)
    - Gráfico de importancia guardado en ./modelos/
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Importar desde tu proyecto
from config import (
    ARCHIVO_FEATURES,
    FEATURES_BASE,
    FEATURES_CUOTAS,
    FEATURES_CUOTAS_DERIVADAS,
    FEATURES_XG,
    FEATURES_H2H,
    FEATURES_H2H_DERIVADAS,
    FEATURES_TABLA,
    FEATURES_ASIAN_HANDICAP,
    PARAMS_OPTIMOS,
    RANDOM_SEED,
    TEST_SIZE,
    RUTA_MODELOS,
)
from utils import agregar_xg_rolling, agregar_features_tabla, agregar_features_cuotas_derivadas, agregar_features_asian_handicap

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

UMBRAL_IMPORTANCIA = 0.01  # Features con importancia < 1% son candidatas a eliminar
TOP_N_MOSTRAR = 20         # Mostrar top N features en el gráfico

# ============================================================================
# CARGAR DATOS
# ============================================================================

print("="*70)
print("ANÁLISIS DE IMPORTANCIA DE FEATURES")
print("="*70)

df = pd.read_csv(ARCHIVO_FEATURES)
print(f"\n✅ Cargados: {len(df)} partidos")

# Aplicar feature engineering
df = agregar_xg_rolling(df)
df = agregar_features_tabla(df)
df = agregar_features_cuotas_derivadas(df)
df = agregar_features_asian_handicap(df)

# Construir lista de features
all_features = (
    FEATURES_BASE + 
    FEATURES_CUOTAS + 
    FEATURES_CUOTAS_DERIVADAS + 
    FEATURES_XG + 
    FEATURES_H2H + 
    FEATURES_H2H_DERIVADAS + 
    FEATURES_TABLA +
    FEATURES_ASIAN_HANDICAP
)

features = [f for f in all_features if f in df.columns]

X = df[features].fillna(0)
y = df['FTR_numeric']

# Split temporal
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=False, random_state=RANDOM_SEED
)

print(f"📊 Features totales: {len(features)}")
print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

# ============================================================================
# ENTRENAR MODELO Y OBTENER IMPORTANCIAS
# ============================================================================

print("\n" + "="*70)
print("ENTRENANDO MODELO PARA ANÁLISIS")
print("="*70)

# Usar los parámetros óptimos de config.py
modelo = RandomForestClassifier(**PARAMS_OPTIMOS)
modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)
f1_actual = f1_score(y_test, pred, average='weighted')
print(f"\n✅ F1-Score actual: {f1_actual:.4f}")

# Obtener importancias
importancias = modelo.feature_importances_

df_imp = pd.DataFrame({
    'Feature': features,
    'Importancia': importancias
}).sort_values('Importancia', ascending=False).reset_index(drop=True)

# Agregar columna de categoría
def get_categoria(feature):
    if feature in FEATURES_BASE:
        return 'Base'
    elif feature in FEATURES_CUOTAS:
        return 'Cuotas'
    elif feature in FEATURES_CUOTAS_DERIVADAS:
        return 'Cuotas Deriv.'
    elif feature in FEATURES_XG:
        return 'xG'
    elif feature in FEATURES_H2H:
        return 'H2H'
    elif feature in FEATURES_H2H_DERIVADAS:
        return 'H2H Deriv.'
    elif feature in FEATURES_TABLA:
        return 'Tabla'
    elif feature in FEATURES_ASIAN_HANDICAP:
        return 'Asian HCP'
    else:
        return 'Otra'

df_imp['Categoria'] = df_imp['Feature'].apply(get_categoria)
df_imp['Ranking'] = range(1, len(df_imp) + 1)
df_imp['Acumulado'] = df_imp['Importancia'].cumsum()

# ============================================================================
# MOSTRAR RANKING COMPLETO
# ============================================================================

print("\n" + "="*70)
print("RANKING COMPLETO DE FEATURES")
print("="*70)

print(f"\n{'Rank':<6} {'Feature':<30} {'Importancia':<12} {'Categoría':<15} {'Acum.':<8}")
print("-" * 75)

for _, row in df_imp.iterrows():
    marcador = "⚠️" if row['Importancia'] < UMBRAL_IMPORTANCIA else "  "
    print(f"{row['Ranking']:<6} {row['Feature']:<30} {row['Importancia']:.4f}       {row['Categoria']:<15} {row['Acumulado']:.2%} {marcador}")

# ============================================================================
# FEATURES CANDIDATAS A ELIMINAR
# ============================================================================

print("\n" + "="*70)
print(f"FEATURES CON IMPORTANCIA < {UMBRAL_IMPORTANCIA:.0%} (candidatas a eliminar)")
print("="*70)

candidatas = df_imp[df_imp['Importancia'] < UMBRAL_IMPORTANCIA]

if len(candidatas) == 0:
    print("\n✅ Todas las features tienen importancia >= umbral")
else:
    print(f"\n⚠️  {len(candidatas)} features con baja importancia:\n")
    
    for _, row in candidatas.iterrows():
        print(f"   • {row['Feature']:<30} {row['Importancia']:.4f} ({row['Categoria']})")
    
    # Agrupar por categoría
    print(f"\n📊 Resumen por categoría:")
    resumen = candidatas.groupby('Categoria').size()
    for cat, count in resumen.items():
        total_cat = len(df_imp[df_imp['Categoria'] == cat])
        print(f"   • {cat}: {count}/{total_cat} features con baja importancia")

# ============================================================================
# ANÁLISIS POR CATEGORÍA
# ============================================================================

print("\n" + "="*70)
print("IMPORTANCIA TOTAL POR CATEGORÍA")
print("="*70)

resumen_cat = df_imp.groupby('Categoria').agg({
    'Importancia': ['sum', 'mean', 'count']
}).round(4)
resumen_cat.columns = ['Suma', 'Promedio', 'Count']
resumen_cat = resumen_cat.sort_values('Suma', ascending=False)

print(f"\n{'Categoría':<15} {'Suma':<10} {'Promedio':<10} {'Features':<10}")
print("-" * 50)
for cat, row in resumen_cat.iterrows():
    print(f"{cat:<15} {row['Suma']:.4f}     {row['Promedio']:.4f}     {int(row['Count'])}")

# ============================================================================
# TOP Y BOTTOM FEATURES
# ============================================================================

print("\n" + "="*70)
print("TOP 10 FEATURES MÁS IMPORTANTES")
print("="*70)

for i, (_, row) in enumerate(df_imp.head(10).iterrows(), 1):
    barra = "█" * int(row['Importancia'] * 100)
    print(f"   {i:2}. {row['Feature']:<28} {row['Importancia']:.4f} {barra}")

print("\n" + "="*70)
print("BOTTOM 10 FEATURES MENOS IMPORTANTES")
print("="*70)

for i, (_, row) in enumerate(df_imp.tail(10).iterrows(), 1):
    print(f"   {i:2}. {row['Feature']:<28} {row['Importancia']:.4f} ({row['Categoria']})")

# ============================================================================
# SIMULACIÓN: ¿QUÉ PASA SI ELIMINAMOS LAS PEORES?
# ============================================================================

print("\n" + "="*70)
print("SIMULACIÓN: ELIMINAR FEATURES DE BAJA IMPORTANCIA")
print("="*70)

# Probar eliminando features con < 1%, < 0.5%, etc.
umbrales_test = [0.005, 0.01, 0.015, 0.02]

print(f"\n{'Umbral':<10} {'Eliminadas':<12} {'Restantes':<12} {'F1-Score':<12} {'Cambio':<10}")
print("-" * 60)

for umbral in umbrales_test:
    features_filtradas = df_imp[df_imp['Importancia'] >= umbral]['Feature'].tolist()
    n_eliminadas = len(features) - len(features_filtradas)
    
    if len(features_filtradas) < 5:  # Mínimo de features
        continue
    
    X_train_filt = X_train[features_filtradas]
    X_test_filt = X_test[features_filtradas]
    
    modelo_filt = RandomForestClassifier(**PARAMS_OPTIMOS)
    modelo_filt.fit(X_train_filt, y_train)
    pred_filt = modelo_filt.predict(X_test_filt)
    f1_filt = f1_score(y_test, pred_filt, average='weighted')
    
    cambio = f1_filt - f1_actual
    indicador = "📈" if cambio > 0 else "📉" if cambio < 0 else "➡️"
    
    print(f"{umbral:.1%}       {n_eliminadas:<12} {len(features_filtradas):<12} {f1_filt:.4f}       {cambio:+.4f} {indicador}")

# ============================================================================
# GENERAR GRÁFICO
# ============================================================================

print("\n" + "="*70)
print("GENERANDO VISUALIZACIÓN")
print("="*70)

# Colores por categoría
colores = {
    'Base': '#2ecc71',
    'Cuotas': '#3498db',
    'Cuotas Deriv.': '#9b59b6',
    'xG': '#e74c3c',
    'H2H': '#f39c12',
    'H2H Deriv.': '#e67e22',
    'Tabla': '#1abc9c',
    'Asian HCP': '#d35400',
    'Otra': '#95a5a6'
}

fig, axes = plt.subplots(1, 2, figsize=(16, 10))

# Gráfico 1: Top N features
ax1 = axes[0]
top_df = df_imp.head(TOP_N_MOSTRAR)
colors = [colores.get(cat, '#95a5a6') for cat in top_df['Categoria']]
bars = ax1.barh(range(len(top_df)), top_df['Importancia'], color=colors)
ax1.set_yticks(range(len(top_df)))
ax1.set_yticklabels(top_df['Feature'])
ax1.invert_yaxis()
ax1.set_xlabel('Importancia')
ax1.set_title(f'Top {TOP_N_MOSTRAR} Features más Importantes', fontsize=14, fontweight='bold')
ax1.axvline(x=UMBRAL_IMPORTANCIA, color='red', linestyle='--', label=f'Umbral ({UMBRAL_IMPORTANCIA:.0%})')
ax1.legend()

# Agregar valores en las barras
for i, (bar, imp) in enumerate(zip(bars, top_df['Importancia'])):
    ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, 
             f'{imp:.3f}', va='center', fontsize=9)

# Gráfico 2: Importancia por categoría
ax2 = axes[1]
cat_order = resumen_cat.index.tolist()
cat_colors = [colores.get(cat, '#95a5a6') for cat in cat_order]
bars2 = ax2.barh(cat_order, resumen_cat['Suma'], color=cat_colors)
ax2.set_xlabel('Importancia Total')
ax2.set_title('Importancia Total por Categoría', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

# Agregar valores
for bar, val in zip(bars2, resumen_cat['Suma']):
    ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}', va='center', fontsize=10)

plt.tight_layout()

# Guardar
archivo_grafico = f"{RUTA_MODELOS}/feature_importance_analysis.png"
plt.savefig(archivo_grafico, dpi=150, bbox_inches='tight')
print(f"✅ Gráfico guardado: {archivo_grafico}")
plt.close()

# ============================================================================
# RECOMENDACIONES
# ============================================================================

print("\n" + "="*70)
print("RECOMENDACIONES")
print("="*70)

# Features a considerar eliminar
if len(candidatas) > 0:
    print(f"\n🔧 FEATURES A CONSIDERAR ELIMINAR ({len(candidatas)}):")
    lista_eliminar = candidatas['Feature'].tolist()
    print(f"   {lista_eliminar}")
    
    # Código para copiar
    print(f"\n📋 CÓDIGO PARA EXCLUIR (agregar a config.py):")
    print(f"""
FEATURES_EXCLUIR = {lista_eliminar}

# Y modificar ALL_FEATURES para excluirlas:
ALL_FEATURES = [f for f in (
    FEATURES_BASE + FEATURES_CUOTAS + FEATURES_CUOTAS_DERIVADAS +
    FEATURES_XG + FEATURES_H2H + FEATURES_H2H_DERIVADAS + FEATURES_TABLA
) if f not in FEATURES_EXCLUIR]
""")

# Categorías débiles
cat_debiles = resumen_cat[resumen_cat['Suma'] < 0.05]
if len(cat_debiles) > 0:
    print(f"\n⚠️  CATEGORÍAS CON BAJA IMPORTANCIA TOTAL (<5%):")
    for cat in cat_debiles.index:
        print(f"   • {cat}: {resumen_cat.loc[cat, 'Suma']:.2%} - Considerar eliminar grupo completo")

print("\n" + "="*70)
print("✅ ANÁLISIS COMPLETADO")
print("="*70)
print(f"\n💡 Siguiente paso: Ejecuta optimizar_pesos_optuna.py con las features filtradas")
print(f"   para encontrar los pesos óptimos para el nuevo conjunto de features.\n")