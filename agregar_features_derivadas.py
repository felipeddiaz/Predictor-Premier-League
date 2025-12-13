# agregar_derivadas_rendimiento.py
import pandas as pd

df = pd.read_csv('./datos/procesados/premier_league_con_features.csv')

# Solo agregar las 3 de rendimiento que faltan
df['Goal_Diff'] = df['HT_AvgGoals'] - df['AT_AvgGoals']
df['Form_Diff'] = df['HT_Form_W'] - df['AT_Form_W']
df['Shots_Diff'] = df['HT_AvgShotsTarget'] - df['AT_AvgShotsTarget']

df.to_csv('./datos/procesados/premier_league_con_features.csv', index=False)

print("✅ Agregadas: Goal_Diff, Form_Diff, Shots_Diff")
print("🚀 Ejecuta: python 02_entrenar_con_cuotas.py")