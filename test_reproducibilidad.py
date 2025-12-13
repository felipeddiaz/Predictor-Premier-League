#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para verificar que los resultados son reproducibles
Ejecuta 2 veces y compara los resultados
"""
import subprocess
import sys
import re

def extraer_metricas(output):
    """Extrae accuracy y f1-score del output"""
    
    # Buscar Accuracy
    acc_match = re.search(r'Accuracy:\s+([\d.]+)%', output)
    f1_match = re.search(r'F1-Score:\s+([\d.]+)', output)
    
    if acc_match and f1_match:
        acc = float(acc_match.group(1))
        f1 = float(f1_match.group(1))
        return acc, f1
    return None, None

print("\n" + "="*70)
print("TEST DE REPRODUCIBILIDAD")
print("="*70)
print("\n🔄 Ejecutando el script PRIMERA VEZ...")
resultado1 = subprocess.run(
    [sys.executable, '02_entrenar_modelo.py'],
    capture_output=True,
    text=True,
    cwd='.'
)
output1 = resultado1.stdout + resultado1.stderr
acc1, f1_1 = extraer_metricas(output1)

print("✅ Primera ejecución completada")
print(f"   Accuracy: {acc1}%")
print(f"   F1-Score: {f1_1}")

print("\n🔄 Ejecutando el script SEGUNDA VEZ...")
resultado2 = subprocess.run(
    [sys.executable, '02_entrenar_modelo.py'],
    capture_output=True,
    text=True,
    cwd='.'
)
output2 = resultado2.stdout + resultado2.stderr
acc2, f1_2 = extraer_metricas(output2)

print("✅ Segunda ejecución completada")
print(f"   Accuracy: {acc2}%")
print(f"   F1-Score: {f1_2}")

print("\n" + "="*70)
print("COMPARACIÓN")
print("="*70)

if acc1 and f1_1 and acc2 and f1_2:
    acc_diff = abs(acc1 - acc2)
    f1_diff = abs(f1_1 - f1_2)
    
    print(f"\nAccuracy:")
    print(f"   Ejecución 1: {acc1}%")
    print(f"   Ejecución 2: {acc2}%")
    print(f"   Diferencia:  {acc_diff}%")
    
    print(f"\nF1-Score:")
    print(f"   Ejecución 1: {f1_1}")
    print(f"   Ejecución 2: {f1_2}")
    print(f"   Diferencia:  {f1_diff}")
    
    if acc_diff < 0.01 and f1_diff < 0.0001:
        print("\n✅ ¡REPRODUCIBILIDAD CONFIRMADA! Los resultados son IDÉNTICOS")
    elif acc_diff < 0.1 and f1_diff < 0.001:
        print("\n⚠️  Pequeña variación (normal con paralelización)")
    else:
        print("\n❌ Hay variabilidad. Esto NO debería ocurrir.")
else:
    print("\n❌ No se pudieron extraer las métricas del output")
    print("\nOutput 1:")
    print(output1[-500:])
    print("\nOutput 2:")
    print(output2[-500:])

print("\n" + "="*70 + "\n")
