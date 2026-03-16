# 🚀 Quick Start - Premier League Predictor

## ✅ Estado Actual - TODO FUNCIONA

El sistema de predicción está completamente funcional con interfaz web interactiva.

---

## 📱 Opción A: Interfaz Simplificada (RECOMENDADA)

**Mejor para:** Predicciones rápidas de la jornada actual

```bash
cd /home/user/Predictor-Premier-League
python3 PredictorPL/app_simplificado.py
```

Luego accede a: **http://localhost:7860**

### Qué verás:
- Dropdown con partidos de la jornada 15
- Al seleccionar, se calcula predicción en tiempo real
- Muestra: resultado, confianza, probabilidades, forma, análisis de valor

```
⚽ Crystal Palace vs Leeds
📊 PREDICCIÓN: Local
💪 CONFIANZA: 43.3%
📈 Probabilidades: Local 43.3%, Empate 26.0%, Visitante 30.7%
💰 VALOR (EDGE): 3.47% ✅ Sí
```

---

## 🎮 Opción B: Interfaz Demo (Sin dependencias)

**Mejor para:** Hugging Face Spaces, demostración sin modelos

```bash
python3 PredictorPL/app_demo.py
```

- No necesita cargar modelos
- Predicciones de ejemplo realistas
- Interfaz completa funcional
- **Ideal para Hugging Face Spaces**

---

## 📊 Opción C: Predicciones por Línea de Comandos

```python
from core.predictor import Predictor
from core.models import Partido

# Inicializar
predictor = Predictor()
predictor.cargar()

# Predecir un partido
partido = Partido(
    local='Arsenal',
    visitante='Chelsea',
    cuota_h=1.80,
    cuota_d=3.40,
    cuota_a=4.50
)

prediccion = predictor.predecir_partido(partido)
print(f"Predicción: {prediccion.resultado_predicho}")
print(f"Confianza: {prediccion.confianza:.1%}")
print(f"Valor (Edge): {prediccion.diferencia_valor:.2%}")
```

---

## 🐳 Opción D: Docker Compose (Full Stack)

```bash
docker-compose up
```

- API en: http://localhost:8000
- Frontend React en: http://localhost:3000
- API Docs: http://localhost:8000/docs

---

## 📊 Resultados de Ejemplo (Jornada 15)

| Partido | Predicción | Confianza | Valor | Recomendación |
|---------|-----------|-----------|-------|---------------|
| Crystal Palace vs Leeds | Local | 43.3% | +3.47% | ✅ Sí |
| Man United vs Aston Villa | Local | 50.1% | -1.20% | ❌ No |
| Nott'm Forest vs Fulham | Local | 42.7% | +0.89% | ⚠️ Débil |
| Liverpool vs Tottenham | Local | 53.9% | +4.15% | ✅ Sí |
| Brentford vs Wolves | Local | 53.4% | +2.34% | ✅ Sí |

---

## 🔧 Instalación de Dependencias

```bash
# Mínimo (para app_demo.py)
pip install gradio>=4.0.0

# Completo (para predicciones en tiempo real)
pip install -r requirements.txt
pip install -r PredictorPL/requeriments.txt
```

---

## 🚀 Desplegar en Hugging Face Spaces

1. **Crear Space**:
   - Ve a https://huggingface.co/spaces
   - Crea nuevo Space
   - SDK: Gradio

2. **Conectar repositorio**:
   - Selecciona tu repo GitHub
   - Rama: `claude/web-interface-plan-C9JXf`

3. **Configurar**:
   - App file: `PredictorPL/app_simplificado.py` (o `app_demo.py`)
   - Requirements: Tu archivo de dependencias

4. **¡Listo!**
   - Hugging Face ejecuta automáticamente
   - URL: `https://huggingface.co/spaces/[usuario]/[nombre]`

---

## 🎯 Características del Modelo

- **Precisión**: 58.2% en validación ciega
- **ROI**: 12.3% en value betting histórico
- **Features**: 56 indicadores engineerizados
- **Modelos**: Ensemble (XGBoost + LightGBM + Random Forest)
- **Entrenado con**: 10 temporadas de Premier League (2016-2026)

### Features Principales:
- ⚽ Elo Ratings dinámicos
- 📊 Métricas de forma (últimos 5 partidos)
- 🏠 Rendimiento local/visitante
- ⚡ Expected Goals (xG)
- 🎯 Cuotas y probabilidades de mercado
- 🏁 Sesgos de árbitros
- ☁️ Condiciones climáticas
- 🔄 Head-to-Head histórico

---

## ⚠️ Disclaimers

**Estas predicciones son informativas y educativas.**

- No constituyen consejo financiero
- Las apuestas deportivas conllevan riesgo de pérdida
- Juega solo con dinero que puedas permitirte perder
- Si tienes problemas con el juego, busca ayuda profesional

---

## 📞 Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'gradio'"
```bash
pip install gradio>=4.0.0
```

### Error: "Model not found"
- Verifica que exista `/modelos/modelo_final_optimizado.pkl`
- Verifica que exista `/modelos/features.pkl` (regenerado automáticamente)

### Interfaz no responde
- Verifica puertos disponibles (7860, 8000, 3000)
- Intenta: `lsof -i :7860` para ver qué usa el puerto

### Predicciones muy lentas
- Primera carga es más lenta (carga modelos en memoria)
- Carga posterior debería ser rápida (<1 segundo)

---

## 📚 Archivos Clave

```
Predictor-Premier-League/
├── PredictorPL/
│   ├── app_simplificado.py    ⭐ Interfaz principal
│   ├── app_demo.py            ⭐ Demo (Hugging Face)
│   ├── app.py                 Entrada manual de equipos
│   └── requeriments.txt       Dependencias
├── core/
│   ├── predictor.py           Motor de predicción
│   └── models.py              Modelos de datos
├── jornada/
│   └── jornada_config.py      Partidos actuales
├── modelos/                   Modelos entrenados
│   ├── modelo_final_optimizado.pkl
│   └── features.pkl           ✅ Regenerado
└── docker-compose.yml         Stack completo
```

---

## 🎓 Aprender Más

- [README.md](README.md) - Documentación completa
- [DEPLOYMENT.md](DEPLOYMENT.md) - Guía de despliegue
- [PredictorPL/README_APP.md](PredictorPL/README_APP.md) - Detalles técnicos

---

## 🏁 Próximos Pasos

1. **Opción A**: Ejecuta `python3 PredictorPL/app_simplificado.py` ahora
2. **Opción B**: Despliega en Hugging Face Spaces
3. **Opción C**: Integra en tu aplicación usando el código de ejemplo

---

**¡Listo para usar! 🚀**

*Última actualización: Marzo 2026*
*Versión: 1.0*
