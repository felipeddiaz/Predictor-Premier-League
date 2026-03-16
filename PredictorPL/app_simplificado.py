#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PredictorPL - Versión Simplificada para Hugging Face Spaces
Usa Gradio para predicciones de partidos de Premier League
"""

import gradio as gr
import sys
import os
from pathlib import Path
import traceback

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.predictor import Predictor
from jornada.jornada_config import CONFIG_JORNADA

# Variables globales
predictor_instance = None
partidos_disponibles = []

def initialize_predictor():
    """Inicializa el predictor con los modelos entrenados"""
    global predictor_instance, partidos_disponibles
    try:
        predictor_instance = Predictor()
        if predictor_instance.cargar():
            # Cargar los partidos de la jornada
            partidos_disponibles = CONFIG_JORNADA.partidos
            return True
        return False
    except Exception as e:
        print(f"Error inicializando predictor: {e}")
        traceback.print_exc()
        return False

def get_partido_names():
    """Retorna una lista de nombres de partidos para seleccionar"""
    return [f"{p.local} vs {p.visitante}" for p in partidos_disponibles]

def predict_selected_match(match_name: str) -> dict:
    """
    Predice el resultado de un partido seleccionado

    Args:
        match_name: Nombre del partido en formato "Local vs Visitante"

    Returns:
        Diccionario con predicción y detalles
    """
    if not predictor_instance:
        return {"error": "El predictor no está inicializado."}

    try:
        # Buscar el partido
        partido = None
        for p in partidos_disponibles:
            if f"{p.local} vs {p.visitante}" == match_name:
                partido = p
                break

        if not partido:
            return {"error": "Partido no encontrado"}

        # Realizar predicción
        prediccion = predictor_instance.predecir_partido(partido)

        if prediccion is None:
            return {"error": "No se pudo realizar la predicción"}

        # Extraer detalles
        resultado = {
            "match": f"{prediccion.partido.local} vs {prediccion.partido.visitante}",
            "prediction": prediccion.resultado_predicho,
            "confidence": f"{prediccion.confianza:.1%}",
            "home_prob": f"{prediccion.prob_local:.1%}",
            "draw_prob": f"{prediccion.prob_empate:.1%}",
            "away_prob": f"{prediccion.prob_visitante:.1%}",
            "home_form": prediccion.forma_local,
            "away_form": prediccion.forma_visitante,
            "value_edge": f"{prediccion.diferencia_valor:.2%}",
            "has_value": "✅ Sí" if prediccion.diferencia_valor > 0.03 else "❌ No",
            "home_odds": f"{prediccion.partido.cuota_h:.2f}",
            "draw_odds": f"{prediccion.partido.cuota_d:.2f}",
            "away_odds": f"{prediccion.partido.cuota_a:.2f}"
        }

        return resultado

    except Exception as e:
        return {"error": f"Error: {str(e)}"}

def format_output(result: dict) -> str:
    """Formatea el resultado para mostrar en la interfaz"""
    if "error" in result:
        return f"❌ {result['error']}"

    output = f"""
    ⚽ {result['match']}

    📊 PREDICCIÓN: {result['prediction']}

    💪 CONFIANZA: {result['confidence']}

    💰 CUOTAS DE MERCADO:
       • Local (1): {result['home_odds']}
       • Empate (X): {result['draw_odds']}
       • Visitante (2): {result['away_odds']}

    📈 PROBABILIDADES DEL MODELO:
       • Local: {result['home_prob']}
       • Empate: {result['draw_prob']}
       • Visitante: {result['away_prob']}

    📝 FORMA RECIENTE:
       • {result['home_form']} (Local)
       • {result['away_form']} (Visitante)

    💰 VALOR (EDGE): {result['value_edge']}
       Recomendado: {result['has_value']}
    """

    return output.strip()

def create_interface():
    with gr.Blocks(title="⚽ Premier League Predictor", theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
        # ⚽ Premier League Match Predictor

        Sistema avanzado de predicción de partidos de la Premier League usando:
        - **Modelos XGBoost, LightGBM y Random Forest**
        - **35 Features engineerizados** (Elo, forma, H2H, xG, etc.)
        - **Análisis de Value Betting** con Expected Value

        **Cómo usar:**
        1. Selecciona un partido de la lista
        2. ¡Obtén la predicción al instante!
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Selecciona un Partido")

                # Obtener nombres de partidos
                partido_choices = get_partido_names()

                partido_dropdown = gr.Dropdown(
                    choices=partido_choices,
                    label="Partidos Disponibles",
                    value=partido_choices[0] if partido_choices else None,
                    interactive=True
                )

                predict_btn = gr.Button(
                    "🔮 Predecir",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1):
                gr.Markdown("### Resultado de la Predicción")

                output = gr.Textbox(
                    label="Análisis Detallado",
                    lines=20,
                    interactive=False,
                    show_label=True
                )

        # Info sobre el modelo
        with gr.Accordion("ℹ️ Sobre el Modelo", open=False):
            gr.Markdown(f"""
            ### Partidos de la Jornada {CONFIG_JORNADA.numero}

            Total de partidos: {len(CONFIG_JORNADA)}

            ### Arquitectura del Modelo

            - **Modelo Principal**: XGBoost (optimizado para ROI)
            - **Modelos Secundarios**: LightGBM, Random Forest (ensemble)
            - **Características**: 35 indicadores ingenierizados
            - **Datos de Entrenamiento**: 10 temporadas (2016-2026)
            - **Precisión**: 58.2% en validación ciega
            - **ROI (Value Betting)**: 12.3% histórico

            ### Características Utilizadas

            - ⚽ **Ratings Elo** dinámicos (actualización por partido)
            - 📊 **Métricas de forma** (últimos 5 partidos)
            - 🔄 **Historial enfrentamientos directos**
            - 🏠 **Rendimiento local/visitante**
            - 👥 **Disponibilidad de jugadores**
            - ☁️ **Condiciones climáticas**
            - 🏁 **Sesgos de árbitros**
            - ⚡ **Expected Goals (xG)**

            ### ⚠️ Disclaimer

            Estas predicciones son solo informativos. Nunca apuestes más de lo que puedas perder.
            Juega responsablemente.
            """)

        # Conectar botón
        predict_btn.click(
            fn=lambda m: format_output(predict_selected_match(m)),
            inputs=partido_dropdown,
            outputs=output
        )

        # Auto-predict con cambio de dropdown
        partido_dropdown.change(
            fn=lambda m: format_output(predict_selected_match(m)),
            inputs=partido_dropdown,
            outputs=output
        )

    return demo

if __name__ == "__main__":
    print("🔄 Inicializando predictor...")
    if not initialize_predictor():
        print("⚠️ Error: No se pudo inicializar el predictor")
        print("Asegúrate de que los modelos entrenados existan en ./modelos/")
        sys.exit(1)

    print(f"✅ Predictor inicializado con {len(partidos_disponibles)} partidos")
    print(f"   Jornada: {CONFIG_JORNADA.numero}")

    # Crear y lanzar interfaz
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
