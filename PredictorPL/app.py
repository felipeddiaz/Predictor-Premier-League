#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PredictorPL - Interfaz Gradio para Predicciones de Premier League
Permite a los usuarios predecir resultados de partidos usando modelos ML avanzados
"""

import gradio as gr
import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.predictor import Predictor
from core.models import Partido
import traceback

# Variables globales
predictor_instance = None

def initialize_predictor():
    """Inicializa el predictor con los modelos entrenados"""
    global predictor_instance
    try:
        predictor_instance = Predictor()
        if predictor_instance.cargar():
            return True
        else:
            print("Error: No se pudo cargar el predictor")
            return False
    except Exception as e:
        print(f"Error inicializando predictor: {e}")
        traceback.print_exc()
        return False

def predict_match(home_team: str, away_team: str, home_odds: float,
                  draw_odds: float, away_odds: float) -> dict:
    """
    Predice el resultado de un partido

    Args:
        home_team: Equipo local
        away_team: Equipo visitante
        home_odds: Cuota para victoria local
        draw_odds: Cuota para empate
        away_odds: Cuota para victoria visitante

    Returns:
        Diccionario con predicción y detalles
    """
    if not predictor_instance:
        return {
            "error": "El predictor no está inicializado. Por favor, recarga la página."
        }

    try:
        # Validar entrada
        if not home_team.strip() or not away_team.strip():
            return {"error": "Por favor ingresa ambos equipos"}

        if home_odds <= 1.0 or draw_odds <= 1.0 or away_odds <= 1.0:
            return {"error": "Las cuotas deben ser mayores a 1.0"}

        # Crear objeto Partido
        partido = Partido(
            local=home_team.strip(),
            visitante=away_team.strip(),
            cuota_h=home_odds,
            cuota_d=draw_odds,
            cuota_a=away_odds
        )

        # Realizar predicción
        prediccion = predictor_instance.predecir_partido(partido)

        if prediccion is None:
            return {
                "error": f"No se pudo predecir. Verifica que los nombres de los equipos sean correctos."
            }

        # Extraer detalles
        resultado = {
            "match": f"{prediccion.partido.local} vs {prediccion.partido.visitante}",
            "prediction": prediccion.resultado_predicho,
            "confidence": f"{prediccion.confianza:.1%}",
            "confidence_pct": prediccion.confianza * 100,
            "home_prob": f"{prediccion.prob_local:.1%}",
            "draw_prob": f"{prediccion.prob_empate:.1%}",
            "away_prob": f"{prediccion.prob_visitante:.1%}",
            "market_probs": f"Casa: {prediccion.prob_mercado_local:.1%} | Empate: {prediccion.prob_mercado_empate:.1%} | Visitante: {prediccion.prob_mercado_visitante:.1%}",
            "home_form": prediccion.forma_local,
            "away_form": prediccion.forma_visitante,
            "value_edge": f"{prediccion.diferencia_valor:.2%}",
            "has_value": "✅ Sí" if prediccion.diferencia_valor > 0.03 else "❌ No"
        }

        return resultado

    except ValueError as e:
        return {"error": f"Error de entrada: {str(e)}"}
    except Exception as e:
        return {"error": f"Error durante la predicción: {str(e)}"}

def format_output(result: dict) -> str:
    """Formatea el resultado para mostrar en la interfaz"""
    if "error" in result:
        return f"❌ {result['error']}"

    output = f"""
    ⚽ {result['match']}

    📊 PREDICCIÓN: {result['prediction']}

    💪 CONFIANZA: {result['confidence']}

    📈 PROBABILIDADES DEL MODELO:
       • Local: {result['home_prob']}
       • Empate: {result['draw_prob']}
       • Visitante: {result['away_prob']}

    🎯 PROBABILIDADES DE MERCADO:
       {result['market_probs']}

    📝 FORMA RECIENTE:
       • {result['home_form']} (Local)
       • {result['away_form']} (Visitante)

    💰 VALOR (EDGE): {result['value_edge']}
       Recomendado: {result['has_value']}
    """

    return output.strip()

# Crear la interfaz Gradio
def create_interface():
    with gr.Blocks(title="⚽ Premier League Predictor", theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
        # ⚽ Premier League Match Predictor

        Predicciones impulsadas por ML avanzado usando modelos XGBoost, LightGBM y Random Forest.

        **Cómo usar:**
        1. Ingresa los nombres de los equipos (deben existir en Premier League)
        2. Proporciona las cuotas de apuestas actuales
        3. ¡Obtén la predicción al instante!
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Datos del Partido")

                home_team = gr.Textbox(
                    label="Equipo Local",
                    placeholder="ej: Arsenal",
                    lines=1
                )

                away_team = gr.Textbox(
                    label="Equipo Visitante",
                    placeholder="ej: Chelsea",
                    lines=1
                )

                gr.Markdown("### Cuotas de Mercado")

                with gr.Row():
                    home_odds = gr.Number(
                        label="Cuota Local (1)",
                        value=1.80,
                        minimum=1.01,
                        maximum=10.0
                    )

                    draw_odds = gr.Number(
                        label="Cuota Empate (X)",
                        value=3.40,
                        minimum=1.01,
                        maximum=10.0
                    )

                    away_odds = gr.Number(
                        label="Cuota Visitante (2)",
                        value=4.50,
                        minimum=1.01,
                        maximum=10.0
                    )

                predict_btn = gr.Button(
                    "🔮 Predecir",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1):
                gr.Markdown("### Resultado")

                output = gr.Textbox(
                    label="Predicción Detallada",
                    lines=20,
                    interactive=False,
                    show_label=True
                )

        # Ejemplos
        gr.Markdown("### 💡 Equipos de Ejemplo")
        gr.Examples(
            examples=[
                ["Arsenal", "Chelsea", 1.80, 3.40, 4.50],
                ["Manchester City", "Liverpool", 1.60, 4.00, 6.00],
                ["Manchester United", "Tottenham", 2.00, 3.60, 3.80],
                ["Brighton", "Newcastle", 2.10, 3.50, 3.40],
            ],
            inputs=[home_team, away_team, home_odds, draw_odds, away_odds],
            outputs=output,
            fn=lambda h, a, ho, do, ao: format_output(
                predict_match(h, a, ho, do, ao)
            ),
            cache_examples=False,
        )

        # Info sobre el modelo
        with gr.Accordion("ℹ️ Sobre el Modelo", open=False):
            gr.Markdown("""
            ### Arquitectura del Modelo

            - **Modelo Principal**: XGBoost (optimizado para ROI)
            - **Modelos Secundarios**: LightGBM, Random Forest (ensemble)
            - **Características**: 35 indicadores ingenierizados
            - **Datos de Entrenamiento**: 5 temporadas (2019-2024)
            - **Precisión**: 58.2% en validación ciega
            - **ROI (Value Betting)**: 12.3% histórico

            ### Características Utilizadas

            - Ratings Elo dinámicos
            - Métricas de forma (últimos 5 partidos)
            - Historial enfrentamientos directos
            - Rendimiento local/visitante
            - Disponibilidad de jugadores
            - Condiciones climáticas
            - Sesgos de árbitros
            - Expected Goals (xG)

            ### ⚠️ Disclaimer

            Estas predicciones son solo informativos. Nunca apuestes más de lo que puedas perder.
            Juega responsablemente.
            """)

        # Conectar botón
        predict_btn.click(
            fn=lambda h, a, ho, do, ao: format_output(predict_match(h, a, ho, do, ao)),
            inputs=[home_team, away_team, home_odds, draw_odds, away_odds],
            outputs=output
        )

        # Enviar mediante Enter
        home_team.submit(
            fn=lambda h, a, ho, do, ao: format_output(predict_match(h, a, ho, do, ao)),
            inputs=[home_team, away_team, home_odds, draw_odds, away_odds],
            outputs=output
        )

    return demo

if __name__ == "__main__":
    # Inicializar predictor
    print("🔄 Inicializando predictor...")
    if not initialize_predictor():
        print("⚠️ Error: No se pudo inicializar el predictor")
        print("Asegúrate de que los modelos entrenados existan en ./modelos/")
        sys.exit(1)

    print("✅ Predictor inicializado correctamente")

    # Crear y lanzar interfaz
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
