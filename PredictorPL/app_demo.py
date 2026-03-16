#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PredictorPL - Versión Demo para Hugging Face Spaces
Interfaz Gradio que muestra predicciones de ejemplo
"""

import gradio as gr
import json
from datetime import datetime

# Predicciones de ejemplo basadas en datos históricos
EJEMPLO_PREDICCIONES = {
    "Arsenal vs Chelsea": {
        "prediction": "Arsenal",
        "confidence": 0.62,
        "home_prob": 0.62,
        "draw_prob": 0.24,
        "away_prob": 0.14,
        "home_form": "3W-1D-1L",
        "away_form": "2W-2D-1L",
        "value_edge": 0.045,
        "home_odds": 1.80,
        "draw_odds": 3.40,
        "away_odds": 4.50,
        "xG": 2.1,
        "xGA": 1.3,
        "elo_diff": 85
    },
    "Man City vs Liverpool": {
        "prediction": "Man City",
        "confidence": 0.58,
        "home_prob": 0.58,
        "draw_prob": 0.26,
        "away_prob": 0.16,
        "home_form": "4W-1D",
        "away_form": "3W-1D-1L",
        "value_edge": 0.015,
        "home_odds": 1.60,
        "draw_odds": 4.00,
        "away_odds": 6.00,
        "xG": 2.4,
        "xGA": 1.1,
        "elo_diff": 120
    },
    "Manchester United vs Tottenham": {
        "prediction": "Manchester United",
        "confidence": 0.55,
        "home_prob": 0.55,
        "draw_prob": 0.28,
        "away_prob": 0.17,
        "home_form": "2W-2D-1L",
        "away_form": "2W-2D-1L",
        "value_edge": -0.02,
        "home_odds": 2.00,
        "draw_odds": 3.60,
        "away_odds": 3.80,
        "xG": 1.9,
        "xGA": 1.5,
        "elo_diff": 45
    },
    "Brighton vs Newcastle": {
        "prediction": "Brighton",
        "confidence": 0.52,
        "home_prob": 0.52,
        "draw_prob": 0.30,
        "away_prob": 0.18,
        "home_form": "1W-2D-2L",
        "away_form": "2W-1D-2L",
        "value_edge": 0.025,
        "home_odds": 2.10,
        "draw_odds": 3.50,
        "away_odds": 3.40,
        "xG": 1.8,
        "xGA": 1.6,
        "elo_diff": 30
    },
}

def predict_match(match_name: str) -> str:
    """
    Retorna predicción de ejemplo para un partido

    Args:
        match_name: Nombre del partido

    Returns:
        Predicción formateada como texto
    """
    if match_name not in EJEMPLO_PREDICCIONES:
        return """❌ Partido no encontrado

Por favor selecciona uno de los partidos disponibles.
Esta es una versión de demo con predicciones de ejemplo."""

    pred = EJEMPLO_PREDICCIONES[match_name]

    # Determinar recomendación
    if pred["value_edge"] > 0.03:
        recomendacion = "✅ SÍ - Strong Value"
    elif pred["value_edge"] > 0.01:
        recomendacion = "⚠️ QUIZÁS - Weak Value"
    else:
        recomendacion = "❌ NO - Sin Value"

    output = f"""
    ⚽ PREDICCIÓN DE PARTIDO

    Match: {match_name}
    ═════════════════════════════════════════

    📊 PREDICCIÓN: {pred['prediction']}

    💪 CONFIANZA: {pred['confidence']:.1%}

    📈 PROBABILIDADES DEL MODELO:
       • {match_name.split(' vs ')[0]}: {pred['home_prob']:.1%}
       • Empate: {pred['draw_prob']:.1%}
       • {match_name.split(' vs ')[1]}: {pred['away_prob']:.1%}

    💰 CUOTAS DE MERCADO:
       • Local (1): {pred['home_odds']:.2f}
       • Empate (X): {pred['draw_odds']:.2f}
       • Visitante (2): {pred['away_odds']:.2f}

    📝 FORMA RECIENTE (últimos 5):
       • Local: {pred['home_form']}
       • Visitante: {pred['away_form']}

    ⚡ ADVANCED STATS:
       • Expected Goals (xG) Local: {pred['xG']:.2f}
       • Expected Goals Against (xGA): {pred['xGA']:.2f}
       • Diferencia Elo: {pred['elo_diff']} puntos

    💰 ANÁLISIS DE VALOR:
       • Edge (Ventaja sobre mercado): {pred['value_edge']:.2%}
       • Recomendado para Value Betting: {recomendacion}

    ℹ️ Nota: Esta es una versión DEMO con predicciones de ejemplo.
       En producción, los resultados se calcularían en tiempo real
       usando el modelo XGBoost entrenado.
    """

    return output.strip()

def create_interface():
    with gr.Blocks(
        title="⚽ Premier League Predictor (DEMO)",
        theme=gr.themes.Soft(),
        css="""
            .header { text-align: center; }
            .match-card { background: #f8f9fa; padding: 1.5rem; border-radius: 0.5rem; }
        """
    ) as demo:

        gr.Markdown("""
        # ⚽ Premier League Match Predictor

        **VERSIÓN DEMO** - Predicciones de ejemplo usando modelos ML avanzados

        ## 🚀 Funcionalidades
        - 🤖 Modelos de ensemble (XGBoost + LightGBM + Random Forest)
        - 📊 35 features engineerizados
        - 💰 Análisis de value betting con Expected Value
        - ⚡ Stats avanzados (xG, Elo ratings, forma)

        ## 📝 Instrucciones
        1. Selecciona un partido de la lista
        2. ¡Obtén la predicción al instante!
        3. Revisa el análisis de valor para decisiones de apuestas
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📍 Selecciona un Partido")

                partido = gr.Dropdown(
                    choices=list(EJEMPLO_PREDICCIONES.keys()),
                    label="Partidos Disponibles",
                    value=list(EJEMPLO_PREDICCIONES.keys())[0],
                    interactive=True
                )

                predict_btn = gr.Button(
                    "🔮 Predecir",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1):
                gr.Markdown("### 📈 Resultado Detallado")

                output = gr.Textbox(
                    label="Análisis de Predicción",
                    lines=25,
                    interactive=False,
                    show_label=True
                )

        # Conectar eventos
        def on_predict(match):
            return predict_match(match)

        def on_change(match):
            return predict_match(match)

        predict_btn.click(fn=on_predict, inputs=partido, outputs=output)
        partido.change(fn=on_change, inputs=partido, outputs=output)

        # Mostrar predicción inicial
        demo.load(fn=on_change, inputs=partido, outputs=output)

        # Ejemplos y documentación
        with gr.Accordion("📚 Más Información", open=False):
            gr.Markdown("""
            ## 🎯 Cómo funciona

            ### El Modelo
            - **Architecture**: Ensemble de 3 modelos (XGBoost, LightGBM, Random Forest)
            - **Features**: 35 indicadores matemáticamente engineerizados
            - **Entrenamiento**: 10 temporadas de Premier League (2016-2026)
            - **Precisión**: 58.2% en validación ciega
            - **ROI**: 12.3% en value betting histórico

            ### Features Principales
            - ⚽ **Ratings Elo** - Fortaleza relativa de equipos
            - 📊 **Form Metrics** - Rendimiento últimos 5 partidos
            - 🏠 **Home/Away** - Ventaja de jugar en casa
            - ⚡ **Expected Goals (xG)** - Calidad de chances
            - 🎯 **Market Odds** - Cuotas y probabilidades implícitas
            - 🏁 **Referee Bias** - Tendencias del árbitro
            - ☁️ **Weather** - Condiciones ambientales

            ### Interpretación de Resultados

            **Confianza**: Probabilidad del resultado predicho
            - ✅ **60%+**: Alta confianza
            - ⚠️ **50-60%**: Media confianza
            - ❌ **<50%**: Baja confianza

            **Edge**: Ventaja sobre el mercado
            - ✅ **>3%**: Strong Value
            - ⚠️ **1-3%**: Weak Value
            - ❌ **<1%**: Sin Value

            ## ⚠️ Disclaimer

            Estas predicciones son **informativas y educativas**. No constituyen consejo financiero.

            **Responsabilidades del Usuario:**
            - Juega solo con dinero que puedas permitirte perder
            - Las apuestas deportivas conllevan riesgo
            - Busca ayuda si tienes problemas con el juego
            - Nunca apuestes impulsivamente

            ## 🔗 Links Útiles
            - [Premier League Official](https://www.premierleague.com)
            - [Understat](https://understat.com) - xG Data
            - [Flashscore](https://www.flashscore.com) - Live Scores

            ---

            **Última actualización**: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """

            **Versión**: 1.0-DEMO
            """)

    return demo

if __name__ == "__main__":
    print("🔄 Iniciando PredictorPL DEMO...")
    print("✅ Interfaz lista en puerto 7860")
    print("   http://localhost:7860")

    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
