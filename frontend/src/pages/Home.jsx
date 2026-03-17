import { useState } from 'react'
import Header from '../components/Header/Header'
import PredictionForm from '../components/PredictionForm/PredictionForm'
import ResultDisplay from '../components/ResultDisplay/ResultDisplay'
import Loading from '../components/Loading/Loading'
import ErrorMessage from '../components/ErrorMessage/ErrorMessage'
import { usePrediction } from '../hooks/usePrediction'
import './Home.css'

export const Home = () => {
  const { prediction, loading, error, predict, reset } = usePrediction()
  const [dismissedError, setDismissedError] = useState(false)

  const handlePredictionSubmit = async (matchData) => {
    setDismissedError(false)
    try {
      await predict(matchData, true)
    } catch (err) {
      console.error('Prediction error:', err)
    }
  }

  const handleReset = () => {
    reset()
    setDismissedError(false)
  }

  const handleDismissError = () => {
    setDismissedError(true)
  }

  return (
    <div className="home">
      <Header />

      <main className="home-main">
        <div className="container">
          {!dismissedError && error && (
            <ErrorMessage message={error} onDismiss={handleDismissError} />
          )}

          {!prediction && (
            <section className="form-section">
              <h2>Ingresa los equipos para obtener una predicción</h2>
              <PredictionForm onSubmit={handlePredictionSubmit} loading={loading} />
            </section>
          )}

          {loading && <Loading message="Analizando partido..." />}

          {prediction && !loading && (
            <section className="results-section">
              <ResultDisplay prediction={prediction} />

              <div className="action-buttons">
                <button className="btn-reset" onClick={handleReset}>
                  ← Nueva Predicción
                </button>
              </div>
            </section>
          )}

          {!prediction && !loading && (
            <section className="info-section">
              <h2>¿Cómo funciona?</h2>
              <div className="features-grid">
                <div className="feature-card">
                  <span className="feature-icon">🧠</span>
                  <h3>Machine Learning</h3>
                  <p>Modelos entrenados con 10+ temporadas de datos históricos</p>
                </div>

                <div className="feature-card">
                  <span className="feature-icon">📊</span>
                  <h3>56+ Indicadores</h3>
                  <p>Análisis profundo de estadísticas de equipo y jugadores</p>
                </div>

                <div className="feature-card">
                  <span className="feature-icon">💰</span>
                  <h3>Cálculo de Edge</h3>
                  <p>Comparación con cuotas del mercado para identificar valor</p>
                </div>

                <div className="feature-card">
                  <span className="feature-icon">⚡</span>
                  <h3>Predicción en Tiempo Real</h3>
                  <p>Resultados instantáneos y actualizables</p>
                </div>
              </div>
            </section>
          )}
        </div>
      </main>

      <footer className="home-footer">
        <p>© 2026 Premier League Predictor • Predicciones Educativas • ML Powered</p>
      </footer>
    </div>
  )
}

export default Home
